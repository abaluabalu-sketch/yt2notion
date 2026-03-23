#!/usr/bin/env python3 -u
"""
yt2notion.py — YouTube → Notion summarizer

Usage: python yt2notion.py
Prompts for a YouTube URL, then:
  1. Fetches metadata (title, thumbnail, language) via yt-dlp
  2. Gets transcript via 3-tier strategy:
     a. youtube_transcript_api — prefers manual subtitles over auto-generated
     b. yt-dlp --write-sub — with Chrome cookies for members-only videos
     c. whisper.cpp large-v3 — local transcription with Metal GPU acceleration
  3. Summarizes 5-10 key topics with timestamps via GPT-4o-mini
     - Summary language matches the dominant language of the transcript
  4. Reformats transcript as a conversation via GPT-4o-mini
     - Speaker labels: real names > inferred roles > Person 1/2 > no labels
  5. Creates a structured Notion page with:
     - YouTube bookmark + thumbnail
     - Summary with clickable timestamp links (youtube.com?v=ID&t=Xs)
     - Full transcript as readable conversation

Dependencies:
  pip: openai, notion-client, python-dotenv, yt-dlp, youtube-transcript-api, imageio-ffmpeg
  system: whisper.cpp (compiled with Metal), Node.js (for yt-dlp JS challenges)

Config (.env):
  OPENAI_API_KEY=sk-proj-...
  NOTION_API_KEY=secret_...
  NOTION_DATABASE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

import os
import re
import sys
import json
import time
import tempfile
import subprocess
import textwrap
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Ensure local Node.js is on PATH (needed by yt-dlp for JS challenges) ─
_node_dir = Path.home() / ".local" / "node" / "bin"
if _node_dir.exists() and str(_node_dir) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_node_dir}:{os.environ.get('PATH', '')}"

# ── ffmpeg path (via imageio-ffmpeg for portability) ──────────────────────
def get_ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # fall back to system PATH


FFMPEG_PATH = get_ffmpeg_path()

# ── Cookies: read directly from Chrome (always fresh) ─────────────────────

def _yt_dlp_extra_args() -> list[str]:
    """Return extra yt-dlp args: browser cookies + remote EJS solver."""
    args = ["--remote-components", "ejs:github"]
    # Read cookies live from Chrome — no manual export needed
    args += ["--cookies-from-browser", "chrome"]
    return args


# ── Claude CLI (no API key needed — uses Claude Code session) ─────────────
CLAUDE_BIN = Path.home() / ".local" / "bin" / "claude"

def call_claude(prompt: str, max_tokens: int = 8192) -> str:
    """Call the claude CLI with a prompt, return the response text."""
    result = subprocess.run(
        [str(CLAUDE_BIN), "--print", "--output-format", "text"],
        input=prompt,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit(f"Error calling claude CLI: {result.stderr}")
    return result.stdout.strip()


def get_notion():
    from notion_client import Client
    token = os.getenv("NOTION_API_KEY")
    if not token:
        sys.exit("Error: NOTION_API_KEY not set in .env")
    return Client(auth=token)


# ── YouTube helpers ────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    sys.exit(f"Error: Could not extract video ID from URL: {url}")


def fetch_metadata(url: str) -> dict:
    """Fetch video title and thumbnail URL via yt-dlp."""
    print("  Fetching video metadata...")
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist",
             "--ffmpeg-location", FFMPEG_PATH,
             *_yt_dlp_extra_args(), url],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        title = data.get("title", "Untitled Video")
        # Prefer maxresdefault thumbnail, fall back to best available
        thumbnail = data.get("thumbnail") or ""
        language = data.get("language") or ""
        return {"title": title, "thumbnail": thumbnail, "language": language}
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error fetching metadata: {e.stderr}")
    except json.JSONDecodeError:
        sys.exit("Error: Could not parse yt-dlp output")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or H:MM:SS format."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def get_youtube_transcript(video_id: str, url: str) -> list[dict] | None:
    """Try to fetch manually uploaded YouTube subtitles only.

    Auto-generated subtitles are NEVER used — Whisper produces better quality.

    Strategy:
      1. youtube_transcript_api — manual subtitles only (is_generated=False)
      2. yt-dlp --write-sub     — manual subtitles only, with Chrome cookies
    Returns None if no manual subtitles found → caller falls back to Whisper.
    """
    # ── Method 1: youtube_transcript_api ──────────────────────────────────
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        print("  Trying YouTube transcript API (manual subtitles only)...")
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Manual subtitles ONLY — skip auto-generated entirely
        manual = [t for t in transcript_list if not t.is_generated]

        if manual:
            chosen = manual[0]
            print(f"  Found manual subtitles in {chosen.language!r} ({chosen.language_code})")
            data = chosen.fetch()
            segments = [{"text": s.text.strip(), "start": s.start} for s in data]
            print(f"  {len(segments)} segments fetched")
            return segments
        else:
            print("  No manual subtitles found, will use Whisper")
            return None

    except Exception as e:
        print(f"  youtube_transcript_api failed ({type(e).__name__}), trying yt-dlp subtitles...")

    # ── Method 2: yt-dlp subtitle download (handles members-only + cookies) ─
    try:
        import glob as _glob
        with tempfile.TemporaryDirectory() as tmpdir:
            out_template = str(Path(tmpdir) / "sub")
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "--skip-download",
                    "--write-sub",        # manually uploaded subtitles ONLY
                    # NOTE: --write-auto-sub intentionally omitted
                    "--sub-langs", "all",
                    "--sub-format", "vtt",
                    "--ffmpeg-location", FFMPEG_PATH,
                    *_yt_dlp_extra_args(),
                    "-o", out_template,
                    url,
                ],
                capture_output=True, text=True
            )
            # Find any downloaded .vtt file
            vtt_files = _glob.glob(str(Path(tmpdir) / "*.vtt"))
            if not vtt_files:
                print("  No subtitle files found via yt-dlp")
                return None

            # Only use manually uploaded subs — exclude any .auto. files
            manual_vtt = [f for f in vtt_files if ".auto." not in f]
            if not manual_vtt:
                print("  Only auto-generated subtitles found via yt-dlp, will use Whisper")
                return None

            chosen_vtt = manual_vtt[0]
            lang_tag = Path(chosen_vtt).stem.split(".")[-1]
            print(f"  Found manual subtitles via yt-dlp (lang: {lang_tag})")

            segments = _parse_vtt(chosen_vtt)
            print(f"  {len(segments)} segments parsed from subtitle file")
            return segments

    except Exception as e:
        print(f"  yt-dlp subtitle download failed ({type(e).__name__}): {e}")

    return None


def _parse_vtt(vtt_path: str) -> list[dict]:
    """Parse a WebVTT subtitle file into {text, start} segments."""
    import re as _re
    segments = []
    seen_texts = set()  # deduplicate overlapping cues (common in auto-subs)

    time_re = _re.compile(
        r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
    )
    with open(vtt_path, encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        time_line = None
        text_lines = []
        for line in lines:
            if time_re.match(line):
                time_line = line
            elif time_line and line and not line.startswith("WEBVTT") and not line.isdigit():
                # Strip VTT tags like <c>, <00:00:00.000>, </c>
                cleaned = _re.sub(r"<[^>]+>", "", line).strip()
                if cleaned:
                    text_lines.append(cleaned)

        if time_line and text_lines:
            m = time_re.match(time_line)
            start = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 +
                     int(m.group(3)) + int(m.group(4)) / 1000)
            text = " ".join(text_lines)
            if text not in seen_texts:
                seen_texts.add(text)
                segments.append({"text": text, "start": start})

    return segments


def _download_audio(url: str, tmpdir: str) -> Path:
    """Download audio from YouTube URL into tmpdir, return file path."""
    audio_path = Path(tmpdir) / "audio.m4a"
    try:
        subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "-f", "bestaudio[ext=m4a]/bestaudio",
                "--ffmpeg-location", FFMPEG_PATH,
                *_yt_dlp_extra_args(),
                "-o", str(audio_path),
                url,
            ],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error downloading audio: {e.stderr}")
    return audio_path


# ── whisper.cpp paths ─────────────────────────────────────────────────────
WHISPER_CPP_BIN   = Path.home() / ".local" / "whisper-cpp" / "whisper-cli"
WHISPER_CPP_MODEL = Path.home() / ".local" / "whisper-cpp" / "models" / "ggml-large-v3-turbo.bin"

def _parse_whisper_json(json_path: Path) -> list[dict]:
    """Parse whisper.cpp JSON output into {text, start} segments.

    Also strips trailing hallucination loops — whisper.cpp sometimes repeats
    the same phrase dozens of times at the end of audio (silence/music).
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = []
    for item in data.get("transcription", []):
        text = item.get("text", "").strip()
        ts_from = item.get("timestamps", {}).get("from", "00:00:00.000")
        ts_to = item.get("timestamps", {}).get("to", ts_from)
        parts_from = ts_from.replace(",", ".").split(":")
        parts_to = ts_to.replace(",", ".").split(":")
        start = float(parts_from[0]) * 3600 + float(parts_from[1]) * 60 + float(parts_from[2])
        end = float(parts_to[0]) * 3600 + float(parts_to[1]) * 60 + float(parts_to[2])
        if text:
            segments.append({"text": text, "start": start, "end": end})

    # ── Remove trailing hallucination loops ──────────────────────────────
    # If the last N segments are all the same text, it's a whisper hallucination.
    # Walk backwards and strip repeated trailing phrases.
    if len(segments) > 5:
        segments = _strip_trailing_hallucinations(segments)

    return segments


def _strip_trailing_hallucinations(segments: list[dict]) -> list[dict]:
    """Remove repeated segments from the tail of whisper output.

    Detects when the same phrase (or very similar) repeats 3+ times
    at the end — a common whisper.cpp artifact on silence/music.
    Also removes interior runs of 3+ identical consecutive segments.
    """
    # Normalize text for comparison
    def norm(t: str) -> str:
        return re.sub(r'\s+', ' ', t.strip().lower())

    # 1. Strip trailing repetitions
    if len(segments) >= 3:
        tail_text = norm(segments[-1]["text"])
        # Count how many trailing segments match the last one
        repeat_count = 0
        for seg in reversed(segments):
            if norm(seg["text"]) == tail_text:
                repeat_count += 1
            else:
                break
        if repeat_count >= 3:
            print(f"  Stripped {repeat_count} trailing hallucinated segments "
                  f"(\"{segments[-1]['text'][:60]}...\")")
            segments = segments[:-repeat_count]

    # 2. Remove interior runs of 3+ identical consecutive segments
    cleaned = []
    run_count = 1
    for i, seg in enumerate(segments):
        if i > 0 and norm(seg["text"]) == norm(segments[i - 1]["text"]):
            run_count += 1
        else:
            run_count = 1
        if run_count <= 2:  # keep at most 2 consecutive identical segments
            cleaned.append(seg)

    if len(cleaned) < len(segments):
        print(f"  Removed {len(segments) - len(cleaned)} duplicate interior segments")

    return cleaned




def transcribe_with_whisper_local(url: str) -> list[dict]:
    """Download audio and transcribe with whisper.cpp (Metal GPU accelerated)."""
    if not WHISPER_CPP_BIN.exists():
        sys.exit(f"Error: whisper.cpp not found at {WHISPER_CPP_BIN}")
    if not WHISPER_CPP_MODEL.exists():
        sys.exit(f"Error: Whisper model not found at {WHISPER_CPP_MODEL}")

    print("  Downloading audio for whisper.cpp transcription...")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = _download_audio(url, tmpdir)

        # Convert to 16kHz mono WAV (required by whisper.cpp)
        wav_path = Path(tmpdir) / "audio.wav"
        print("  Converting audio to 16kHz WAV...")
        subprocess.run(
            [FFMPEG_PATH, "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
            capture_output=True, check=True
        )

        # ── Transcription: whisper.cpp large-v3 ─────────────────────────────
        print("  Transcribing with large-v3-turbo (Metal GPU)...")
        r1 = subprocess.run(
            [str(WHISPER_CPP_BIN),
             "-m", str(WHISPER_CPP_MODEL),
             "-f", str(wav_path),
             "-l", "auto",
             "--output-json",
             "-of", str(Path(tmpdir) / "main_out")],
            capture_output=True, text=True
        )
        if r1.returncode != 0:
            sys.exit(f"Error: whisper.cpp failed:\n{r1.stderr}")

        json_path = Path(tmpdir) / "main_out.json"
        if not json_path.exists():
            sys.exit(f"Error: whisper.cpp produced no JSON.\nstderr: {r1.stderr}")
        segments = _parse_whisper_json(json_path)

    print(f"  whisper.cpp returned {len(segments)} segments")
    return segments


def segments_to_text(segments: list[dict]) -> str:
    """Convert transcript segments to timestamped text for Claude."""
    lines = []
    for seg in segments:
        ts = format_timestamp(seg["start"])
        lines.append(f"[{ts}] {seg['text']}")
    return "\n".join(lines)


# ── Summarization ──────────────────────────────────────────────────────────

def summarize(timestamped_text: str) -> str:
    """Use Claude CLI to extract key topics with timestamps."""
    print("  Summarizing with Claude...")
    prompt = (
        "You are summarizing a video transcript for a Notion page.\n"
        "The transcript below has timestamps in [MM:SS] format at the start of each segment.\n\n"
        "IMPORTANT: Always write the summary in Traditional Chinese (繁體中文). "
        "Keep technical terms, brand names, product names, and proper nouns in their "
        "original language (e.g. Claude Code, Anthropic, API, Bash, GitHub, MCP, SDK). "
        "Everything else should be in natural, fluent Traditional Chinese.\n\n"
        "Identify 5 to 10 key topics covered in the video. For each topic:\n"
        "- Use the timestamp of when it starts\n"
        "- Write a concise 1–2 sentence description\n\n"
        "Format: one topic per line, like this:\n"
        "[MM:SS] Topic title: brief description\n\n"
        "Return ONLY the bulleted list. No preamble, no conclusion.\n\n"
        "TRANSCRIPT:\n" + timestamped_text
    )
    return call_claude(prompt, max_tokens=1024)


def _call_claude_async(prompt: str) -> subprocess.Popen:
    """Launch a Claude CLI call as a non-blocking subprocess."""
    return subprocess.Popen(
        [str(CLAUDE_BIN), "--print", "--output-format", "text"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )


def _format_chunk(chunk_text: str, speaker_context: str, chunk_idx: int) -> str:
    """Send one chunk to Claude CLI for formatting + cleaning. Blocking call."""
    prompt = (
        "You are a professional transcript editor. Format this raw transcript chunk "
        "as a clean, readable conversation.\n\n"
        f"{speaker_context}"
        "FORMATTING RULES:\n"
        "1. Label each speaker turn with their name followed by colon\n"
        "2. Merge consecutive sentences from the same speaker into one paragraph\n"
        "3. Use blank lines between speaker turns\n"
        "4. Continue from where the previous chunk left off — the last speaker "
        "may still be talking\n\n"
        "CLEANING RULES (apply these for readability):\n"
        "- Remove filler words: \"um\", \"uh\", \"like\" (unless characterizing)\n"
        "- Collapse false starts: \"I— I think—\" → \"I think\"\n"
        "- Remove repeated words: \"it's it's really\" → \"it's really\"\n"
        "- Use em-dash (—) for mid-sentence interruptions\n"
        "- Mark unclear audio as [inaudible]\n"
        "- Mark overlapping speech as [crosstalk]\n"
        "- Fix obvious grammar from speech-to-text errors\n"
        "- Keep technical terms, proper nouns, and numbers accurate\n\n"
        "OUTPUT FORMAT:\n"
        "Speaker Name: Their cleaned words here in one paragraph.\n\n"
        "Another Speaker: Their response here.\n\n"
        "Output ONLY the formatted conversation. No preamble, no commentary.\n\n"
        "TRANSCRIPT CHUNK:\n" + chunk_text
    )
    result = subprocess.run(
        [str(CLAUDE_BIN), "--print", "--output-format", "text"],
        input=prompt, capture_output=True, text=True
    )
    if result.returncode != 0:
        return chunk_text  # fallback to raw text
    return result.stdout.strip()


def format_conversation(segments: list[dict]) -> str:
    """Detect speakers, clean transcript, and format as conversation.

    Three-phase approach:
      Phase 1: Identify speakers from a sample (~15s)
      Phase 2: Split transcript into ~10K char chunks
      Phase 3: Format chunks in parallel via concurrent Claude CLI calls (~30-60s)
    """
    print("  Detecting speakers & formatting conversation (Claude)...")
    raw_text = " ".join(seg["text"] for seg in segments)

    # ── Phase 1: Identify speakers from sample ───────────────────────────
    sample_size = min(8000, len(raw_text))
    sample = raw_text[:sample_size]

    analysis_prompt = (
        "You are analyzing a video transcript to identify speakers.\n\n"
        "ANALYZE this transcript sample and return a JSON object with:\n"
        "1. \"speakers\": list of speaker objects, each with:\n"
        "   - \"name\": real name if mentioned, otherwise Host/Guest 1/Guest 2\n"
        "   - \"role\": brief description (e.g. \"interviewer\", \"Anthropic engineer\")\n"
        "   - \"style\": speaking pattern hints (e.g. \"asks questions\", \"technical details\")\n"
        "2. \"speaker_count\": number of distinct speakers\n\n"
        "Return ONLY valid JSON. No markdown fences, no commentary.\n\n"
        "TRANSCRIPT SAMPLE:\n" + sample
    )

    speaker_context = ""
    try:
        analysis = call_claude(analysis_prompt, max_tokens=1024)
        analysis = analysis.strip()
        if analysis.startswith("```"):
            analysis = re.sub(r"^```\w*\n?", "", analysis)
            analysis = re.sub(r"\n?```$", "", analysis)
        speaker_info = json.loads(analysis)
        speakers = speaker_info.get("speakers", [])
        if speakers:
            names = [s.get("name", "Unknown") for s in speakers]
            print(f"  Identified {len(speakers)} speakers: {', '.join(names)}")
            speaker_context = "IDENTIFIED SPEAKERS:\n"
            for s in speakers:
                speaker_context += (
                    f"- {s.get('name', '?')}: {s.get('role', '')} "
                    f"({s.get('style', '')})\n"
                )
            speaker_context += "\n"
    except Exception as e:
        print(f"  Speaker analysis failed ({e}), continuing without speaker hints")

    # ── Phase 2: Split into chunks (~10K chars each) ─────────────────────
    chunk_size = 10000
    chunks = []
    for i in range(0, len(raw_text), chunk_size):
        end = min(i + chunk_size, len(raw_text))
        # Try to break at a sentence boundary
        if end < len(raw_text):
            last_period = raw_text.rfind(". ", i + chunk_size - 500, end)
            if last_period > i:
                end = last_period + 2
        chunks.append(raw_text[i:end])

    print(f"  Split into {len(chunks)} chunks, formatting in parallel...")

    # ── Phase 3: Format chunks in parallel ───────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def format_one(args):
        idx, chunk = args
        return idx, _format_chunk(chunk, speaker_context, idx)

    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
        futures = {pool.submit(format_one, (i, c)): i for i, c in enumerate(chunks)}
        for future in as_completed(futures):
            idx, formatted = future.result()
            results[idx] = formatted
            print(f"  Chunk {idx + 1}/{len(chunks)} done")

    # Merge results
    merged = "\n\n".join(r for r in results if r)

    # Count speakers in output
    speakers = set()
    for line in merged.split("\n"):
        if ":" in line and not line.startswith(" "):
            speaker = line.split(":")[0].strip()
            if speaker and len(speaker) < 40 and not any(c.isdigit() for c in speaker):
                speakers.add(speaker)
    if speakers:
        print(f"  Formatted with {len(speakers)} speakers: {', '.join(sorted(speakers))}")

    return merged


# ── Notion helpers ─────────────────────────────────────────────────────────

def chunk_text(text: str, max_len: int = 1900) -> list[str]:
    """Split text into chunks that fit within Notion's paragraph block limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try to split at a sentence boundary
        split_at = text.rfind(". ", 0, max_len)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_len)
        if split_at == -1:
            split_at = max_len
        else:
            split_at += 1  # include the period
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    return chunks


def paragraph_block(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        },
    }


def heading_block(text: str, level: int = 2) -> dict:
    h = f"heading_{level}"
    return {
        "object": "block",
        "type": h,
        h: {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        },
    }


def bullet_block(text: str) -> dict:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        },
    }


def timestamp_to_seconds(ts: str) -> int:
    """Convert MM:SS or H:MM:SS string to total seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0


def bullet_block_with_timestamp_link(text: str, video_id: str) -> dict:
    """Create a bullet block where [MM:SS] timestamps are clickable YouTube links."""
    match = re.match(r"^\[(\d+:\d{2}(?::\d{2})?)\]\s*(.*)", text)
    if not match:
        return bullet_block(text)

    ts_str = match.group(1)
    rest = match.group(2)
    seconds = timestamp_to_seconds(ts_str)
    yt_link = f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"

    rich_text = [
        {
            "type": "text",
            "text": {"content": f"[{ts_str}]", "link": {"url": yt_link}},
            "annotations": {"bold": True, "color": "blue"},
        },
        {
            "type": "text",
            "text": {"content": f" {rest}"},
        },
    ]
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": rich_text},
    }


def divider_block() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def bookmark_block(url: str) -> dict:
    return {
        "object": "block",
        "type": "bookmark",
        "bookmark": {"url": url},
    }


def image_block(url: str) -> dict:
    return {
        "object": "block",
        "type": "image",
        "image": {"type": "external", "external": {"url": url}},
    }


def create_notion_page(
    title: str,
    youtube_url: str,
    thumbnail_url: str,
    summary: str,
    conversation_text: str,
    video_id: str = "",
) -> str:
    """Create the Notion page and return its URL."""
    notion = get_notion()
    parent_id = os.getenv("NOTION_DATABASE_ID")
    if not parent_id:
        sys.exit("Error: NOTION_DATABASE_ID not set in .env")

    # Build summary bullet blocks (timestamps are clickable YouTube links)
    summary_bullets = []
    for line in summary.splitlines():
        line = line.strip().lstrip("•-* ")
        if line:
            summary_bullets.append(bullet_block_with_timestamp_link(line, video_id))

    # Build transcript paragraph blocks (chunked)
    transcript_blocks = [paragraph_block(chunk) for chunk in chunk_text(conversation_text)]

    # Compose all blocks
    blocks: list[dict] = []

    # YouTube bookmark
    blocks.append(bookmark_block(youtube_url))

    # Thumbnail
    if thumbnail_url:
        blocks.append(image_block(thumbnail_url))

    blocks.append(divider_block())

    # Summary section
    blocks.append(heading_block("Summary"))
    blocks.extend(summary_bullets)

    blocks.append(divider_block())

    # Full transcript section
    blocks.append(heading_block("Full Transcript"))
    blocks.extend(transcript_blocks)

    print("  Creating Notion page...")

    # Notion API allows max 100 blocks per request; split into batches
    page = notion.pages.create(
        parent={"type": "database_id", "database_id": parent_id},
        properties={
            "title": {
                "title": [{"type": "text", "text": {"content": title}}]
            },
            "URL": {
                "url": youtube_url
            },
        },
        children=blocks[:100],
    )
    page_id = page["id"]
    page_url = page["url"]

    # Append remaining blocks in batches of 100
    remaining = blocks[100:]
    while remaining:
        notion.blocks.children.append(
            block_id=page_id,
            children=remaining[:100],
        )
        remaining = remaining[100:]

    return page_url


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("YouTube → Notion Summarizer")
    print("=" * 40)

    url = input("Paste YouTube URL: ").strip()
    if not url:
        sys.exit("No URL provided.")

    # Normalize URL
    if not url.startswith("http"):
        url = "https://" + url

    total_start = time.time()
    timings = {}

    t0 = time.time()
    print("\n[1/5] Extracting video ID...")
    video_id = extract_video_id(url)
    print(f"  Video ID: {video_id}")
    timings["1. Extract ID"] = time.time() - t0

    t0 = time.time()
    print("\n[2/5] Fetching metadata...")
    meta = fetch_metadata(url)
    print(f"  Title: {meta['title']}")
    timings["2. Metadata"] = time.time() - t0

    t0 = time.time()
    print("\n[3/5] Getting transcript...")
    segments = get_youtube_transcript(video_id, url)
    if segments is None:
        segments = transcribe_with_whisper_local(url)
    timings["3. Transcript"] = time.time() - t0

    t0 = time.time()
    print("\n[4/5] Generating summary...")
    timestamped_text = segments_to_text(segments)
    summary = summarize(timestamped_text)
    print("\n  Key topics:")
    for line in summary.splitlines():
        print(f"    {line}")
    timings["4. Summary (Claude)"] = time.time() - t0

    t0 = time.time()
    # Detect speakers and format as conversation via Claude
    conversation_text = format_conversation(segments)
    timings["4b. Conversation (Claude)"] = time.time() - t0

    t0 = time.time()
    print("\n[5/5] Creating Notion page...")
    page_url = create_notion_page(
        title=meta["title"],
        youtube_url=url,
        thumbnail_url=meta["thumbnail"],
        summary=summary,
        conversation_text=conversation_text,
        video_id=video_id,
    )
    timings["5. Notion upload"] = time.time() - t0

    total = time.time() - total_start
    print("\n" + "=" * 40)
    print("Done!")
    print(f"Notion page: {page_url}")
    print(f"\n⏱  Timing breakdown:")
    for step, secs in timings.items():
        print(f"  {step}: {secs:.1f}s")
    print(f"  {'─' * 30}")
    print(f"  TOTAL: {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
