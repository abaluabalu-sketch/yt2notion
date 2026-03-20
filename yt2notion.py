#!/usr/bin/env python3
"""
yt2notion.py — YouTube → Notion summarizer

Usage: python yt2notion.py
Prompts for a YouTube URL, then:
  1. Fetches metadata (title, thumbnail)
  2. Gets transcript (YouTube captions or Whisper fallback)
  3. Summarizes key topics with timestamps via Claude
  4. Creates a Notion page with the results
"""

import os
import re
import sys
import json
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


# ── API clients (lazy init) ────────────────────────────────────────────────
def get_openai():
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit("Error: OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=key)


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
    """Try to fetch existing YouTube captions. Returns list of {text, start} or None.

    Strategy:
      1. youtube_transcript_api — fast, prefers manual subtitles over auto-generated
      2. yt-dlp --write-sub     — uses Chrome cookies, works for members-only videos
    """
    # ── Method 1: youtube_transcript_api ──────────────────────────────────
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        print("  Trying YouTube transcript API...")
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Prefer manually created subtitles over auto-generated
        manual = [t for t in transcript_list if not t.is_generated]
        auto   = [t for t in transcript_list if t.is_generated]
        candidates = manual + auto  # manual first

        if candidates:
            chosen = candidates[0]
            print(f"  Found {'manual' if not chosen.is_generated else 'auto-generated'} "
                  f"subtitles in {chosen.language!r} ({chosen.language_code})")
            data = chosen.fetch()
            segments = [{"text": s.text.strip(), "start": s.start} for s in data]
            print(f"  {len(segments)} segments fetched")
            return segments

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
                    "--write-sub",        # manually uploaded subtitles
                    "--write-auto-sub",   # auto-generated as fallback
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

            # Prefer manually uploaded subs (yt-dlp names them differently from .auto.)
            manual_vtt = [f for f in vtt_files if ".auto." not in f]
            chosen_vtt = (manual_vtt or vtt_files)[0]
            lang_tag = Path(chosen_vtt).stem.split(".")[-1]
            is_auto = ".auto." in chosen_vtt
            print(f"  Found {'auto-generated' if is_auto else 'manual'} subtitles "
                  f"via yt-dlp (lang: {lang_tag})")

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


def transcribe_with_whisper_local(url: str) -> list[dict]:
    """Download audio and transcribe with local Whisper model (fallback)."""
    print("  Downloading audio for local Whisper transcription...")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = _download_audio(url, tmpdir)

        # Convert to raw PCM f32le via our bundled ffmpeg, then load as numpy array.
        # This bypasses whisper's load_audio() which requires ffmpeg in PATH.
        print("  Converting audio to PCM with bundled ffmpeg...")
        import numpy as np
        pcm_result = subprocess.run(
            [FFMPEG_PATH, "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
            capture_output=True, check=True
        )
        audio_np = np.frombuffer(pcm_result.stdout, np.float32).copy()

        print("  Transcribing with local Whisper model (this may take a few minutes)...")
        try:
            import whisper
        except ImportError:
            sys.exit("Error: 'whisper' package not installed. Run: pip install openai-whisper")

        model = whisper.load_model("large-v3")
        result = model.transcribe(audio_np, verbose=False, fp16=False)

    segments = [
        {"text": seg["text"].strip(), "start": seg["start"]}
        for seg in result["segments"]
    ]
    print(f"  Whisper returned {len(segments)} segments")
    return segments


def segments_to_text(segments: list[dict]) -> str:
    """Convert transcript segments to a timestamped plain-text string for Claude."""
    lines = []
    for seg in segments:
        ts = format_timestamp(seg["start"])
        lines.append(f"[{ts}] {seg['text']}")
    return "\n".join(lines)


# ── Summarization ──────────────────────────────────────────────────────────

def summarize(timestamped_text: str) -> str:
    """Use OpenAI GPT to extract key topics with timestamps."""
    print("  Summarizing with GPT-4o-mini...")
    client = get_openai()
    prompt = (
        "You are summarizing a video transcript for a Notion page.\n"
        "The transcript below has timestamps in [MM:SS] format at the start of each segment.\n\n"
        "IMPORTANT: Detect the dominant language of the transcript and write your entire "
        "summary in that same language. For example, if the transcript is mostly Chinese, "
        "write the summary in Chinese. If mostly English, write in English.\n\n"
        "Identify 5 to 10 key topics covered in the video. For each topic:\n"
        "- Use the timestamp of when it starts\n"
        "- Write a concise 1–2 sentence description\n\n"
        "Format: one topic per line, like this:\n"
        "[MM:SS] Topic title: brief description\n\n"
        "Return ONLY the bulleted list. No preamble, no conclusion.\n\n"
        "TRANSCRIPT:\n" + timestamped_text
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def format_conversation(timestamped_text: str) -> str:
    """Use OpenAI GPT to reformat transcript as a conversation with speaker labels."""
    print("  Formatting transcript as conversation...")
    client = get_openai()
    prompt = (
        "You are reformatting a video transcript into a clean, readable conversation.\n\n"
        "Speaker labeling rules (in priority order):\n"
        "1. Use real names if they are clearly mentioned in the transcript.\n"
        "2. If names are unclear, infer roles from context:\n"
        "   - Interview or podcast → 'Host' and 'Guest'\n"
        "   - Q&A format → 'Interviewer' and 'Guest'\n"
        "   - Two hosts → 'Host 1' and 'Host 2'\n"
        "   - Lecture or course → 'Instructor'\n"
        "   - Documentary narration → 'Narrator'\n"
        "3. Only use 'Person 1' / 'Person 2' as a last resort if no role can be inferred.\n"
        "4. If there is only one speaker, output clean paragraphs with NO labels at all.\n\n"
        "Formatting rules:\n"
        "- Remove all timestamps.\n"
        "- Merge consecutive lines from the same speaker into one paragraph.\n"
        "- Keep ALL content — do NOT summarize, skip, or shorten anything.\n"
        "- Format: one speaker turn per line, like:\n"
        "  Name or Role: What they said...\n\n"
        "Return ONLY the reformatted transcript. No preamble, no explanation.\n\n"
        "TRANSCRIPT:\n" + timestamped_text
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


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
            }
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

    print("\n[1/6] Extracting video ID...")
    video_id = extract_video_id(url)
    print(f"  Video ID: {video_id}")

    print("\n[2/6] Fetching metadata...")
    meta = fetch_metadata(url)
    print(f"  Title: {meta['title']}")

    print("\n[3/6] Getting transcript...")
    segments = get_youtube_transcript(video_id, url)
    if segments is None:
        segments = transcribe_with_whisper_local(url)

    print("\n[4/6] Generating summary...")
    timestamped_text = segments_to_text(segments)
    summary = summarize(timestamped_text)
    print("\n  Key topics:")
    for line in summary.splitlines():
        print(f"    {line}")

    print("\n[5/6] Formatting transcript as conversation...")
    conversation_text = format_conversation(timestamped_text)

    print("\n[6/6] Creating Notion page...")
    page_url = create_notion_page(
        title=meta["title"],
        youtube_url=url,
        thumbnail_url=meta["thumbnail"],
        summary=summary,
        conversation_text=conversation_text,
        video_id=video_id,
    )

    print("\n" + "=" * 40)
    print("Done!")
    print(f"Notion page: {page_url}")


if __name__ == "__main__":
    main()
