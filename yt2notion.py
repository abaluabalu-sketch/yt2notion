#!/usr/bin/env python3 -u
"""
yt2notion.py — YouTube → Notion summarizer

Usage: python yt2notion.py [URL] [--no-frames]
Prompts for a YouTube URL (or reads it from argv/stdin), then:
  1. Fetches metadata (title, thumbnail, language) via yt-dlp
  2. Gets transcript via 3-tier strategy:
     a. youtube_transcript_api — prefers manual subtitles over auto-generated
     b. yt-dlp --write-sub — with Chrome cookies for members-only videos
     c. whisper.cpp large-v3 — local transcription with Metal GPU acceleration
  3. Watches the video (frame analysis; skip with --no-frames or YT2NOTION_NO_FRAMES=1):
     a. ffmpeg scene detection extracts keyframes where slides flip
     b. Apple Vision OCR reads slide text locally (free)
     c. Sparse-text frames (charts/diagrams) described by Claude CLI,
        gpt-4o-mini vision as fallback
  4. Summarizes 5-10 key topics with timestamps (transcript + slide notes)
     - Summary language matches the dominant language of the transcript
  5. Reformats transcript as a conversation via Claude CLI
     - Speaker labels: real names > inferred roles > Person 1/2 > no labels
  6. Creates a structured Notion page with:
     - YouTube bookmark + thumbnail
     - Summary with clickable timestamp links (youtube.com?v=ID&t=Xs)
     - 投影片重點: embedded slide images (Notion File Upload API) + timestamped notes
     - Full transcript as readable conversation

Dependencies:
  pip: openai, notion-client, python-dotenv, yt-dlp, youtube-transcript-api,
       imageio-ffmpeg, pyobjc-framework-Vision, pyobjc-framework-Quartz
  system: whisper.cpp (compiled with Metal), Node.js (for yt-dlp JS challenges)

Config (.env):
  OPENAI_API_KEY=sk-proj-...
  NOTION_API_KEY=secret_...
  NOTION_DATABASE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import base64
import difflib
import argparse
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

# ── Ensure user bin dirs are on PATH (yt-dlp console script) ──────────────
# Prepend unconditionally so priority is deterministic regardless of the
# caller's PATH: ~/.local/bin (uv-installed, current yt-dlp) wins over the
# pip user-scripts dir (Python 3.9, frozen at an old yt-dlp).
import site
for _bin_dir in (Path("/opt/homebrew/bin"),          # deno (yt-dlp EJS runtime)
                 Path(site.USER_BASE) / "bin",       # pip user scripts
                 Path.home() / ".local" / "bin"):    # uv-installed yt-dlp
    if _bin_dir.exists():
        os.environ["PATH"] = f"{_bin_dir}:{os.environ.get('PATH', '')}"

# ── ffmpeg path (via imageio-ffmpeg for portability) ──────────────────────
def get_ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # fall back to system PATH


FFMPEG_PATH = get_ffmpeg_path()

# ── Cookies: read directly from Chrome (always fresh) ─────────────────────

_YT_DLP_HAS_REMOTE_COMPONENTS: bool | None = None

def _yt_dlp_extra_args() -> list[str]:
    """Return extra yt-dlp args: browser cookies + remote EJS solver."""
    global _YT_DLP_HAS_REMOTE_COMPONENTS
    args = []
    # --remote-components only exists in recent yt-dlp; probe once
    if _YT_DLP_HAS_REMOTE_COMPONENTS is None:
        try:
            help_text = subprocess.run(
                ["yt-dlp", "--help"], capture_output=True, text=True
            ).stdout
            _YT_DLP_HAS_REMOTE_COMPONENTS = "--remote-components" in help_text
        except FileNotFoundError:
            _YT_DLP_HAS_REMOTE_COMPONENTS = False
    if _YT_DLP_HAS_REMOTE_COMPONENTS:
        args += ["--remote-components", "ejs:github"]
    # Read cookies live from Chrome — no manual export needed
    args += ["--cookies-from-browser", "chrome"]
    return args


# ── Claude CLI (no API key needed — uses Claude Code session) ─────────────
CLAUDE_BIN = Path.home() / ".local" / "bin" / "claude"

def call_claude(prompt: str, max_tokens: int = 8192) -> str:
    """Call the claude CLI with a prompt, return the response text.

    Falls back to OpenAI API (gpt-4o-mini) if the Claude CLI is unavailable
    (e.g. when running inside a Claude Code session — 'Auto mode temporarily unavailable').
    """
    # Try Claude CLI first
    result = subprocess.run(
        [str(CLAUDE_BIN), "--print", "--output-format", "text"],
        input=prompt,
        capture_output=True, text=True
    )
    if result.returncode == 0:
        output = result.stdout.strip()
        if output and "auto mode temporarily unavailable" not in output.lower():
            return output

    # Fall back to OpenAI API
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        sys.exit("Error: Claude CLI unavailable and OPENAI_API_KEY not set in .env")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        sys.exit(f"Error calling OpenAI API fallback: {e}")


def get_notion():
    from notion_client import Client
    token = os.getenv("NOTION_API_KEY")
    if not token:
        sys.exit("Error: NOTION_API_KEY not set in .env")
    return Client(auth=token)


# ── YouTube helpers ────────────────────────────────────────────────────────

def detect_platform(url: str) -> str:
    """Return 'vimeo' or 'youtube' (default) based on URL domain."""
    if re.search(r"vimeo\.com", url, re.IGNORECASE):
        return "vimeo"
    return "youtube"


def extract_youtube_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    sys.exit(f"Error: Could not extract YouTube video ID from URL: {url}")


def extract_vimeo_id(url: str) -> str:
    """Extract Vimeo video ID from URL formats like vimeo.com/123456789."""
    match = re.search(r"vimeo\.com/(\d+)", url)
    if match:
        return match.group(1)
    sys.exit(f"Error: Could not extract Vimeo video ID from URL: {url}")


def extract_video_id(url: str, platform: str = "youtube") -> str:
    """Dispatch to platform-specific video ID extractor."""
    if platform == "vimeo":
        return extract_vimeo_id(url)
    return extract_youtube_id(url)


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


def get_vimeo_transcript(url: str) -> list[dict] | None:
    """Try to fetch subtitles for a Vimeo video via yt-dlp (Tier 2 only).

    Vimeo has no transcript API equivalent to YouTubeTranscriptApi, so we
    go straight to yt-dlp --write-sub. Returns None → caller falls back to Whisper.
    """
    import glob as _glob
    print("  Trying yt-dlp subtitles for Vimeo...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_template = str(Path(tmpdir) / "sub")
            subprocess.run(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "--skip-download",
                    "--write-sub",
                    "--sub-langs", "all",
                    "--sub-format", "vtt",
                    "--ffmpeg-location", FFMPEG_PATH,
                    *_yt_dlp_extra_args(),
                    "-o", out_template,
                    url,
                ],
                capture_output=True, text=True
            )
            vtt_files = _glob.glob(str(Path(tmpdir) / "*.vtt"))
            if not vtt_files:
                print("  No Vimeo subtitle files found, will use Whisper")
                return None
            chosen_vtt = vtt_files[0]
            lang_tag = Path(chosen_vtt).stem.split(".")[-1]
            print(f"  Found Vimeo subtitles via yt-dlp (lang: {lang_tag})")
            segments = _parse_vtt(chosen_vtt)
            print(f"  {len(segments)} segments parsed from Vimeo subtitle file")
            return segments
    except Exception as e:
        print(f"  Vimeo subtitle fetch failed ({type(e).__name__}): {e}")
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


# ── Frame analysis (watch the video, not just listen) ─────────────────────
#
# Strategy: ffmpeg scene detection extracts only the frames where the picture
# changes (slide flips); Apple Vision OCR reads them locally for free; only
# frames with little text (charts / diagrams / demos) go to Claude CLI, with
# gpt-4o-mini vision as fallback. All failures degrade to the audio-only page.

FRAME_SAMPLE_FPS      = 1.0    # sampling rate for the visual-state pass
FRAME_PIXEL_DELTA     = 24     # gray-level change for a pixel to count as changed
FRAME_SOFT_CHANGE_RATIO = 0.04 # ≤ this frame-to-frame ratio = quiet (element settling)
FRAME_SPLIT_DRIFT_RATIO = 0.10 # > this drift from segment start = new visual state
FRAME_CAPTURE_BACKOFF = 1.0    # capture at least this long before the segment boundary
FRAME_DEDUP_RATIO     = 0.10   # ≤ this ratio between captured frames = re-shown slide
FRAME_MIN_STABLE_SECONDS = 5.0 # a visual state must persist this long to count
FRAME_TOPIC_MAX_SPAN_SECONDS = 90  # topic → state matching: search from topic start to next topic, capped here
FRAME_MAX_KEYFRAMES   = 40     # hard cap on analyzed keyframes per video
FRAME_MAX_VISION      = 10     # max frames sent to Claude/GPT vision per video
FRAME_VISION_BATCH    = 5      # images per vision call
FRAME_SPARSE_OCR_CHARS = 40    # OCR shorter than this → frame needs vision
FRAME_MAX_EMBEDS      = 25     # max slide images embedded in the Notion page


def _download_video(url: str, tmpdir: str) -> Path:
    """Download a video-only stream (≤720p) for frame extraction."""
    video_path = Path(tmpdir) / "video.mp4"
    subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "-f", "bv*[height<=720][ext=mp4]/bv*[height<=720]/bv*",
            "--ffmpeg-location", FFMPEG_PATH,
            *_yt_dlp_extra_args(),
            "-o", str(video_path),
            url,
        ],
        capture_output=True, text=True, check=True
    )
    if not video_path.exists():
        # yt-dlp may pick a non-mp4 container despite -o; grab whatever it wrote
        candidates = [p for p in Path(tmpdir).iterdir()
                      if p.stem == "video" and p.is_file()]
        if not candidates:
            raise RuntimeError("yt-dlp produced no video file")
        video_path = candidates[0]
    return video_path


# Thumbnail geometry for state comparison: 32x18 grayscale (576 bytes).
# Big enough to distinguish slides that share a template, small enough to
# blur away webcam picture-in-picture corners and cursor movement.
_THUMB_W, _THUMB_H = 32, 18


def _changed_ratio(a: bytes, b: bytes) -> float:
    """Fraction of pixels that shifted by more than FRAME_PIXEL_DELTA gray levels."""
    changed = sum(1 for x, y in zip(a, b) if abs(x - y) > FRAME_PIXEL_DELTA)
    return changed / len(a)


def _sample_thumbs(video_path: Path) -> list[bytes]:
    """One ffmpeg pass: FRAME_SAMPLE_FPS grayscale thumbnails as raw bytes.

    Frame i corresponds to t ≈ i / FRAME_SAMPLE_FPS seconds.
    """
    vf = f"fps={FRAME_SAMPLE_FPS},scale={_THUMB_W}:{_THUMB_H}:flags=area,format=gray"
    for hwaccel in (["-hwaccel", "videotoolbox"], []):
        result = subprocess.run(
            [FFMPEG_PATH, *hwaccel, "-i", str(video_path),
             "-vf", vf, "-f", "rawvideo", "-"],
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout:
            break
    else:
        raise RuntimeError(
            f"ffmpeg thumb sampling failed: {result.stderr[-500:].decode(errors='replace')}")

    size = _THUMB_W * _THUMB_H
    raw = result.stdout
    return [raw[i:i + size] for i in range(0, len(raw) - size + 1, size)]


def _find_stable_states(thumbs: list[bytes]) -> list[dict]:
    """Group 1fps thumbnails into visual-state segments.

    A segment closes when the picture drifts materially from the segment's
    first frame. Each build stage of a progressively-built slide becomes its
    own segment, captured at the last QUIET moment before its boundary (the
    stage with all its elements so far); `_dedup_slides` later collapses the
    stage chain into the single completed slide via OCR-subset merging.
    Returns [{"start","end","capture","mid","thumb"}].
    """
    if not thumbs:
        return []
    dt = 1.0 / FRAME_SAMPLE_FPS

    segments: list[dict] = []

    def close_segment(seg_start: int, end_i: int, quiet_i: int) -> None:
        start_ts, end_ts = seg_start * dt, (end_i + 1) * dt
        capture_i = max(seg_start, quiet_i)
        capture_ts = max(start_ts, min(capture_i * dt, end_ts - FRAME_CAPTURE_BACKOFF))
        segments.append({
            "start": start_ts, "end": end_ts, "capture": capture_ts,
            "mid": (start_ts + end_ts) / 2, "thumb": thumbs[capture_i],
        })

    anchor, seg_start = thumbs[0], 0
    seg_last_quiet = 0
    for i in range(1, len(thumbs)):
        f = thumbs[i]
        if _changed_ratio(f, thumbs[i - 1]) <= FRAME_SOFT_CHANGE_RATIO:
            seg_last_quiet = i
        if _changed_ratio(f, anchor) > FRAME_SPLIT_DRIFT_RATIO:
            # Picture moved on (slide flip, cut, or next build stage) —
            # close at the last quiet frame so mid-transition blur is skipped
            close_segment(seg_start, i - 1, seg_last_quiet)
            anchor, seg_start = f, i
            seg_last_quiet = i
    close_segment(seg_start, len(thumbs) - 1, seg_last_quiet)

    states = [s for s in segments if s["end"] - s["start"] >= FRAME_MIN_STABLE_SECONDS]
    if not states:  # short/dynamic video — keep the longest segments anyway
        states = sorted(segments, key=lambda s: s["start"] - s["end"])[:3]
        states.sort(key=lambda s: s["start"])

    # Global dedup: a segment visually identical to an earlier one is a
    # re-shown slide — keep the first occurrence only
    kept: list[dict] = []
    for s in states:
        if all(_changed_ratio(s["thumb"], k["thumb"]) > FRAME_DEDUP_RATIO
               for k in kept):
            kept.append(s)

    # Cap: subsample evenly, always keeping first and last
    if len(kept) > FRAME_MAX_KEYFRAMES:
        step = (len(kept) - 1) / (FRAME_MAX_KEYFRAMES - 1)
        indices = sorted({round(i * step) for i in range(FRAME_MAX_KEYFRAMES)})
        kept = [kept[i] for i in indices]
    return kept


def _extract_frame_at(video_path: Path, ts: float, out_path: Path) -> bool:
    """Extract one full-res frame at ts (fast input seek); True on success."""
    result = subprocess.run(
        [FFMPEG_PATH, "-ss", f"{ts:.2f}", "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", "-y", str(out_path)],
        capture_output=True,
    )
    return result.returncode == 0 and out_path.exists()


def _extract_keyframes(
    video_path: Path, frames_dir: Path, ocr_langs: list[str] | None = None,
) -> list[tuple[Path, float, dict, str]]:
    """Completed-slide keyframes: [(jpg_path, start_timestamp, state, ocr_text)].

    Each segment's frame is captured at its END (`state["capture"]`) — the
    moment the author has added the last element — not the middle, which for
    progressively-built slides is half-constructed. Safety net: if the end
    capture OCRs much poorer than the segment's mid frame (caught a fade-out),
    fall back to the mid frame. The returned timestamp is the segment START so
    the Notion timestamp link jumps to where the slide begins.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    states = _find_stable_states(_sample_thumbs(video_path))
    kept: list[tuple[Path, float, dict, str]] = []
    for i, state in enumerate(states):
        out = frames_dir / f"{i + 1:04d}.jpg"
        if not _extract_frame_at(video_path, state["capture"], out):
            continue
        text = _ocr_image(out, ocr_langs)
        # Fallback: end capture caught a transition → mid frame reads richer
        if abs(state["capture"] - state["mid"]) > 1.5:
            alt = frames_dir / f"{i + 1:04d}_mid.jpg"
            if _extract_frame_at(video_path, state["mid"], alt):
                alt_text = _ocr_image(alt, ocr_langs)
                if len(text.strip()) < 0.6 * len(alt_text.strip()):
                    alt.replace(out)
                    text = alt_text
                else:
                    alt.unlink(missing_ok=True)
        kept.append((out, state["start"], state, text))
    return kept


_OCR_UNAVAILABLE = False

def _is_cjk_heavy(text: str) -> bool:
    """True when a text sample is CJK-heavy (Chinese slides likely).

    Mirrors PDF_reader's language heuristic: >15% of alphabetic characters
    being Han ideographs means the video is Chinese, so OCR should lead with
    zh-Hant. yt-dlp's metadata `language` field is often empty, so the
    transcript (which we already have) is the more reliable signal.
    """
    sample = text[:4000]
    han = sum(1 for ch in sample if "一" <= ch <= "鿿")
    letters = sum(1 for ch in sample if ch.isalpha() or "一" <= ch <= "鿿")
    return letters > 0 and han / letters > 0.15


def _ocr_languages(video_language: str = "") -> list[str]:
    """Order Apple Vision OCR languages by the video's language.

    Language order is a strong priority hint: zh-Hant first mangles English
    slides (English words read as Chinese characters), and vice versa. Default
    English-first (most tech-talk slides are English); lead with Chinese only
    when the video is detected as Chinese-language. Accepts either a language
    code (e.g. "zh-TW") or the literal "zh".
    """
    if video_language and video_language.lower().startswith("zh"):
        return ["zh-Hant", "en-US"]
    return ["en-US", "zh-Hant"]


def _ocr_image(path: Path, languages: list[str] | None = None) -> str:
    """Read text from an image via Apple Vision OCR (local, free).

    Returns "" if pyobjc frameworks are unavailable or OCR fails.
    """
    global _OCR_UNAVAILABLE
    if _OCR_UNAVAILABLE:
        return ""
    try:
        import Quartz
        import Vision
    except ImportError:
        if not _OCR_UNAVAILABLE:
            print("  ⚠ pyobjc Vision/Quartz not installed — skipping local OCR "
                  "(pip install pyobjc-framework-Vision pyobjc-framework-Quartz)")
        _OCR_UNAVAILABLE = True
        return ""
    try:
        data = Quartz.CFDataCreate(None, path.read_bytes(), path.stat().st_size)
        source = Quartz.CGImageSourceCreateWithData(data, None)
        if source is None:
            return ""
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
        if cg_image is None:
            return ""
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setRecognitionLanguages_(languages or ["en-US", "zh-Hant"])
        request.setUsesLanguageCorrection_(True)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        ok, _err = handler.performRequests_error_([request], None)
        if not ok or not request.results():
            return ""
        lines = []
        for obs in request.results():
            candidates = obs.topCandidates_(1)
            if candidates and len(candidates):
                lines.append(candidates[0].string())
        return "\n".join(lines)
    except Exception:
        return ""


def _text_contains(small: str, big: str) -> bool:
    """True when most of `small` appears (in order) inside `big`."""
    if not small:
        return False
    sm = difflib.SequenceMatcher(None, small, big)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / len(small) >= 0.8


def _dedup_slides(slides: list[dict]) -> list[dict]:
    """Collapse build-up stages and re-detections of the same slide.

    A slide whose OCR text is contained in the NEXT slide's text is an earlier
    build stage of that slide (the author kept adding elements): keep the
    later, completed image but the earlier start timestamp — that's when the
    slide first appeared. Near-identical neighbours keep the first occurrence.
    """
    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", t.strip().lower())

    deduped: list[dict] = []
    for slide in slides:
        if deduped:
            prev, cur = norm(deduped[-1]["ocr_text"]), norm(slide["ocr_text"])
            if prev and cur and _text_contains(prev, cur):
                # Build stage (or re-detection): absorb — completed image wins,
                # slide keeps the timestamp of when it first appeared
                merged = dict(slide)
                merged["start"] = deduped[-1]["start"]
                if "state" in merged and "state" in deduped[-1]:
                    merged["state"] = {**merged["state"],
                                       "start": deduped[-1]["state"]["start"]}
                deduped[-1] = merged
                continue
            if prev and cur and difflib.SequenceMatcher(None, prev, cur).ratio() > 0.85:
                continue  # near-identical but shrinking (element removed) — keep first
        deduped.append(slide)
    if len(deduped) < len(slides):
        print(f"  Merged {len(slides) - len(deduped)} build-stage/duplicate slides (OCR)")
    return deduped


def _describe_frames_with_vision(frames: list[dict]) -> dict[Path, str]:
    """Describe visually-dense frames (charts/diagrams/demos) with a vision model.

    Primary: Claude CLI reading image files directly (subscription, no API cost).
    Fallback: gpt-4o-mini vision with base64 images.
    Returns {image_path: one-line description}.
    """
    descriptions: dict[Path, str] = {}
    if not frames:
        return descriptions

    for i in range(0, len(frames), FRAME_VISION_BATCH):
        batch = frames[i:i + FRAME_VISION_BATCH]
        listing = "\n".join(
            f"IMAGE {j + 1}: {s['image_path']}" for j, s in enumerate(batch)
        )
        prompt = (
            "You are describing video frames (screenshots) to enrich study notes.\n"
            "Read each of these image files:\n\n" + listing + "\n\n"
            "For EACH image, write exactly one line describing the key visual "
            "content — what a chart/diagram/demo/slide shows and its takeaway, "
            "not cosmetic details. Write in Traditional Chinese (繁體中文), keeping "
            "technical terms, brand names, and proper nouns in their original "
            "language.\n"
            "If an image contains NO informative content — only a talking "
            "person, a logo, a title card, or a transition — output exactly "
            "SKIP as its description.\n\n"
            "Output format — one line per image, nothing else:\n"
            "IMAGE 1: <description or SKIP>\n"
            "IMAGE 2: <description or SKIP>\n"
        )
        parsed: dict[int, str] = {}

        # Primary: Claude CLI (it can Read image files). Run with cwd set to
        # the frames dir — in --print mode the CLI refuses to Read files
        # outside its working directory.
        result = subprocess.run(
            [str(CLAUDE_BIN), "--print", "--output-format", "text"],
            input=prompt, capture_output=True, text=True,
            cwd=str(batch[0]["image_path"].parent),
        )
        if (result.returncode == 0
                and "auto mode temporarily unavailable" not in result.stdout.lower()):
            for line in result.stdout.splitlines():
                m = re.match(r"^\s*IMAGE\s*(\d+)\s*[::]\s*(.+)", line)
                if m:
                    parsed[int(m.group(1))] = m.group(2).strip()
        if not parsed:
            print(f"  ⚠ Claude vision gave no descriptions "
                  f"(output: {result.stdout.strip()[:100]!r}), trying fallback...")

        # Fallback: gpt-4o-mini vision (base64 images)
        if len(parsed) < len(batch):
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)
                    content: list[dict] = [{"type": "text", "text": prompt}]
                    for s in batch:
                        b64 = base64.b64encode(s["image_path"].read_bytes()).decode()
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        })
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": content}],
                        max_tokens=1024,
                    )
                    for line in (response.choices[0].message.content or "").splitlines():
                        m = re.match(r"^\s*IMAGE\s*(\d+)\s*[::]\s*(.+)", line)
                        if m and int(m.group(1)) not in parsed:
                            parsed[int(m.group(1))] = m.group(2).strip()
                except Exception as e:
                    print(f"  ⚠ Vision fallback failed ({e})")

        for j, s in enumerate(batch):
            if j + 1 in parsed:
                descriptions[s["image_path"]] = parsed[j + 1]

    print(f"  Visual descriptions obtained: {len(descriptions)}/{len(frames)}")
    return descriptions


def _ocr_excerpt(ocr_text: str, max_len: int = 200) -> str:
    """Condense OCR text into a one-line note (title line first)."""
    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
    return " · ".join(lines)[:max_len]


def analyze_video_frames(url: str, workdir: str, video_language: str = "") -> dict:
    """Watch the video: stable-state keyframes, OCR them, describe visuals.

    Returns a frame context:
      {"slides": [{"start","image_path","ocr_text","note","state"}],
       "states": [...], "video_path": Path, "frames_dir": Path, "ocr_langs": [...]}
    Files live under workdir — keep it alive until the Notion upload.
    """
    print("  Downloading video stream (≤720p, video-only)...")
    video_path = _download_video(url, workdir)

    ocr_langs = _ocr_languages(video_language)
    print("  Detecting slide segments (end-of-state capture @1fps, "
          f"OCR langs {ocr_langs})...")
    frames_dir = Path(workdir) / "frames"
    keyframes = _extract_keyframes(video_path, frames_dir, ocr_langs)
    ctx = {"slides": [], "states": [s for _, _, s, _ in keyframes],
           "video_path": video_path, "frames_dir": frames_dir,
           "ocr_langs": ocr_langs}
    if not keyframes:
        print("  No slide segments found — skipping visuals")
        return ctx
    print(f"  {len(keyframes)} completed slides captured")

    slides = [
        {"start": ts, "image_path": path, "ocr_text": ocr_text, "state": state}
        for path, ts, state, ocr_text in keyframes
    ]
    slides = _dedup_slides(slides)
    # After build-stage merging the slides' states carry the widened spans —
    # topic matching must see those, not the absorbed partial stages
    ctx["states"] = [s["state"] for s in slides]

    # Frames the OCR couldn't explain go to a vision model (capped)
    sparse = [s for s in slides
              if len(s["ocr_text"].strip()) < FRAME_SPARSE_OCR_CHARS]
    if len(sparse) > FRAME_MAX_VISION:
        step = len(sparse) / FRAME_MAX_VISION
        sparse = [sparse[int(i * step)] for i in range(FRAME_MAX_VISION)]
    if sparse:
        print(f"  Describing {len(sparse)} visual frames with Claude...")
    descriptions = _describe_frames_with_vision(sparse)

    skip_state_ids = set()
    for s in slides:
        desc = descriptions.get(s["image_path"], "")
        if desc.strip().upper().startswith("SKIP"):
            s["note"] = ""  # vision model judged it non-informative
            skip_state_ids.add(id(s["state"]))
        else:
            s["note"] = desc or _ocr_excerpt(s["ocr_text"])
    if skip_state_ids:
        print(f"  {len(skip_state_ids)} non-informative frames dropped (vision SKIP)")
    ctx["skip_state_ids"] = skip_state_ids
    slides = [s for s in slides if s["note"].strip()]
    print(f"  {len(slides)} slides with notes")
    ctx["slides"] = slides
    return ctx


def attach_topic_frames(ctx: dict, summary: str) -> None:
    """Layer 2: give each summary topic the frame on screen when it starts.

    Tags matching slides with topic_ts; extracts extra frames for topics whose
    state produced no slide. Mutates ctx["slides"] in place. Never raises.
    """
    try:
        states, slides = ctx.get("states", []), ctx.get("slides", [])
        video_path = ctx.get("video_path")
        if not states or not video_path or not Path(video_path).exists():
            return

        topic_ts = []
        for line in summary.splitlines():
            m = re.match(r"^\s*[•\-\*]?\s*\[(\d+:\d{2}(?::\d{2})?)\]", line.strip())
            if m:
                topic_ts.append(timestamp_to_seconds(m.group(1)))

        by_state_id = {id(s["state"]): s for s in slides if "state" in s}
        skip_ids = ctx.get("skip_state_ids", set())
        used_states: set[int] = set()
        added = 0
        video_end = max(st["end"] for st in states)
        for i, t in enumerate(topic_ts):
            # State with the longest overlap of this topic's span
            # (topic start → next topic start, capped)
            next_t = topic_ts[i + 1] if i + 1 < len(topic_ts) else video_end
            window_end = min(next_t, t + FRAME_TOPIC_MAX_SPAN_SECONDS)
            best, best_overlap = None, 0.0
            for st in states:
                overlap = min(st["end"], window_end) - max(st["start"], t)
                if overlap > best_overlap:
                    best, best_overlap = st, overlap
            if best is None or id(best) in used_states or id(best) in skip_ids:
                continue  # no visual, already used, or vision-judged junk
            used_states.add(id(best))

            slide = by_state_id.get(id(best))
            if slide is not None:
                if slide["note"]:
                    slide["topic_ts"] = t
                continue
            # State had no slide (e.g. dropped by cap) — extract its completed
            # frame now
            out = ctx["frames_dir"] / f"topic_{int(t):06d}.jpg"
            if _extract_frame_at(Path(video_path), best["capture"], out):
                ocr_text = _ocr_image(out, ctx.get("ocr_langs"))
                slides.append({
                    "start": best["start"], "image_path": out, "ocr_text": ocr_text,
                    "note": _ocr_excerpt(ocr_text), "state": best, "topic_ts": t,
                })
                added += 1
        slides.sort(key=lambda s: s["start"])
        tagged = sum(1 for s in slides if s.get("topic_ts") is not None)
        print(f"  Topic frames: {tagged}/{len(topic_ts)} topics illustrated"
              + (f" (+{added} extra frames)" if added else ""))
    except Exception as e:
        print(f"  ⚠ Topic-frame matching failed ({e}) — keeping state slides only")


def slides_to_text(slides: list[dict]) -> str:
    """Convert slide notes to timestamped text for the summary prompt."""
    lines = []
    for s in slides:
        ts = format_timestamp(s["start"])
        lines.append(f"[{ts}] {s['note'][:300]}")
    return "\n".join(lines)


# ── Summarization ──────────────────────────────────────────────────────────

def _clean_summary(text: str) -> str:
    """Strip model wrapper noise (preamble like "Here's the summary:", code
    fences) that would otherwise render as junk bullets on the Notion page.

    Keeps only '[MM:SS] ...' topic lines when any exist; otherwise just drops
    code-fence markers.
    """
    lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("```")]
    topic_lines = [l for l in lines
                   if re.match(r"^\s*[•\-\*]?\s*\[\d+:\d{2}(?::\d{2})?\]", l.strip())]
    return "\n".join(topic_lines or lines)


def summarize(timestamped_text: str, visual_text: str = "") -> str:
    """Use Claude CLI to extract key topics with timestamps."""
    print("  Summarizing with Claude...")
    visual_section = ""
    if visual_text:
        visual_section = (
            "\n\nSLIDES (on-screen visual content extracted from the video, "
            "timestamped — use it to sharpen topic titles and to catch topics "
            "shown on slides but not spoken aloud):\n" + visual_text
        )
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
        "TRANSCRIPT:\n" + timestamped_text + visual_section
    )
    return _clean_summary(call_claude(prompt, max_tokens=1024))


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


def bullet_block_with_timestamp_plain(text: str) -> dict:
    """Create a bullet block where [MM:SS] timestamps are bold plain text.

    Used for non-YouTube platforms (e.g. Vimeo) where timestamp hyperlinks
    are not standardized.
    """
    match = re.match(r"^\[(\d+:\d{2}(?::\d{2})?)\]\s*(.*)", text)
    if not match:
        return bullet_block(text)
    ts_str = match.group(1)
    rest = match.group(2)
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": f"[{ts_str}]"},
                    "annotations": {"bold": True},
                },
                {
                    "type": "text",
                    "text": {"content": f" {rest}"},
                },
            ]
        },
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


NOTION_VERSION = "2022-06-28"

def _upload_file_to_notion(path: Path) -> str | None:
    """Upload a local image via Notion's File Upload API; return file_upload id.

    Returns None on any failure (caller falls back to text-only notes).
    """
    import httpx  # transitive dep of notion-client/openai
    token = os.getenv("NOTION_API_KEY")
    if not token or path.stat().st_size > 4_500_000:  # stay under 5 MiB plan limit
        return None
    headers = {"Authorization": f"Bearer {token}", "Notion-Version": NOTION_VERSION}
    try:
        r = httpx.post(
            "https://api.notion.com/v1/file_uploads",
            headers=headers,
            json={"mode": "single_part", "filename": path.name,
                  "content_type": "image/jpeg"},
            timeout=30,
        )
        r.raise_for_status()
        upload_id = r.json()["id"]
        with open(path, "rb") as f:
            r2 = httpx.post(
                f"https://api.notion.com/v1/file_uploads/{upload_id}/send",
                headers=headers,
                files={"file": (path.name, f, "image/jpeg")},
                timeout=60,
            )
        r2.raise_for_status()
        return upload_id
    except Exception as e:
        print(f"  ⚠ Slide upload failed ({e})")
        return None


def image_block_uploaded(file_upload_id: str) -> dict:
    return {
        "object": "block",
        "type": "image",
        "image": {"type": "file_upload", "file_upload": {"id": file_upload_id}},
    }


def create_notion_page(
    title: str,
    video_url: str,
    thumbnail_url: str,
    summary: str,
    conversation_text: str,
    video_id: str = "",
    platform: str = "youtube",
    slides: list[dict] | None = None,
) -> str:
    """Create the Notion page and return its URL."""
    notion = get_notion()
    parent_id = os.getenv("NOTION_DATABASE_ID")
    if not parent_id:
        sys.exit("Error: NOTION_DATABASE_ID not set in .env")

    # Topic-tagged slides illustrate their summary bullet; the rest go to
    # the 投影片重點 section. One shared embed budget across both.
    slides = slides or []
    topic_slides = {s["topic_ts"]: s for s in slides if s.get("topic_ts") is not None}
    embeds_left = FRAME_MAX_EMBEDS
    if slides:
        print(f"  Uploading up to {min(len(slides), FRAME_MAX_EMBEDS)} slide images to Notion...")

    # Build summary bullet blocks (each followed by its topic frame, if any)
    # YouTube: timestamps are clickable links; Vimeo/other: bold plain text
    summary_bullets = []
    for line in summary.splitlines():
        line = line.strip().lstrip("•-* ")
        if not line:
            continue
        if platform == "youtube":
            summary_bullets.append(bullet_block_with_timestamp_link(line, video_id))
        else:
            summary_bullets.append(bullet_block_with_timestamp_plain(line))
        m = re.match(r"^\[(\d+:\d{2}(?::\d{2})?)\]", line)
        slide = topic_slides.get(timestamp_to_seconds(m.group(1))) if m else None
        if slide is not None and embeds_left > 0:
            upload_id = _upload_file_to_notion(slide["image_path"])
            if upload_id:
                summary_bullets.append(image_block_uploaded(upload_id))
                slide["_embedded"] = True
                embeds_left -= 1

    # Build transcript paragraph blocks (chunked)
    transcript_blocks = [paragraph_block(chunk) for chunk in chunk_text(conversation_text)]

    # Compose all blocks
    blocks: list[dict] = []

    # Video bookmark
    blocks.append(bookmark_block(video_url))

    # Thumbnail
    if thumbnail_url:
        blocks.append(image_block(thumbnail_url))

    blocks.append(divider_block())

    # Summary section
    blocks.append(heading_block("Summary"))
    blocks.extend(summary_bullets)

    # 投影片重點: informative slides not already shown under a summary topic
    leftover = [s for s in slides if not s.get("_embedded") and s["note"].strip()]
    if leftover:
        blocks.append(divider_block())
        blocks.append(heading_block("投影片重點"))
        for slide in leftover:
            ts = format_timestamp(slide["start"])
            line = f"[{ts}] {slide['note']}"
            if platform == "youtube":
                blocks.append(bullet_block_with_timestamp_link(line, video_id))
            else:
                blocks.append(bullet_block_with_timestamp_plain(line))
            if embeds_left > 0:
                upload_id = _upload_file_to_notion(slide["image_path"])
                if upload_id:
                    blocks.append(image_block_uploaded(upload_id))
                    embeds_left -= 1

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
                "url": video_url
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
    parser = argparse.ArgumentParser(description="Video → Notion Summarizer (YouTube + Vimeo)")
    parser.add_argument("url", nargs="?", default="",
                        help="Video URL (omit to read from stdin/prompt)")
    parser.add_argument("--no-frames", action="store_true",
                        help="Skip frame analysis (visual notes from slide keyframes)")
    args = parser.parse_args()

    print("Video → Notion Summarizer (YouTube + Vimeo)")
    print("=" * 40)

    url = args.url.strip() or input("Paste YouTube or Vimeo URL: ").strip()
    if not url:
        sys.exit("No URL provided.")

    # Normalize URL
    if not url.startswith("http"):
        url = "https://" + url

    frames_enabled = not args.no_frames and not os.getenv("YT2NOTION_NO_FRAMES")

    total_start = time.time()
    timings = {}

    # Detect platform first — determines transcript strategy and timestamp links
    platform = detect_platform(url)
    print(f"  Detected platform: {platform}")

    t0 = time.time()
    print("\n[1/6] Extracting video ID...")
    video_id = extract_video_id(url, platform)
    print(f"  Video ID: {video_id}")
    timings["1. Extract ID"] = time.time() - t0

    t0 = time.time()
    print("\n[2/6] Fetching metadata...")
    meta = fetch_metadata(url)
    print(f"  Title: {meta['title']}")
    timings["2. Metadata"] = time.time() - t0

    t0 = time.time()
    print("\n[3/6] Getting transcript...")
    if platform == "youtube":
        segments = get_youtube_transcript(video_id, url)
    else:
        segments = get_vimeo_transcript(url)
    if segments is None:
        segments = transcribe_with_whisper_local(url)
    timings["3. Transcript"] = time.time() - t0

    # Frame analysis — keyframe images must outlive Notion upload, so the
    # temp dir wraps steps 4–6. Any failure degrades to the audio-only page.
    with tempfile.TemporaryDirectory() as frames_workdir:
        frame_ctx: dict = {}
        t0 = time.time()
        if frames_enabled:
            print("\n[4/6] Analyzing video frames...")
            # OCR language order: trust the transcript over yt-dlp's often-empty
            # metadata language — a Chinese video OCR'd English-first is garbled.
            lang_hint = meta.get("language", "")
            if not lang_hint.lower().startswith("zh"):
                if _is_cjk_heavy(" ".join(s["text"] for s in segments[:80])):
                    lang_hint = "zh"
            try:
                frame_ctx = analyze_video_frames(url, frames_workdir, lang_hint)
            except Exception as e:
                print(f"  ⚠ Frame analysis failed ({e}) — continuing without visual notes")
                frame_ctx = {}
            timings["4. Frame analysis"] = time.time() - t0
        else:
            print("\n[4/6] Frame analysis skipped (--no-frames)")
        slides = frame_ctx.get("slides", [])

        t0 = time.time()
        print("\n[5/6] Generating summary...")
        timestamped_text = segments_to_text(segments)
        summary = summarize(timestamped_text, slides_to_text(slides) if slides else "")
        print("\n  Key topics:")
        for line in summary.splitlines():
            print(f"    {line}")
        if frame_ctx:
            # Layer 2: illustrate each summary topic with its on-screen frame
            attach_topic_frames(frame_ctx, summary)
            slides = frame_ctx.get("slides", [])
        timings["5. Summary (Claude)"] = time.time() - t0

        t0 = time.time()
        # Detect speakers and format as conversation via Claude
        conversation_text = format_conversation(segments)
        timings["5b. Conversation (Claude)"] = time.time() - t0

        t0 = time.time()
        print("\n[6/6] Creating Notion page...")
        page_url = create_notion_page(
            title=meta["title"],
            video_url=url,
            thumbnail_url=meta["thumbnail"],
            summary=summary,
            conversation_text=conversation_text,
            video_id=video_id,
            platform=platform,
            slides=slides,
        )
        timings["6. Notion upload"] = time.time() - t0

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
