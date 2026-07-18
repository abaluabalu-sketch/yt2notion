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
     - Summary whose timestamps jump to the matching transcript section,
       each with a ▶ link out to that moment on YouTube
     - The slide that was on screen embedded under its summary topic
     - Full transcript as readable conversation, split into topic sections

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
FRAME_INK_REMOVED_MAX = 0.02   # ≤ this fraction of content lost → same slide, still growing
FRAME_BUILD_CHANGE_MAX = 0.35  # > this much of the picture changed → a new slide, not a build step
FRAME_INK_IGNORE_BOTTOM = 0.15 # ignore bottom rows (burned-in subtitles) when measuring content
FRAME_DEDUP_RATIO     = 0.10   # ≤ this ratio between captured frames = re-shown slide
FRAME_MIN_STABLE_SECONDS = 5.0 # a whole slide (all its build stages) must persist this long
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


def _ink_mask(thumb: bytes) -> int:
    """Bitmask of 'content' pixels — those standing out from the background.

    Background is the frame's median gray level, so this works for both
    light-on-dark and dark-on-light slides. The bottom
    FRAME_INK_IGNORE_BOTTOM of rows is excluded because burned-in subtitles
    live there and would otherwise read as content constantly appearing and
    disappearing.
    """
    rows = _THUMB_H - int(_THUMB_H * FRAME_INK_IGNORE_BOTTOM)
    usable = thumb[:rows * _THUMB_W]
    ordered = sorted(usable)
    background = ordered[len(ordered) // 2]
    mask = 0
    for i, pixel in enumerate(usable):
        if abs(pixel - background) > FRAME_PIXEL_DELTA:
            mask |= 1 << i
    return mask


def _ink_size(mask: int) -> int:
    return bin(mask).count("1")


def _is_build_continuation(earlier: bytes, later: bytes) -> bool:
    """True when `later` is `earlier` with more elements added to it.

    Three conditions, all needed: the rest of the picture is untouched (an
    added element changes a limited region — a slide flip repaints everything,
    even when the new slide happens to put content in the same places),
    nothing that was there vanished, and something new did appear.
    """
    if _changed_ratio(earlier, later) > FRAME_BUILD_CHANGE_MAX:
        return False  # a different picture, not the same slide growing
    a, b = _ink_mask(earlier), _ink_mask(later)
    total = _THUMB_W * (_THUMB_H - int(_THUMB_H * FRAME_INK_IGNORE_BOTTOM))
    removed = _ink_size(a & ~b) / total
    return removed <= FRAME_INK_REMOVED_MAX and (b & ~a) != 0


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


def _pick_capture_index(thumbs: list[bytes], lo: int, hi: int) -> int:
    """Index of the most complete settled frame in [lo, hi].

    Only quiet frames are eligible, so a mid-fade frame is never chosen. Among
    those, the one with the most content wins (latest on ties) — content peaks
    once the author has added the final element.
    """
    best_i, best_ink = lo, -1
    for i in range(lo, hi + 1):
        quiet = i == lo or _changed_ratio(thumbs[i], thumbs[i - 1]) <= FRAME_SOFT_CHANGE_RATIO
        if not quiet:
            continue
        ink = _ink_size(_ink_mask(thumbs[i]))
        if ink >= best_ink:  # >= so a later frame wins an equal-content tie
            best_i, best_ink = i, ink
    return best_i


def _find_stable_states(thumbs: list[bytes]) -> list[dict]:
    """Group 1fps thumbnails into one entry per distinct slide.

    Stages are split wherever the picture drifts from the stage's first frame,
    then consecutive stages are grouped while content only grows — the
    signature of an author progressively building one slide. Each group is
    captured at the most complete settled frame of its FINAL stage, so the
    screenshot shows the slide after the last element landed.

    The minimum-duration filter is applied to the whole group, never to a
    single stage: a completed state that is only briefly on screen before the
    author moves on must not be discarded.
    Returns [{"start","end","capture","mid","thumb"}].
    """
    if not thumbs:
        return []
    dt = 1.0 / FRAME_SAMPLE_FPS

    # ── Stages: split on drift from the stage's own first frame ──────────
    stages: list[tuple[int, int]] = []  # (start_i, end_i) inclusive
    anchor, stage_start = thumbs[0], 0
    for i in range(1, len(thumbs)):
        if _changed_ratio(thumbs[i], anchor) > FRAME_SPLIT_DRIFT_RATIO:
            stages.append((stage_start, i - 1))
            anchor, stage_start = thumbs[i], i
    stages.append((stage_start, len(thumbs) - 1))

    # ── Groups: consecutive stages where content only grows = one slide ──
    groups: list[list[tuple[int, int]]] = []
    for stage in stages:
        if groups and _is_build_continuation(thumbs[groups[-1][-1][1]], thumbs[stage[1]]):
            groups[-1].append(stage)
        else:
            groups.append([stage])

    states: list[dict] = []
    for group in groups:
        start_i, end_i = group[0][0], group[-1][1]
        final_lo, final_hi = group[-1]
        capture_i = _pick_capture_index(thumbs, final_lo, final_hi)
        start_ts, end_ts = start_i * dt, (end_i + 1) * dt
        capture_ts = max(start_ts, min(capture_i * dt, end_ts - 0.5))
        states.append({
            "start": start_ts, "end": end_ts, "capture": capture_ts,
            "mid": (start_ts + end_ts) / 2, "thumb": thumbs[capture_i],
        })

    all_states = states
    states = [s for s in states if s["end"] - s["start"] >= FRAME_MIN_STABLE_SECONDS]
    if not states:  # short/dynamic video — keep the longest slides anyway
        states = sorted(all_states, key=lambda s: s["start"] - s["end"])[:3]
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

    One frame per slide, taken at `state["capture"]` — the settled frame with
    the most content in the slide's final build stage, i.e. after the author
    added the last element. The returned timestamp is the slide's START so the
    Notion timestamp link jumps to where the slide first appeared.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    states = _find_stable_states(_sample_thumbs(video_path))
    kept: list[tuple[Path, float, dict, str]] = []
    for i, state in enumerate(states):
        out = frames_dir / f"{i + 1:04d}.jpg"
        if not _extract_frame_at(video_path, state["capture"], out):
            continue
        kept.append((out, state["start"], state, _ocr_image(out, ocr_langs)))
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

    Safety net for build-ups the pixel-level grouping in `_find_stable_states`
    did not catch. A slide whose OCR text is contained in the NEXT slide's is
    an earlier build stage of it; near-identical neighbours are the same slide
    re-detected. Either way the merge keeps the MORE COMPLETE image and the
    EARLIER timestamp — never a half-built frame.
    """
    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", t.strip().lower())

    def absorb(previous: dict, current: dict, keep: dict) -> dict:
        """Merge two detections of one slide, keeping `keep`'s image."""
        merged = dict(keep)
        merged["start"] = previous["start"]
        if "state" in merged and "state" in previous:
            merged["state"] = {**merged["state"], "start": previous["state"]["start"]}
        return merged

    deduped: list[dict] = []
    for slide in slides:
        if deduped:
            previous = deduped[-1]
            prev, cur = norm(previous["ocr_text"]), norm(slide["ocr_text"])
            if prev and cur and _text_contains(prev, cur):
                deduped[-1] = absorb(previous, slide, slide)  # later = completed
                continue
            if prev and cur and difflib.SequenceMatcher(None, prev, cur).ratio() > 0.85:
                # Same slide seen twice — keep whichever frame reads richer
                richer = slide if len(cur) > len(prev) else previous
                deduped[-1] = absorb(previous, slide, richer)
                continue
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


def _split_raw_text(raw_text: str, chunk_size: int = 10000) -> list[str]:
    """Split raw transcript text into ~chunk_size pieces at sentence bounds."""
    chunks = []
    for i in range(0, len(raw_text), chunk_size):
        end = min(i + chunk_size, len(raw_text))
        if end < len(raw_text):
            last_period = raw_text.rfind(". ", i + chunk_size - 500, end)
            if last_period > i:
                end = last_period + 2
        chunks.append(raw_text[i:end])
    return chunks


def format_conversation(segments: list[dict], topics: list[dict] | None = None) -> list[dict]:
    """Detect speakers, clean transcript, and format as conversation sections.

    Returns [{"ts_str": str|None, "title": str, "text": str}]. When `topics`
    (from `parse_summary_topics`) is given, the transcript is split at topic
    boundaries so each section can be an anchor target for its summary bullet;
    otherwise a single untitled section is returned.

    Phase 1: identify speakers from a sample
    Phase 2: split into topic sections, sub-chunked to ~10K chars
    Phase 3: format every chunk in parallel via concurrent Claude CLI calls
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

    # ── Phase 2: Split into topic sections, then ~10K char chunks ────────
    sections: list[dict] = []
    if topics:
        for i, topic in enumerate(topics):
            # First topic absorbs any lead-in before its timestamp
            start = float("-inf") if i == 0 else topic["seconds"]
            end = topics[i + 1]["seconds"] if i + 1 < len(topics) else float("inf")
            text = " ".join(s["text"] for s in segments if start <= s["start"] < end)
            if text.strip():
                sections.append({"ts_str": topic["ts_str"], "title": topic["title"],
                                 "raw": text})
    if not sections:  # no topics parsed, or they matched nothing
        sections = [{"ts_str": None, "title": "", "raw": raw_text}]

    # Flatten to (section_idx, chunk_idx, text) so every chunk runs in parallel
    tasks: list[tuple[int, int, str]] = []
    for si, sec in enumerate(sections):
        for ci, chunk in enumerate(_split_raw_text(sec["raw"])):
            tasks.append((si, ci, chunk))

    print(f"  Split into {len(sections)} sections / {len(tasks)} chunks, "
          "formatting in parallel...")

    # ── Phase 3: Format chunks in parallel ───────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def format_one(task):
        si, ci, chunk = task
        return si, ci, _format_chunk(chunk, speaker_context, ci)

    parts: dict[int, dict[int, str]] = {si: {} for si in range(len(sections))}
    done = 0
    with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as pool:
        futures = [pool.submit(format_one, t) for t in tasks]
        for future in as_completed(futures):
            si, ci, formatted = future.result()
            parts[si][ci] = formatted
            done += 1
            print(f"  Chunk {done}/{len(tasks)} done")

    for si, sec in enumerate(sections):
        ordered = [parts[si][ci] for ci in sorted(parts[si]) if parts[si][ci]]
        sec["text"] = "\n\n".join(ordered)
    sections = [s for s in sections if s.get("text", "").strip()]

    # Count speakers in output
    speakers = set()
    for line in "\n".join(s["text"] for s in sections).split("\n"):
        if ":" in line and not line.startswith(" "):
            speaker = line.split(":")[0].strip()
            if speaker and len(speaker) < 40 and not any(c.isdigit() for c in speaker):
                speakers.add(speaker)
    if speakers:
        print(f"  Formatted with {len(speakers)} speakers: {', '.join(sorted(speakers))}")

    return sections


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


def parse_summary_topics(summary: str) -> list[dict]:
    """Extract [{seconds, ts_str, title}] from the summary's topic lines.

    Shared by the transcript sectioning and the Notion anchor linking so both
    see exactly the same topic list.
    """
    topics = []
    for line in summary.splitlines():
        line = line.strip().lstrip("•-* ")
        m = re.match(r"^\[(\d+:\d{2}(?::\d{2})?)\]\s*(.*)", line)
        if not m:
            continue
        ts_str, rest = m.group(1), m.group(2)
        # Topic title = text before the first colon (full- or half-width)
        title = re.split(r"[：:]", rest, 1)[0].strip() or rest.strip()
        topics.append({
            "seconds": timestamp_to_seconds(ts_str),
            "ts_str": ts_str,
            "title": title[:80],
        })
    return topics


def summary_bullet_block(text: str, video_id: str, platform: str = "youtube") -> dict:
    """Summary bullet: [MM:SS] jumps to the transcript, ▶ opens the video.

    The timestamp's in-page link can only be attached once the transcript
    blocks exist (Notion block ids are assigned at creation), so it is left
    unlinked here and patched afterwards by `link_summary_to_transcript`.
    """
    match = re.match(r"^\[(\d+:\d{2}(?::\d{2})?)\]\s*(.*)", text)
    if not match:
        return bullet_block(text)
    ts_str, rest = match.group(1), match.group(2)

    rich_text = [{
        "type": "text",
        "text": {"content": f"[{ts_str}]"},
        "annotations": {"bold": True, "color": "blue"},
    }]
    if platform == "youtube" and video_id:
        seconds = timestamp_to_seconds(ts_str)
        rich_text.append({
            "type": "text",
            "text": {"content": " ▶",
                     "link": {"url": f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"}},
            "annotations": {"color": "gray"},
        })
    rich_text.append({"type": "text", "text": {"content": f" {rest}"}})
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": rich_text},
    }


def _rich_text_for_update(items: list[dict]) -> list[dict]:
    """Rebuild API-returned rich_text into a payload accepted on update."""
    out = []
    for r in items:
        if r.get("type") != "text":
            continue
        item = {"type": "text", "text": {"content": r["text"]["content"]}}
        if r["text"].get("link"):
            item["text"]["link"] = r["text"]["link"]
        ann = {k: v for k, v in (r.get("annotations") or {}).items()
               if k in ("bold", "italic", "strikethrough", "underline", "code", "color")}
        if ann:
            out.append({**item, "annotations": ann})
        else:
            out.append(item)
    return out


def _block_plain_text(block: dict) -> str:
    body = block.get(block.get("type"), {})
    return "".join(r.get("plain_text", "") for r in body.get("rich_text", []))


def link_summary_to_transcript(notion, page_id: str, page_url: str) -> None:
    """Point each summary timestamp at its transcript section (never raises).

    Walks the page's top-level blocks, maps transcript section headings by
    timestamp, then patches the matching summary bullets with an in-page
    anchor link (`<page_url>#<block_id>`).
    """
    try:
        blocks, cursor = [], None
        while True:
            kwargs = {"block_id": page_id, "page_size": 100}
            if cursor:
                kwargs["start_cursor"] = cursor
            resp = notion.blocks.children.list(**kwargs)
            blocks.extend(resp.get("results", []))
            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")

        # Transcript section headings, keyed by timestamp
        anchors, in_transcript = {}, False
        for b in blocks:
            if b.get("type") == "heading_2":
                in_transcript = _block_plain_text(b).strip() == "Full Transcript"
            elif in_transcript and b.get("type") == "heading_3":
                m = re.match(r"^\[(\d+:\d{2}(?::\d{2})?)\]", _block_plain_text(b).strip())
                if m:
                    anchors[m.group(1)] = b["id"]
        if not anchors:
            return

        # Patch summary bullets only (between the Summary heading and its divider)
        in_summary, patched = False, 0
        for b in blocks:
            btype = b.get("type")
            if btype == "heading_2":
                in_summary = _block_plain_text(b).strip() == "Summary"
                continue
            if btype == "divider":
                in_summary = False
                continue
            if not in_summary or btype != "bulleted_list_item":
                continue
            rich = b["bulleted_list_item"].get("rich_text", [])
            if not rich:
                continue
            m = re.match(r"^\[(\d+:\d{2}(?::\d{2})?)\]", rich[0].get("plain_text", ""))
            block_id = anchors.get(m.group(1)) if m else None
            if not block_id:
                continue
            new_rich = _rich_text_for_update(rich)
            if not new_rich:
                continue
            new_rich[0]["text"]["link"] = {
                "url": f"{page_url}#{block_id.replace('-', '')}"
            }
            notion.blocks.update(block_id=b["id"],
                                 bulleted_list_item={"rich_text": new_rich})
            patched += 1
        if patched:
            print(f"  Linked {patched} summary timestamps to transcript sections")
    except Exception as e:
        print(f"  ⚠ Could not link summary to transcript ({e}) — ▶ video links still work")


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
    transcript_sections: list[dict],
    video_id: str = "",
    platform: str = "youtube",
    slides: list[dict] | None = None,
) -> str:
    """Create the Notion page and return its URL."""
    notion = get_notion()
    parent_id = os.getenv("NOTION_DATABASE_ID")
    if not parent_id:
        sys.exit("Error: NOTION_DATABASE_ID not set in .env")

    # Only topic-tagged slides are shown, as an image under their summary
    # bullet. The rest still shaped the summary via the SLIDES prompt section.
    slides = slides or []
    topic_slides = {s["topic_ts"]: s for s in slides if s.get("topic_ts") is not None}
    embeds_left = FRAME_MAX_EMBEDS
    if topic_slides:
        print(f"  Uploading up to {min(len(topic_slides), FRAME_MAX_EMBEDS)} "
              "topic images to Notion...")

    # Build summary bullet blocks (each followed by its topic frame, if any).
    # The [MM:SS] is linked to its transcript section after the page exists;
    # the ▶ next to it opens the video at that moment.
    summary_bullets = []
    for line in summary.splitlines():
        line = line.strip().lstrip("•-* ")
        if not line:
            continue
        summary_bullets.append(summary_bullet_block(line, video_id, platform))
        m = re.match(r"^\[(\d+:\d{2}(?::\d{2})?)\]", line)
        slide = topic_slides.get(timestamp_to_seconds(m.group(1))) if m else None
        if slide is not None and embeds_left > 0:
            upload_id = _upload_file_to_notion(slide["image_path"])
            if upload_id:
                summary_bullets.append(image_block_uploaded(upload_id))
                embeds_left -= 1

    # Build transcript paragraph blocks (chunked)
    # Transcript renders as one section per summary topic; each section's
    # heading is the anchor a summary timestamp jumps to
    transcript_blocks: list[dict] = []
    for sec in transcript_sections:
        if sec.get("ts_str"):
            # NB: must not shadow the `title` parameter — it is the page title
            section_heading = f"[{sec['ts_str']}] {sec.get('title', '')}".strip()
            transcript_blocks.append(heading_block(section_heading, level=3))
        for chunk in chunk_text(sec["text"]):
            transcript_blocks.append(paragraph_block(chunk))

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

    # Now that transcript blocks have ids, point summary timestamps at them
    link_summary_to_transcript(notion, page_id, page_url)

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
        # Detect speakers and format as conversation, split into topic sections
        # so each summary timestamp has a transcript anchor to jump to
        transcript_sections = format_conversation(segments, parse_summary_topics(summary))
        timings["5b. Conversation (Claude)"] = time.time() - t0

        t0 = time.time()
        print("\n[6/6] Creating Notion page...")
        page_url = create_notion_page(
            title=meta["title"],
            video_url=url,
            thumbnail_url=meta["thumbnail"],
            summary=summary,
            transcript_sections=transcript_sections,
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
