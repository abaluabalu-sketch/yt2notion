#!/usr/bin/env python3 -u
"""
audio2notion.py — Local audio note → Notion page

Usage: python audio2notion.py /path/to/audio.m4a

Steps:
  1. Convert audio to 16kHz WAV (ffmpeg)
  2. Transcribe with whisper.cpp large-v3-turbo (Metal GPU)
  3. Clean up transcript via Claude (filler words, false starts, grammar)
  4. Create Notion page with clean transcription
"""

import os
import re
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Ensure local Node.js is on PATH ──────────────────────────────────────────
_node_dir = Path.home() / ".local" / "node" / "bin"
if _node_dir.exists() and str(_node_dir) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_node_dir}:{os.environ.get('PATH', '')}"

# ── ffmpeg path ───────────────────────────────────────────────────────────────
def get_ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"

FFMPEG_PATH = get_ffmpeg_path()

# ── whisper.cpp paths ─────────────────────────────────────────────────────────
WHISPER_CPP_BIN   = Path.home() / ".local" / "whisper-cpp" / "whisper-cli"
WHISPER_CPP_MODEL = Path.home() / ".local" / "whisper-cpp" / "models" / "ggml-large-v3-turbo.bin"

# ── Claude CLI ────────────────────────────────────────────────────────────────
CLAUDE_BIN = Path.home() / ".local" / "bin" / "claude"


def call_claude(prompt: str, max_tokens: int = 8192) -> str:
    """Call Claude CLI, fall back to OpenAI gpt-4o-mini if unavailable."""
    result = subprocess.run(
        [str(CLAUDE_BIN), "--print", "--output-format", "text"],
        input=prompt,
        capture_output=True, text=True
    )
    if result.returncode == 0:
        output = result.stdout.strip()
        if output and "auto mode temporarily unavailable" not in output.lower():
            return output

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


# ── Transcription ─────────────────────────────────────────────────────────────

def _strip_trailing_hallucinations(segments: list[dict]) -> list[dict]:
    """Remove repeated segments from the tail of whisper output (common artifact)."""
    def norm(t: str) -> str:
        return re.sub(r'\s+', ' ', t.strip().lower())

    if len(segments) >= 3:
        tail_text = norm(segments[-1]["text"])
        repeat_count = sum(1 for seg in reversed(segments) if norm(seg["text"]) == tail_text)
        # stop counting once a non-matching seg is found
        actual_repeat = 0
        for seg in reversed(segments):
            if norm(seg["text"]) == tail_text:
                actual_repeat += 1
            else:
                break
        if actual_repeat >= 3:
            print(f"  Stripped {actual_repeat} trailing hallucinated segments")
            segments = segments[:-actual_repeat]

    cleaned = []
    run_count = 1
    for i, seg in enumerate(segments):
        if i > 0 and norm(seg["text"]) == norm(segments[i - 1]["text"]):
            run_count += 1
        else:
            run_count = 1
        if run_count <= 2:
            cleaned.append(seg)

    if len(cleaned) < len(segments):
        print(f"  Removed {len(segments) - len(cleaned)} duplicate interior segments")

    return cleaned


def _parse_whisper_json(json_path: Path) -> list[dict]:
    """Parse whisper.cpp JSON output into {text, start, end} segments."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = []
    for item in data.get("transcription", []):
        text = item.get("text", "").strip()
        ts_from = item.get("timestamps", {}).get("from", "00:00:00.000")
        ts_to   = item.get("timestamps", {}).get("to", ts_from)
        parts_from = ts_from.replace(",", ".").split(":")
        parts_to   = ts_to.replace(",", ".").split(":")
        start = float(parts_from[0]) * 3600 + float(parts_from[1]) * 60 + float(parts_from[2])
        end   = float(parts_to[0])   * 3600 + float(parts_to[1])   * 60 + float(parts_to[2])
        if text:
            segments.append({"text": text, "start": start, "end": end})

    if len(segments) > 5:
        segments = _strip_trailing_hallucinations(segments)

    return segments


def transcribe_audio(audio_path: Path) -> list[dict]:
    """Convert audio to 16kHz WAV and transcribe with whisper.cpp (Metal GPU)."""
    if not WHISPER_CPP_BIN.exists():
        sys.exit(f"Error: whisper.cpp not found at {WHISPER_CPP_BIN}")
    if not WHISPER_CPP_MODEL.exists():
        sys.exit(f"Error: Whisper model not found at {WHISPER_CPP_MODEL}")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        print("  Converting audio to 16kHz WAV...")
        subprocess.run(
            [FFMPEG_PATH, "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)],
            capture_output=True, check=True
        )

        print("  Transcribing with large-v3-turbo (Metal GPU)...")
        r = subprocess.run(
            [str(WHISPER_CPP_BIN),
             "-m", str(WHISPER_CPP_MODEL),
             "-f", str(wav_path),
             "-l", "auto",
             "--output-json",
             "-of", str(Path(tmpdir) / "out")],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            sys.exit(f"Error: whisper.cpp failed:\n{r.stderr}")

        json_path = Path(tmpdir) / "out.json"
        if not json_path.exists():
            sys.exit(f"Error: whisper.cpp produced no JSON.\nstderr: {r.stderr}")

        segments = _parse_whisper_json(json_path)

    print(f"  whisper.cpp returned {len(segments)} segments")
    return segments


def segments_to_raw_text(segments: list[dict]) -> str:
    return " ".join(seg["text"] for seg in segments)


# ── Title generation ─────────────────────────────────────────────────────────

def generate_title(transcript: str) -> str:
    """Generate a concise Notion page title from the cleaned transcript."""
    print("  Generating title with Claude...")
    sample = transcript[:3000]  # first 3K chars is enough to infer the topic
    prompt = (
        "You are titling a personal voice note for a Notion page.\n"
        "Read the transcript excerpt and return a short, descriptive title.\n\n"
        "RULES:\n"
        "- Max 10 words\n"
        "- Match the language of the transcript (English → English title, 中文 → 中文標題)\n"
        "- Capture the main topic or intent of the note\n"
        "- No quotes, no punctuation at the end\n\n"
        "Return ONLY the title, nothing else.\n\n"
        "TRANSCRIPT:\n" + sample
    )
    result = call_claude(prompt, max_tokens=64)
    return result.strip().strip('"').strip("'")


# ── Transcript cleanup ────────────────────────────────────────────────────────

def _clean_chunk(raw: str) -> str:
    """Send one chunk to Claude for cleanup. Returns cleaned plain text."""
    prompt = (
        "You are editing a personal voice note transcript (single speaker).\n"
        "Clean up the raw speech-to-text output into readable prose.\n\n"
        "RULES:\n"
        "- Remove filler words: \"um\", \"uh\", \"like\", \"you know\"\n"
        "- Collapse false starts: \"I— I think—\" → \"I think\"\n"
        "- Remove repeated words: \"it's it's really\" → \"it's really\"\n"
        "- Fix obvious speech-to-text errors and grammar\n"
        "- Preserve the speaker's natural voice and meaning\n"
        "- Keep technical terms, proper nouns, and numbers accurate\n"
        "- Mark unclear audio as [inaudible]\n"
        "- Output clean flowing paragraphs, no speaker labels, no timestamps\n\n"
        "Output ONLY the cleaned text. No preamble, no commentary.\n\n"
        "RAW TRANSCRIPT:\n" + raw
    )
    result = subprocess.run(
        [str(CLAUDE_BIN), "--print", "--output-format", "text"],
        input=prompt, capture_output=True, text=True
    )
    if result.returncode != 0:
        return raw
    return result.stdout.strip()


def clean_transcript(segments: list[dict]) -> str:
    """Clean raw whisper segments into readable prose via Claude."""
    print("  Cleaning transcript (Claude)...")
    raw_text = segments_to_raw_text(segments)

    # Split into ~10K char chunks at sentence boundaries
    chunk_size = 10000
    chunks = []
    for i in range(0, len(raw_text), chunk_size):
        end = min(i + chunk_size, len(raw_text))
        if end < len(raw_text):
            last_period = raw_text.rfind(". ", i + chunk_size - 500, end)
            if last_period > i:
                end = last_period + 2
        chunks.append(raw_text[i:end])

    print(f"  {len(chunks)} chunk(s) to clean...")
    results = [_clean_chunk(chunk) for chunk in chunks]
    print(f"  Done")
    return "\n\n".join(results)


# ── Notion helpers ────────────────────────────────────────────────────────────

def chunk_text(text: str, max_len: int = 1900) -> list[str]:
    """Split text into chunks within Notion's paragraph block size limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind(". ", 0, max_len)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_len)
        if split_at == -1:
            split_at = max_len
        else:
            split_at += 1
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    return chunks


def paragraph_block(text: str) -> dict:
    return {
        "object": "block", "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def create_notion_page(title: str, transcript: str) -> str:
    """Create the Notion page and return its URL."""
    notion = get_notion()
    parent_id = os.getenv("NOTION_DATABASE_ID")
    if not parent_id:
        sys.exit("Error: NOTION_DATABASE_ID not set in .env")

    blocks: list[dict] = [paragraph_block(c) for c in chunk_text(transcript)]

    print("  Creating Notion page...")
    page = notion.pages.create(
        parent={"type": "database_id", "database_id": parent_id},
        properties={
            "title": {"title": [{"type": "text", "text": {"content": title}}]},
        },
        children=blocks[:100],
    )
    page_id  = page["id"]
    page_url = page["url"]

    remaining = blocks[100:]
    while remaining:
        notion.blocks.children.append(block_id=page_id, children=remaining[:100])
        remaining = remaining[100:]

    return page_url


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) >= 2:
        audio_path = Path(sys.argv[1].strip("'\""))
    else:
        audio_path = Path(input("Audio file path: ").strip().strip("'\""))

    if not audio_path.exists():
        sys.exit(f"Error: File not found: {audio_path}")

    print(f"\nAudio Note → Notion")
    print("=" * 40)
    print(f"  File : {audio_path.name}")

    total_start = time.time()

    print("\n[1/3] Transcribing audio...")
    t0 = time.time()
    segments = transcribe_audio(audio_path)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n[2/3] Cleaning transcript & generating title...")
    t0 = time.time()
    transcript = clean_transcript(segments)
    title = generate_title(transcript)
    print(f"  Title: {title}")
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n[3/3] Creating Notion page...")
    t0 = time.time()
    page_url = create_notion_page(title, transcript)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\nTotal: {time.time() - total_start:.1f}s")
    print(f"\nNotion page: {page_url}")


if __name__ == "__main__":
    main()
