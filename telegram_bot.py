#!/usr/bin/env python3
"""
telegram_bot.py — Telegram bot for yt2notion + audio notes

Supports:
  - YouTube URL → runs yt2notion.py → Notion page
  - .m4a audio file (Apple Voice Memo) → runs audio2notion.py → Notion page

Only responds to messages from your own Telegram user ID (ALLOWED_USER_ID).
"""

import asyncio
import os
import re
import sys
import subprocess
import tempfile
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))
SCRIPT_DIR = Path(__file__).parent
PYTHON_BIN = sys.executable  # use same Python that's running the bot

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "telegram_bot.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── YouTube URL detection ─────────────────────────────────────────────────
YT_PATTERN = re.compile(
    r"(https?://(?:www\.)?(?:youtube\.com/watch\?[^\s]*v=|youtu\.be/)[A-Za-z0-9_\-]{11}[^\s]*)"
)


def extract_youtube_url(text: str) -> "str | None":
    match = YT_PATTERN.search(text)
    return match.group(1) if match else None


# ── Podcast URL detection ──────────────────────────────────────────────────
PODCAST_PATTERN = re.compile(
    r"(https?://(?:"
    r"podcasts\.apple\.com/[^\s]+[?&]i=\d+"
    r"|(?:www\.)?xiaoyuzhoufm\.com/episode/[0-9a-f]+"
    r"|[^\s]+\.(?:mp3|m4a|wav)(?:\?[^\s]*)?"
    r")[^\s]*)",
    re.I,
)
SPOTIFY_PATTERN = re.compile(r"https?://open\.spotify\.com/episode/", re.I)
PODCAST_SCRIPT = SCRIPT_DIR.parent / "Podcast_reader" / "podcast2notion.py"


def extract_podcast_url(text: str) -> "str | None":
    match = PODCAST_PATTERN.search(text)
    return match.group(1) if match else None


# ── Auth guard ────────────────────────────────────────────────────────────
def is_authorized(update: Update) -> bool:
    return update.effective_user.id == ALLOWED_USER_ID


# ── Transcript sender ─────────────────────────────────────────────────────
TELEGRAM_MAX = 4000  # Telegram message limit is 4096; use 4000 for safety


def extract_transcript(output: str) -> "str | None":
    m = re.search(r"===TRANSCRIPT_START===\n(.*?)\n===TRANSCRIPT_END===", output, re.DOTALL)
    return m.group(1).strip() if m else None


async def send_transcript(message, transcript: str):
    """Send transcript text split into Telegram-sized chunks."""
    lines = transcript.splitlines(keepends=True)
    chunk = ""
    for line in lines:
        if len(chunk) + len(line) > TELEGRAM_MAX:
            if chunk.strip():
                await message.reply_text(chunk.strip())
            chunk = ""
        chunk += line
    if chunk.strip():
        await message.reply_text(chunk.strip())


# ── Handlers ──────────────────────────────────────────────────────────────
async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    await update.message.reply_text(
        "👋 Hi Luke! Send me a YouTube URL, a podcast episode link "
        "(Apple Podcasts / 小宇宙 / direct mp3), or an audio file and "
        "I'll transcribe it and save it to Notion."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        log.warning(f"Unauthorized access attempt from user {update.effective_user.id}")
        return

    text = update.message.text or ""
    yt_url = extract_youtube_url(text)
    podcast_url = None if yt_url else extract_podcast_url(text)

    if not yt_url and not podcast_url:
        if SPOTIFY_PATTERN.search(text):
            await update.message.reply_text(
                "Spotify episodes aren't supported (DRM-protected audio) — "
                "send the Apple Podcasts or 小宇宙 link instead."
            )
            return
        await update.message.reply_text(
            "Send me a YouTube URL or a podcast episode link "
            "(Apple Podcasts / 小宇宙 / direct mp3) to save it to Notion."
        )
        return

    url = yt_url or podcast_url
    if yt_url:
        cmd = [PYTHON_BIN, str(SCRIPT_DIR / "yt2notion.py")]
        run_kwargs = dict(input=url)
    else:
        cmd = [PYTHON_BIN, str(PODCAST_SCRIPT), url]
        run_kwargs = {}

    await update.message.reply_text(f"⏳ Processing...\n{url}")
    log.info(f"Processing: {url}")

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours — Whisper on long videos can take a while
            cwd=str(SCRIPT_DIR),
            **run_kwargs,
        )

        output = result.stdout + result.stderr

        # Extract Notion page URL from output
        notion_match = re.search(r"(https://(?:www\.notion\.so|app\.notion\.com)/\S+)", output)
        if notion_match:
            notion_url = notion_match.group(1)
            title_match = re.search(r"Title: (.+)", output)
            title = title_match.group(1) if title_match else "Video"
            await update.message.reply_text(
                f"✅ Saved to Notion!\n\n*{title}*\n\n{notion_url}",
                parse_mode="Markdown"
            )
            log.info(f"Done: {notion_url}")
            transcript = extract_transcript(output)
            if transcript:
                await update.message.reply_text("📝 Full transcript:")
                await send_transcript(update.message, transcript)
        else:
            last_lines = "\n".join(output.strip().splitlines()[-5:])
            await update.message.reply_text(f"❌ Something went wrong:\n\n{last_lines}")
            log.error(f"Failed output:\n{output}")

    except subprocess.TimeoutExpired:
        log.error(f"Timeout after 2h processing {url}")
        await update.message.reply_text("⏰ Timed out after 2 hours. The episode may be too long.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")
        log.exception("Unexpected error")


AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".mp4"}


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle audio files and voice notes sent to the bot."""
    if not is_authorized(update):
        log.warning(f"Unauthorized access attempt from user {update.effective_user.id}")
        return

    msg = update.message

    # Determine the Telegram file object and a display name
    tg_file = None
    display_name = "recording"

    if msg.audio:
        tg_file = msg.audio
        display_name = msg.audio.file_name or "audio"
    elif msg.voice:
        tg_file = msg.voice
        display_name = "voice_note"
    elif msg.document:
        ext = Path(msg.document.file_name or "").suffix.lower()
        if ext in AUDIO_EXTENSIONS:
            tg_file = msg.document
            display_name = msg.document.file_name
        else:
            await msg.reply_text("Send me an audio file (.m4a, .mp3, .wav, etc.) or a YouTube URL.")
            return
    else:
        return

    title = Path(display_name).stem  # filename without extension

    await msg.reply_text(f"⏳ Transcribing *{display_name}*...", parse_mode="Markdown")
    log.info(f"Processing audio: {display_name}")

    try:
        # Download file from Telegram into a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            suffix = Path(display_name).suffix or ".m4a"
            local_path = Path(tmpdir) / f"audio{suffix}"

            tg_file_obj = await context.bot.get_file(tg_file.file_id)
            await tg_file_obj.download_to_drive(str(local_path))
            log.info(f"Downloaded to {local_path}")

            result = await asyncio.to_thread(
                subprocess.run,
                [PYTHON_BIN, str(SCRIPT_DIR / "audio2notion.py"),
                 str(local_path)],
                capture_output=True,
                text=True,
                timeout=7200,
                cwd=str(SCRIPT_DIR),
            )

        output = result.stdout + result.stderr

        notion_match = re.search(r"(https://(?:www\.notion\.so|app\.notion\.com)/\S+)", output)
        if notion_match:
            notion_url = notion_match.group(1)
            await msg.reply_text(
                f"✅ Transcribed to Notion!\n\n*{title}*\n\n{notion_url}",
                parse_mode="Markdown"
            )
            log.info(f"Done: {notion_url}")
            transcript = extract_transcript(output)
            if transcript:
                await msg.reply_text("📝 Full transcript:")
                await send_transcript(msg, transcript)
        else:
            last_lines = "\n".join(output.strip().splitlines()[-5:])
            await msg.reply_text(f"❌ Something went wrong:\n\n{last_lines}")
            log.error(f"Failed output:\n{output}")

    except subprocess.TimeoutExpired:
        log.error(f"Timeout after 2h processing audio: {display_name}")
        await msg.reply_text("⏰ Timed out after 2 hours.")
    except Exception as e:
        await msg.reply_text(f"❌ Error: {e}")
        log.exception("Unexpected error in handle_audio")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    log.info("Starting yt2notion Telegram bot...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # Audio: voice notes, audio files, and documents with audio extensions
    app.add_handler(MessageHandler(
        filters.AUDIO | filters.VOICE | filters.Document.ALL,
        handle_audio
    ))
    log.info("Bot is running. Send a YouTube URL or drop an audio file.")
    app.run_polling()


if __name__ == "__main__":
    main()
