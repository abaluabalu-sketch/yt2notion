#!/usr/bin/env python3
"""
telegram_bot.py — Telegram bot for yt2notion

Send a YouTube URL from your iPhone to this bot → it runs yt2notion.py
on your Mac and replies with the Notion page URL.

Only responds to messages from your own Telegram user ID (ALLOWED_USER_ID).
"""

import os
import re
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))
SCRIPT_DIR = Path(__file__).parent
PYTHON_BIN = "/opt/anaconda3/bin/python3"

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


def extract_youtube_url(text: str) -> str | None:
    match = YT_PATTERN.search(text)
    return match.group(1) if match else None


# ── Auth guard ────────────────────────────────────────────────────────────
def is_authorized(update: Update) -> bool:
    return update.effective_user.id == ALLOWED_USER_ID


# ── Handlers ──────────────────────────────────────────────────────────────
async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    await update.message.reply_text(
        "👋 Hi Luke! Send me a YouTube URL and I'll save it to Notion.\n\n"
        "Just paste any YouTube link and I'll handle the rest."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        log.warning(f"Unauthorized access attempt from user {update.effective_user.id}")
        return

    text = update.message.text or ""
    yt_url = extract_youtube_url(text)

    if not yt_url:
        await update.message.reply_text("Send me a YouTube URL to save it to Notion.")
        return

    await update.message.reply_text(f"⏳ Processing...\n{yt_url}")
    log.info(f"Processing: {yt_url}")

    try:
        result = subprocess.run(
            [PYTHON_BIN, str(SCRIPT_DIR / "yt2notion.py")],
            input=yt_url,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(SCRIPT_DIR),
        )

        output = result.stdout + result.stderr

        # Extract Notion page URL from output
        notion_match = re.search(r"(https://www\.notion\.so/\S+)", output)
        if notion_match:
            notion_url = notion_match.group(1)
            # Extract title from output
            title_match = re.search(r"Title: (.+)", output)
            title = title_match.group(1) if title_match else "Video"
            await update.message.reply_text(
                f"✅ Saved to Notion!\n\n*{title}*\n\n{notion_url}",
                parse_mode="Markdown"
            )
            log.info(f"Done: {notion_url}")
        else:
            # Something went wrong — send the last few lines of output
            last_lines = "\n".join(output.strip().splitlines()[-5:])
            await update.message.reply_text(f"❌ Something went wrong:\n\n{last_lines}")
            log.error(f"Failed output:\n{output}")

    except subprocess.TimeoutExpired:
        await update.message.reply_text("⏰ Timed out after 10 minutes. Try a shorter video.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")
        log.exception("Unexpected error")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    log.info("Starting yt2notion Telegram bot...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("Bot is running. Send a YouTube URL on Telegram.")
    app.run_polling()


if __name__ == "__main__":
    main()
