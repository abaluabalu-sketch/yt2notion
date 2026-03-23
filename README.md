# yt2notion — YouTube to Notion Summarizer

Paste a YouTube URL → get a structured Notion page with:
- **Summary** in Traditional Chinese with clickable timestamps
- **Full transcript** formatted as a conversation with speaker names
- Cleaned up (filler words removed, false starts collapsed, grammar fixed)

Works from CLI, Claude Code chat, or Telegram on your phone.

---

## Quick Start

```bash
git clone https://github.com/YourUsername/yt2notion.git
cd yt2notion
bash setup.sh        # installs everything automatically
cp .env.example .env # fill in your Notion keys
```

Then:
```bash
echo "https://youtu.be/VIDEO_ID" | python3 yt2notion.py
```

---

## What You Need

| Requirement | Purpose | How to Get |
|---|---|---|
| Python 3.10+ | Run the script | [python.org](https://python.org) |
| Claude subscription | Summarization + formatting (via CLI) | [claude.ai](https://claude.ai) |
| Notion API key | Create pages | [notion.so/my-integrations](https://www.notion.so/my-integrations) |
| Notion Database ID | Target database | From your database URL |
| Google Chrome | YouTube cookies (members-only videos) | Already installed |

> **No OpenAI or Anthropic API key needed.** Summarization uses the `claude` CLI from your Claude Code subscription.

---

## Setup

### Option A: One-click (recommended)
```bash
bash setup.sh
```
This automatically installs:
- Python packages (`requirements.txt`)
- whisper.cpp with Metal GPU support
- Whisper large-v3-turbo model (1.6 GB)
- Node.js (for yt-dlp)
- Claude CLI

### Option B: Manual
See [Manual Installation](#manual-installation) below.

### Configure `.env`
```bash
cp .env.example .env
```
Fill in:
```
NOTION_API_KEY=secret_...
NOTION_DATABASE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### How to get your Notion keys

**Notion API Key:**
1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations)
2. Click **New integration** → name it → copy the token (starts with `secret_`)
3. Open your Notion database → `...` menu → **Add connections** → select your integration

**Database ID:**
1. Open your database in the browser
2. URL: `https://www.notion.so/workspace/DATABASE_ID?v=...`
3. Copy the 32-char hex string

> **Important:** Your Notion database needs a **URL property** (type: URL) for the YouTube link to be saved.

---

## Usage

### From terminal
```bash
echo "https://youtu.be/VIDEO_ID" | python3 yt2notion.py
```

### From Claude Code chat
Just say:
> "Save this to Notion: https://youtu.be/VIDEO_ID"

### From iPhone (Telegram)
Send the YouTube URL to your `@YT2Notion_bot` on Telegram.
(Requires `telegram_bot.py` running on your Mac — see [Telegram Bot](#telegram-bot))

---

## Architecture

```
YouTube URL
    │
    ▼
[1] yt-dlp ──────────────► metadata (title, thumbnail)
    │
    ▼
[2] Transcript (2-tier, never auto-generated)
    ├─ Tier 1: youtube_transcript_api (manual subs only)
    ├─ Tier 2: yt-dlp --write-sub (manual subs + Chrome cookies)
    └─ Tier 3: whisper.cpp large-v3-turbo (local, Metal GPU)
    │
    ▼
[3] Claude CLI ─────────► Summary (Traditional Chinese, 5-10 topics)
    │                      with clickable YouTube timestamps
    ▼
[4] Claude CLI ─────────► Speaker detection + conversation formatting
    │                      (parallel chunked processing)
    │                      + transcript cleaning rules
    ▼
[5] Notion API ─────────► Structured page with URL property
```

### Performance (65-min video)

| Step | Time |
|---|---|
| Whisper transcription (turbo, Metal GPU) | ~6 min |
| Summary (Claude) | ~30s |
| Conversation formatting (Claude, parallel) | ~2.5 min |
| Notion upload | ~3s |
| **Total** | **~9.5 min** |

---

## Features

### Transcript Strategy
- **Manual subtitles only** — auto-generated subs are never used
- **whisper.cpp large-v3-turbo** — 2.3x faster than large-v3, local Metal GPU
- **Hallucination stripping** — detects and removes repeated phrases at end of audio

### Summary
- Always in **Traditional Chinese** (繁體中文)
- Technical terms kept in English (Claude Code, API, SDK, etc.)
- 5-10 key topics with **clickable timestamps** → jumps to that moment on YouTube

### Conversation Formatting
- **Speaker detection** via Claude — identifies real names from context
- **Parallel chunking** — transcript split into ~10K char chunks, processed concurrently
- **Cleaning rules applied:**
  - Filler words removed (um, uh, like)
  - False starts collapsed ("I— I think—" → "I think")
  - Repeated words cleaned
  - Em-dash for interruptions
  - [inaudible] / [crosstalk] markers

### Notion Page
- YouTube bookmark + thumbnail
- URL property with YouTube link
- Summary with clickable timestamp links (bold blue)
- Full transcript as clean conversation

---

## Telegram Bot

Run the bot on your Mac to use from iPhone:

```bash
# First time: set your bot token and user ID in telegram_bot.py
nohup python3 telegram_bot.py &
```

Then send any YouTube URL to your bot on Telegram.

---

## File Structure

```
yt2notion/
├── yt2notion.py        # Main pipeline (~750 lines)
├── telegram_bot.py     # Telegram bot for iPhone access
├── setup.sh            # One-click installer
├── requirements.txt    # Python dependencies
├── .env.example        # Config template
├── .env                # Your keys (not committed)
├── CLAUDE.md           # Instructions for Claude Code
└── README.md           # This file
```

### External paths (created by setup.sh)

```
~/.local/whisper-cpp/whisper-cli                         # whisper.cpp binary
~/.local/whisper-cpp/models/ggml-large-v3-turbo.bin      # Whisper model (1.6 GB)
~/.local/node/bin/                                       # Node.js
~/.local/bin/claude                                      # Claude CLI
```

---

## Manual Installation

<details>
<summary>Click to expand</summary>

### 1. Python packages
```bash
pip3 install -r requirements.txt
```

### 2. whisper.cpp (macOS Apple Silicon)
```bash
git clone https://github.com/ggerganov/whisper.cpp.git ~/.local/whisper-cpp/src
cd ~/.local/whisper-cpp/src
cmake -B build -DGGML_METAL=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
cp build/bin/whisper-cli ~/.local/whisper-cpp/
```

### 3. Whisper model
```bash
curl -L -o ~/.local/whisper-cpp/models/ggml-large-v3-turbo.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
```

### 4. Node.js
```bash
# macOS with Homebrew
brew install node
# Or use setup.sh which installs to ~/.local/node/
```

### 5. Claude CLI
```bash
npm install -g @anthropic-ai/claude-code
claude  # login on first run
```

</details>

---

## Troubleshooting

| Error | Fix |
|---|---|
| `NOTION_API_KEY not set` | Fill in `.env` file |
| `Error fetching metadata` | Make sure Chrome is logged into YouTube |
| `whisper.cpp not found` | Run `bash setup.sh` or see manual install |
| `Error calling claude CLI` | Run `claude` once to login |
| `n challenge solving failed` | Node.js not installed — run `bash setup.sh` |
| Notion page missing URL | Add a "URL" property (type: URL) to your database |

---

## License

MIT
