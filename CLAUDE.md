# CLAUDE.md

## IMPORTANT: YouTube → Notion Command

**When the user pastes a YouTube URL (youtube.com, youtu.be) and mentions Notion, saving, summarizing, or transcribing — IMMEDIATELY run the command below. Do NOT search the web, do NOT try to fetch the video yourself, do NOT use MCP tools. Just run this bash command:**

```bash
cd /Users/luke-mini/Claude/Tools/yt2notion && echo "YOUTUBE_URL_HERE" | python3 yt2notion.py
```

Replace `YOUTUBE_URL_HERE` with the actual URL from the user's message. Set a timeout of 600000ms (10 minutes) since long videos take time to process. Report the Notion page URL when done.

**Trigger phrases:** "save to Notion", "summarize this video", "transcribe to Notion", "yt2notion", or any YouTube URL + Notion intent.

---

## What This Is

Two complementary CLI tools:

- **`yt2notion.py`** — takes a YouTube URL → Notion page with summary (clickable timestamps) + formatted transcript
- **`audio2notion.py`** — takes a local audio file (m4a, mp3, wav, etc.) → Notion page with summary + formatted transcript

The Telegram bot (`telegram_bot.py`) supports both: paste a YouTube URL **or** drop an m4a file into the channel.

## Required Configuration (`.env`)

The script reads from a `.env` file. The **actual variable names used in the code** are:

```
OPENAI_API_KEY=sk-proj-...
NOTION_API_KEY=secret_...
NOTION_DATABASE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Note: `.env.example` lists different variable names (`ANTHROPIC_API_KEY`, `NOTION_TOKEN`, `NOTION_PAGE_ID`) — these are **not** what the current `yt2notion.py` reads. Trust the code, not `.env.example`.

## Architecture

The pipeline runs in 6 sequential steps:

1. **Extract video ID** — regex against the URL
2. **Fetch metadata** — `yt-dlp --dump-json` returns title, thumbnail, language
3. **Get transcript** — 3-tier fallback strategy (see below)
4. **Summarize** — GPT-4o-mini extracts 5–10 key topics with timestamps; language auto-matches transcript
5. **Format conversation** — GPT-4o-mini reformats transcript with inferred speaker labels
6. **Create Notion page** — batched API calls (max 100 blocks per request)

### Transcript Strategy (2-tier, NO auto-generated subtitles)

**Auto-generated subtitles are NEVER used.** Only manually uploaded subtitles or local Whisper.

| Tier | Method | Fallback trigger |
|---|---|---|
| 1 | `youtube_transcript_api` — manual subtitles ONLY (`is_generated=False`) | No manual subtitles found, or any exception |
| 2 | `yt-dlp --write-sub` — manual subtitles only (no `--write-auto-sub`) with Chrome cookies | No `.vtt` files found (excludes `.auto.vtt`) |
| 3 | `whisper.cpp large-v3` + `ggml-small.en-tdrz` (speaker diarization) | Always runs when no manual subtitles exist |

### External System Paths

- `~/.local/whisper-cpp/whisper-cli` — whisper.cpp binary
- `~/.local/whisper-cpp/models/ggml-large-v3.bin` — large-v3 model (~2.9GB)
- `~/.local/node/bin` — added to PATH at startup for yt-dlp JS challenge solving

### Notion API Constraints

- Max 100 blocks per API request — first batch in `pages.create()`, rest via `blocks.children.append()`
- Paragraph text chunked to max 1900 chars (Notion limit is 2000); splits at sentence then word boundaries
- Summary timestamp links rendered as bold blue `rich_text` with `"link": {"url": "...&t=Xs"}`

## audio2notion.py — Local Audio File Pipeline

```
python3 audio2notion.py /path/to/file.m4a
python3 audio2notion.py /path/to/file.m4a --title "Meeting Title"
echo "/path/to/file.m4a" | python3 audio2notion.py
```

Accepts: `.m4a`, `.mp3`, `.wav`, `.ogg`, `.flac`, `.mp4`

Pipeline (4 steps):
1. **Convert** — ffmpeg → 16kHz mono WAV
2. **Transcribe** — whisper.cpp large-v3 (Metal GPU)
3. **Summarize + Format** — Claude CLI (same prompts as yt2notion)
4. **Notion page** — Summary bullets + Full Transcript (no YouTube bookmark/thumbnail)

No `OPENAI_API_KEY` needed — uses Claude CLI session like yt2notion.

## telegram_bot.py — Supported Inputs

| Input | Handler |
|---|---|
| YouTube URL (text) | `handle_message` → runs `yt2notion.py` |
| Audio file / voice note | `handle_audio` → downloads → runs `audio2notion.py` |
| Document with audio extension | `handle_audio` → same |

## Key Functions in yt2notion.py

| Function | Location | Purpose |
|---|---|---|
| `get_youtube_transcript` | line 144 | Tier 1+2 transcript fetching |
| `transcribe_with_whisper_local` | line 361 | Tier 3 whisper.cpp fallback |
| `_parse_vtt` | line 225 | Parse WebVTT subtitle files |
| `summarize` | line 416 | Claude CLI summarization |
| `format_conversation` | line 482 | Claude CLI conversation formatting |
| `create_notion_page` | line 690 | Build and upload all Notion blocks |
| `bullet_block_with_timestamp_link` | line 641 | Render clickable timestamp links |
| `chunk_text` | line 578 | Split text for Notion's block size limit |

## Dependencies

**Python packages** (see `requirements.txt`): `openai`, `notion-client`, `python-dotenv`, `yt-dlp`, `youtube-transcript-api`, `imageio-ffmpeg`

**System dependencies**: `whisper.cpp` compiled with Metal GPU support, `Node.js` (for yt-dlp JS challenges)
