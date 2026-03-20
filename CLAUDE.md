# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A single-file Python CLI tool (`yt2notion.py`) that takes a YouTube URL and creates a structured Notion page containing a summary with clickable timestamps and a formatted full transcript.

## Running the Script

```bash
pip install -r requirements.txt
python yt2notion.py
```

The script prompts interactively for a YouTube URL — there are no CLI arguments.

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

### Transcript 3-Tier Strategy

| Tier | Method | Fallback trigger |
|---|---|---|
| 1 | `youtube_transcript_api` (prefers manual over auto-generated) | Any exception |
| 2 | `yt-dlp --write-sub` with Chrome cookies (handles members-only) | No `.vtt` files found |
| 3 | `whisper.cpp large-v3` (local, Metal GPU, ~5× realtime) | Always available as last resort |

### External System Paths

- `~/.local/whisper-cpp/whisper-cli` — whisper.cpp binary
- `~/.local/whisper-cpp/models/ggml-large-v3.bin` — large-v3 model (~2.9GB)
- `~/.local/node/bin` — added to PATH at startup for yt-dlp JS challenge solving

### Notion API Constraints

- Max 100 blocks per API request — first batch in `pages.create()`, rest via `blocks.children.append()`
- Paragraph text chunked to max 1900 chars (Notion limit is 2000); splits at sentence then word boundaries
- Summary timestamp links rendered as bold blue `rich_text` with `"link": {"url": "...&t=Xs"}`

## Key Functions

| Function | Location | Purpose |
|---|---|---|
| `get_youtube_transcript` | line 132 | Tier 1+2 transcript fetching |
| `transcribe_with_whisper_local` | line 272 | Tier 3 whisper.cpp fallback |
| `_parse_vtt` | line 208 | Parse WebVTT subtitle files |
| `summarize` | line 343 | GPT-4o-mini summarization |
| `format_conversation` | line 369 | GPT-4o-mini conversation formatting |
| `create_notion_page` | line 516 | Build and upload all Notion blocks |
| `bullet_block_with_timestamp_link` | line 467 | Render clickable timestamp links |
| `chunk_text` | line 404 | Split text for Notion's block size limit |

## Dependencies

**Python packages** (see `requirements.txt`): `openai`, `notion-client`, `python-dotenv`, `yt-dlp`, `youtube-transcript-api`, `imageio-ffmpeg`

**System dependencies**: `whisper.cpp` compiled with Metal GPU support, `Node.js` (for yt-dlp JS challenges)
