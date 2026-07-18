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

A single-file Python CLI tool (`yt2notion.py`) that takes a YouTube URL and creates a structured Notion page containing a summary with clickable timestamps and a formatted full transcript.

## Required Configuration (`.env`)

The script reads from a `.env` file. The **actual variable names used in the code** are:

```
OPENAI_API_KEY=sk-proj-...
NOTION_API_KEY=secret_...
NOTION_DATABASE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Note: `.env.example` lists different variable names (`ANTHROPIC_API_KEY`, `NOTION_TOKEN`, `NOTION_PAGE_ID`) — these are **not** what the current `yt2notion.py` reads. Trust the code, not `.env.example`.

## Architecture

The pipeline runs in 6 sequential steps (LLM calls use Claude CLI first, gpt-4o-mini as fallback):

1. **Extract video ID** — regex against the URL
2. **Fetch metadata** — `yt-dlp --dump-json` returns title, thumbnail, language
3. **Get transcript** — 3-tier fallback strategy (see below)
4. **Frame analysis** — "watch" the video (see below); skip with `--no-frames` or `YT2NOTION_NO_FRAMES=1`
5. **Summarize + format conversation** — Claude extracts 5–10 key topics with timestamps (transcript + slide notes); the transcript is then split at those topic boundaries (`parse_summary_topics` → `format_conversation(segments, topics)`) and each section reformatted with inferred speaker labels, so every summary topic has a transcript section to anchor to
6. **Create Notion page** — batched API calls (max 100 blocks per request), slide images via File Upload API

### Frame Analysis (hybrid local/cloud, degrades gracefully)

Any failure in this step (download, ffmpeg, OCR, upload) prints a warning and the run continues with the audio-only page.

| Stage | Method | Notes |
|---|---|---|
| Video download | `yt-dlp -f "bv*[height<=720]..."` video-only stream | kept until topic frames attached (Layer 2) |
| Slide segments | 1fps 32×18 gray thumbs (one ffmpeg pass, videotoolbox) → drift segmentation | segment closes when the picture drifts >10% (changed pixels) from the segment's first frame — every settled build stage is its own segment, captured at its end; `_dedup_slides` then merges a stage whose OCR text is contained in the next stage's (`_text_contains` ≥0.8) keeping the completed image + earliest timestamp; must persist ≥5s; re-shown slides deduped (≤10% diff); capped at 40 |
| Keyframes | `ffmpeg -ss {segment capture} -frames:v 1` full-res extract per segment | captured at the last quiet moment BEFORE the segment ends = the completed slide with all elements (not mid-build); OCR-richness fallback to mid frame if end capture caught a fade; bullet timestamp = segment START |
| Read slides | Apple Vision OCR (pyobjc, local, free) | language order from transcript CJK-ratio (`_is_cjk_heavy`), not yt-dlp's often-empty metadata: zh-Hant-first for Chinese videos, else en-US-first; consecutive near-identical OCR deduped |
| Visual frames | Claude CLI reads image files (`--print` mode, cwd=frames dir) | only frames with <40 chars OCR text, capped at 10/video, batches of 5; gpt-4o-mini vision fallback; model outputs SKIP for non-informative frames (talking heads/transitions) which are dropped |
| Topic frames | `attach_topic_frames` after `summarize` | each summary topic gets the state with the longest overlap of its span `[t, next topic)` (capped at 90s); missing states extracted on demand; SKIP states excluded; summary output sanitized by `_clean_summary` (drops preamble/code fences) |
| Notion embed | File Upload API (`POST /v1/file_uploads` + `/send`) via httpx | max 25 images shared across sections; falls back to text-only bullet on upload failure |

Slide notes are injected into the summary prompt (SLIDES section). Topic-tagged slides render as an image directly under their Summary bullet; remaining informative slides render as a `投影片重點` section (timestamped bullet + embedded image) between Summary and Full Transcript (omitted when empty).

### Transcript Strategy (2-tier, NO auto-generated subtitles)

**Auto-generated subtitles are NEVER used.** Only manually uploaded subtitles or local Whisper.

| Tier | Method | Fallback trigger |
|---|---|---|
| 1 | `youtube_transcript_api` — manual subtitles ONLY (`is_generated=False`) | No manual subtitles found, or any exception |
| 2 | `yt-dlp --write-sub` — manual subtitles only (no `--write-auto-sub`) with Chrome cookies | No `.vtt` files found (excludes `.auto.vtt`) |
| 3 | `whisper.cpp large-v3` + `ggml-small.en-tdrz` (speaker diarization) | Always runs when no manual subtitles exist |

### External System Paths

- `~/.local/whisper-cpp/whisper-cli` — whisper.cpp binary
- `~/.local/whisper-cpp/models/ggml-large-v3-turbo.bin` — large-v3-turbo model (1.6GB)
- `~/.local/bin/yt-dlp` — current yt-dlp (uv tool install; the pip user-site one is frozen at Python 3.9 and too old)
- `/opt/homebrew/bin/deno` — JS runtime for yt-dlp EJS challenge solving (node 20 is too old)
- `~/.local/node/bin` — legacy Node.js, still prepended to PATH at startup

At startup the script prepends to PATH (highest priority last): `/opt/homebrew/bin`, pip user-scripts dir, `~/.local/bin` — so it works from launchd/Telegram-bot contexts with a minimal PATH.

### Notion API Constraints

- Max 100 blocks per API request — first batch in `pages.create()`, rest via `blocks.children.append()`
- Paragraph text chunked to max 1900 chars (Notion limit is 2000); splits at sentence then word boundaries
- Summary timestamps are bold blue `rich_text` linking **in-page** to their transcript section (`<page_url>#<block_id_without_dashes>`); a gray ` ▶` next to each links out to `...&t=Xs` on YouTube
- In-page anchors need block ids that exist only after creation, so `create_notion_page` creates the bullets unlinked and `link_summary_to_transcript` patches them in a second pass (`blocks.children.list` → match `[MM:SS]` → `blocks.update`). Summary bullets are identified only between the `Summary` heading and its following divider, so identically-timestamped `投影片重點` bullets are never patched

## Key Functions

(Line numbers approximate — grep the name if it has drifted.)

| Function | Line | Purpose |
|---|---|---|
| `get_youtube_transcript` | ~193 | Tier 1+2 transcript fetching |
| `transcribe_with_whisper_local` | ~441 | Tier 3 whisper.cpp fallback |
| `analyze_video_frames` | ~850 | Frame-analysis orchestrator (download → states → OCR → vision); returns frame ctx dict |
| `attach_topic_frames` | ~915 | Layer 2: tag/extract the frame on screen at each summary topic |
| `_download_video` | ~550 | Video-only ≤720p stream for frame extraction |
| `_sample_thumbs` / `_find_stable_states` | ~590 | 1fps gray thumbs → stable visual states (changed-pixel ratio) |
| `_extract_keyframes` | ~665 | states → full-res mid-state frame extraction |
| `_ocr_image` / `_ocr_languages` | ~621 | Apple Vision OCR + per-video language ordering |
| `_dedup_slides` | ~664 | Merge build stages (OCR-subset → keep completed image, earliest ts) + near-identical slides |
| `_describe_frames_with_vision` | ~682 | Claude CLI (cwd=frames dir) / gpt-4o-mini vision |
| `slides_to_text` | ~813 | Slide notes → summary-prompt SLIDES section |
| `summarize` | ~872 | Claude summarization (transcript + slide notes) |
| `format_conversation` | ~939 | Claude conversation formatting |
| `_upload_file_to_notion` | ~1137 | Notion File Upload API (httpx) → file_upload id |
| `create_notion_page` | ~1230 | Build + upload all Notion blocks incl. 投影片重點 |
| `parse_summary_topics` | ~1299 | Summary lines → `[{seconds, ts_str, title}]` (shared by sectioning + linking) |
| `summary_bullet_block` | ~1322 | Summary bullet: unlinked `[MM:SS]` + gray ▶ YouTube link |
| `link_summary_to_transcript` | ~1380 | 2nd pass: patch summary timestamps with in-page anchors |
| `bullet_block_with_timestamp_link` | ~1440 | Render clickable YouTube timestamp links (投影片重點) |

## Dependencies

**Python packages** (see `requirements.txt`): `openai`, `notion-client`, `python-dotenv`, `yt-dlp`, `youtube-transcript-api`, `imageio-ffmpeg`, `pyobjc-framework-Vision`, `pyobjc-framework-Quartz` (frame OCR, macOS only)

**System dependencies**: `whisper.cpp` compiled with Metal GPU support, `deno` (yt-dlp EJS challenges), current `yt-dlp` via `uv tool install yt-dlp`

**Python version note**: the runtime on this machine is `/usr/bin/python3` (3.9). The script carries `from __future__ import annotations` for 3.9 compatibility — keep it when editing.
