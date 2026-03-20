# yt2notion — YouTube to Notion Summarizer

A Python CLI tool that takes any YouTube URL and automatically:
1. Fetches the video title and thumbnail
2. Gets the transcript (YouTube captions if available, otherwise transcribes locally with Whisper)
3. Summarizes 5–10 key topics with clickable timestamps — **in the same language as the video**
4. Formats the full transcript as a readable conversation with smart speaker labels
5. Creates a structured Notion page with everything organized

---

## What You Need

| Requirement | Purpose |
|---|---|
| Python 3.10+ | Run the script |
| OpenAI API key | Summarization + conversation formatting (GPT-4o-mini) |
| Notion API key | Create pages in your Notion workspace |
| Notion Database ID | The target database where pages will be created |
| Google Chrome | For reading YouTube cookies automatically (members-only videos) |

---

## Installation

### 1. Clone or download the project
Place `yt2notion.py` in a folder, e.g. `~/yt2notion/`

### 2. Install Python dependencies
```bash
pip install openai notion-client python-dotenv yt-dlp \
            youtube-transcript-api openai-whisper \
            imageio-ffmpeg numpy
```

### 3. Install Node.js (required by yt-dlp for YouTube JS challenges)
Download from https://nodejs.org and install, or:
```bash
# macOS with Homebrew
brew install node
```
> Without Node.js, yt-dlp may fail on some videos with a "n challenge solving failed" error.

---

## Configuration

Create a `.env` file in the same folder as `yt2notion.py`:

```
OPENAI_API_KEY=sk-proj-...
NOTION_API_KEY=secret_...
NOTION_DATABASE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### How to get each key

#### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Click **Create new secret key**
3. Copy the key (starts with `sk-proj-` or `sk-`)

#### Notion API Key
1. Go to https://www.notion.so/my-integrations
2. Click **New integration**
3. Give it a name (e.g. "YouTube Summarizer"), select your workspace
4. Copy the **Internal Integration Token** (starts with `secret_`)
5. Open your target Notion database → click `...` menu → **Add connections** → select your integration

#### Notion Database ID
1. Open your Notion database in the browser
2. The URL looks like: `https://www.notion.so/yourworkspace/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx?v=...`
3. Copy the 32-character hex string between the last `/` and `?` — that is your Database ID

---

## Usage

```bash
cd ~/yt2notion
python yt2notion.py
```

You will be prompted to paste a YouTube URL:
```
YouTube → Notion Summarizer
========================================
Paste YouTube URL: https://www.youtube.com/watch?v=VIDEO_ID
```

The script runs 6 steps and prints progress:
```
[1/6] Extracting video ID...
[2/6] Fetching metadata...
[3/6] Getting transcript...
[4/6] Generating summary...
[5/6] Formatting transcript as conversation...
[6/6] Creating Notion page...

========================================
Done!
Notion page: https://www.notion.so/...
```

---

## What the Notion Page Contains

| Section | Description |
|---|---|
| **YouTube bookmark** | Clickable link to the original video |
| **Thumbnail** | Video thumbnail image |
| **Summary** | 5–10 key topics, each with a clickable timestamp that jumps to that moment in the video. Written in the same language as the video (Chinese for Chinese videos, English for English, etc.) |
| **Full Transcript** | Complete transcript formatted as a natural conversation with smart speaker labels (see below) |

---

## How It Works — Technical Details

### Transcript fetching (3-tier strategy)

| Priority | Method | Works for |
|---|---|---|
| **1st** | `youtube-transcript-api` — prefers manually uploaded subtitles over auto-generated | Public videos with captions |
| **2nd** | `yt-dlp --write-sub` with Chrome cookies — downloads subtitle file directly | Members-only videos with subtitles |
| **3rd** | Local Whisper large-v3 — downloads audio and transcribes on your machine | Any video with no subtitles |

You can tell which method was used from the terminal output:
- `Found manual subtitles in 'Chinese'` → Method 1 or 2, manually uploaded
- `Found auto-generated subtitles` → Method 1 or 2, auto-generated
- `Transcribing with local Whisper model` → Method 3

### Summarization (OpenAI GPT-4o-mini)
- Converts segments to timestamped text: `[MM:SS] segment text`
- Sends to GPT-4o-mini to extract 5–10 key topics with timestamps
- **Automatically detects the dominant language** and writes the summary in that language

### Conversation formatting (OpenAI GPT-4o-mini)
Speaker labels are assigned intelligently in this priority order:
1. **Real names** — if mentioned in the transcript
2. **Role-based labels** — inferred from context:
   - Interview / podcast → `Host` and `Guest`
   - Q&A format → `Interviewer` and `Guest`
   - Two hosts → `Host 1` and `Host 2`
   - Lecture → `Instructor`
   - Documentary → `Narrator`
3. **`Person 1` / `Person 2`** — last resort only
4. **No labels** — for solo speaker / monologue videos (clean paragraphs only)

### Clickable timestamps
- Each `[MM:SS]` in the summary is converted to a YouTube deep-link: `https://youtube.com/watch?v=VIDEO_ID&t=Xs`
- Rendered in Notion as bold blue hyperlinks

### Members-only videos
- The script reads cookies directly from **Google Chrome** using `yt-dlp --cookies-from-browser chrome`
- You must be logged into YouTube in Chrome and be a channel member
- No manual cookie export needed — cookies are always fresh

### Notion API batching
- Notion's API accepts max 100 blocks per request
- The script automatically splits large transcripts into batches of 100 blocks

---

## Prompt for AI Replication

If you want to ask an AI coding assistant (e.g. ChatGPT, Codex) to build this tool from scratch, use this prompt:

---

> Build a Python CLI script called `yt2notion.py` that takes a YouTube URL (prompted via input) and creates a Notion page with the video's transcript and summary. Here are the exact requirements:
>
> **Transcript fetching (3-tier strategy):**
> - Tier 1: Use `youtube-transcript-api` (v1.2.4+). Instantiate `YouTubeTranscriptApi()`, call `.list(video_id)` to get all transcripts, prefer manually created ones (`is_generated=False`) over auto-generated. Call `.fetch()` on the chosen transcript and store segments as `{"text": str, "start": float}`.
> - Tier 2: If Tier 1 fails (e.g. members-only video), use `yt-dlp --skip-download --write-sub --write-auto-sub --sub-langs all --sub-format vtt` with `--cookies-from-browser chrome` and `--remote-components ejs:github`. Parse the downloaded `.vtt` file: split by blank lines, extract timestamps (regex `HH:MM:SS.mmm --> HH:MM:SS.mmm`), strip VTT tags (`<c>`, `<00:00:00.000>`, `</c>`), deduplicate overlapping cues. Prefer non-`.auto.` files.
> - Tier 3: If no subtitles found, download audio with `yt-dlp` (`bestaudio[ext=m4a]/bestaudio` format) and transcribe locally using OpenAI Whisper `large-v3` model. Use `imageio-ffmpeg` to get the ffmpeg path, convert audio to raw PCM f32le via subprocess, load as numpy array, pass directly to `whisper.transcribe()` with `fp16=False`.
> - Use `--cookies-from-browser chrome` and `--remote-components ejs:github` for all yt-dlp calls.
>
> **Summarization (OpenAI GPT-4o-mini):**
> - Convert segments to timestamped string: `[MM:SS] text`
> - Send to GPT-4o-mini to extract 5–10 key topics with timestamps and 1–2 sentence descriptions
> - Instruct GPT to detect the dominant language of the transcript and write the summary in that same language
> - Format: `[MM:SS] Topic title: brief description` (one per line)
>
> **Conversation formatting (OpenAI GPT-4o-mini):**
> - Send the timestamped transcript to GPT-4o-mini
> - Speaker labeling priority: (1) real names if mentioned, (2) role-based labels inferred from context (Host/Guest for interviews, Interviewer/Guest for Q&A, Instructor for lectures, Narrator for documentaries), (3) Person 1/Person 2 as last resort, (4) no labels for single-speaker videos
> - Remove timestamps, merge consecutive same-speaker lines, keep ALL content
>
> **Notion page structure (using notion-client):**
> - Create a page in a Notion database (ID from .env)
> - Blocks in order: bookmark (YouTube URL), image (thumbnail), divider, heading "Summary", bullet list of key topics, divider, heading "Full Transcript", paragraphs of conversation text
> - Each `[MM:SS]` timestamp in summary bullets must be a clickable YouTube link (`?v=ID&t=Xs`), rendered bold and blue using Notion rich_text with a link object
> - Chunk paragraph text to max 1900 characters to respect Notion's block limit
> - Use batched API calls (max 100 blocks per request) for large transcripts
>
> **Configuration:**
> - Load `OPENAI_API_KEY`, `NOTION_API_KEY`, `NOTION_DATABASE_ID` from a `.env` file using `python-dotenv`
> - Add `~/.local/node/bin` to PATH at startup for yt-dlp JS runtime support
>
> **Dependencies:** `openai`, `notion-client`, `python-dotenv`, `yt-dlp`, `youtube-transcript-api`, `openai-whisper`, `imageio-ffmpeg`, `numpy`

---

## Troubleshooting

| Error | Fix |
|---|---|
| `OPENAI_API_KEY not set` | Add key to `.env` file |
| `NOTION_API_KEY not set` | Add key to `.env` file |
| `Error fetching metadata` | Make sure Chrome is open and logged into YouTube |
| `cookies are no longer valid` | Refresh YouTube in Chrome, then retry |
| `members-only video` | You must be a paying channel member in Chrome |
| `n challenge solving failed` | Install Node.js |
| Whisper is very slow | Normal on CPU — `large-v3` takes 10–30 min for long videos. Use a Mac with Apple Silicon for faster inference. |
| Notion page missing content | Check that your integration is connected to the database |
