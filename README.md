# yt2notion — YouTube to Notion Summarizer

A Python CLI tool that takes any YouTube URL and automatically:
1. Fetches the video title, thumbnail, and language
2. Gets the transcript (YouTube captions → yt-dlp subtitles → whisper.cpp fallback)
3. Summarizes 5–10 key topics with clickable timestamps — **in the same language as the video**
4. Formats the full transcript as a readable conversation with smart speaker labels
5. Creates a structured Notion page with everything organized

---

## Architecture Overview

```
YouTube URL
    │
    ▼
[1] yt-dlp ──────────────────► metadata (title, thumbnail, language)
    │
    ▼
[2] Transcript (3-tier strategy)
    ├─ Tier 1: youtube_transcript_api  (fast, free, prefers manual subs)
    ├─ Tier 2: yt-dlp --write-sub     (Chrome cookies, members-only)
    └─ Tier 3: whisper.cpp large-v3   (local, Metal GPU, ~5x realtime)
    │
    ▼
[3] GPT-4o-mini ─────────────► summary (5-10 key topics with timestamps)
    │                           (language matches transcript)
    ▼
[4] GPT-4o-mini ─────────────► conversation-formatted transcript
    │                           (smart speaker labels, no timestamps)
    ▼
[5] Notion API ──────────────► structured page with all content
```

---

## What You Need

| Requirement | Purpose |
|---|---|
| Python 3.10+ | Run the script |
| OpenAI API key | Summarization + conversation formatting (GPT-4o-mini) |
| Notion API key | Create pages in your Notion workspace |
| Notion Database ID | The target database where pages will be created |
| Google Chrome | For reading YouTube cookies automatically (members-only videos) |
| whisper.cpp + large-v3 model | Local transcription fallback (when no subtitles exist) |
| Node.js | Required by yt-dlp for YouTube JS challenges |

---

## Installation

### 1. Clone or download the project
Place `yt2notion.py` in a folder, e.g. `~/yt2notion/`

### 2. Install Python dependencies
```bash
pip install openai notion-client python-dotenv yt-dlp \
            youtube-transcript-api imageio-ffmpeg
```

### 3. Install Node.js (required by yt-dlp)
```bash
# macOS with Homebrew
brew install node

# Or download directly from https://nodejs.org
```

### 4. Install whisper.cpp (local transcription engine)
```bash
# Clone and build with Metal GPU support (macOS Apple Silicon)
git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git /tmp/whisper.cpp
cd /tmp/whisper.cpp
cmake -B build -DWHISPER_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# Install binary
mkdir -p ~/.local/whisper-cpp
cp build/bin/whisper-cli ~/.local/whisper-cpp/

# Download large-v3 model (~3GB)
mkdir -p ~/.local/whisper-cpp/models
curl -L -o ~/.local/whisper-cpp/models/ggml-large-v3.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
```

> **Linux/Windows:** Omit `-DWHISPER_METAL=ON` and use `-DWHISPER_CUDA=ON` for NVIDIA GPUs, or no flag for CPU-only.

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
| **Summary** | 5–10 key topics, each with a clickable timestamp that jumps to that moment. Written in the dominant language of the video. |
| **Full Transcript** | Complete transcript formatted as a natural conversation (see speaker labels below) |

---

## How It Works — Technical Details

### Transcript fetching (3-tier strategy)

| Priority | Method | Works for | Speed |
|---|---|---|---|
| **Tier 1** | `youtube_transcript_api` — prefers manual subtitles (`is_generated=False`) over auto-generated | Public videos with captions | Instant |
| **Tier 2** | `yt-dlp --write-sub --write-auto-sub` with `--cookies-from-browser chrome` — downloads VTT subtitle files | Members-only videos with subtitles | ~5s |
| **Tier 3** | `whisper.cpp` large-v3 with Metal GPU — downloads audio, converts to 16kHz WAV, transcribes locally | Any video with no subtitles | ~1min per 10min video |

**How to tell which method was used** (from terminal output):
- `Found manual subtitles in 'Chinese'` → Tier 1 or 2, manually uploaded
- `Found auto-generated subtitles` → Tier 1 or 2, auto-generated
- `Transcribing with whisper.cpp large-v3 (Metal GPU)` → Tier 3

### VTT subtitle parsing
When yt-dlp downloads subtitle files, the parser:
- Splits by blank lines to find cue blocks
- Extracts timestamps via regex: `HH:MM:SS.mmm --> HH:MM:SS.mmm`
- Strips VTT formatting tags (`<c>`, `<00:00:00.000>`, `</c>`)
- Deduplicates overlapping cues (common in auto-generated subtitles)
- Prefers manually uploaded files (no `.auto.` in filename) over auto-generated

### whisper.cpp transcription
- Binary location: `~/.local/whisper-cpp/whisper-cli`
- Model location: `~/.local/whisper-cpp/models/ggml-large-v3.bin` (2.9GB, f16 precision)
- Audio is converted to 16kHz mono PCM WAV via the bundled `imageio-ffmpeg` binary
- Runs with `-l auto` to auto-detect language per segment (handles mixed English/Chinese)
- Outputs JSON with timestamps in `HH:MM:SS.mmm` format, parsed to seconds
- ~20x faster than Python whisper on Apple Silicon via Metal GPU

### Summarization (OpenAI GPT-4o-mini)
- Converts segments to timestamped text: `[MM:SS] segment text`
- Sends to GPT-4o-mini with instructions to:
  - Detect the dominant language and write summary in that language
  - Extract 5–10 key topics with timestamps
  - Format: `[MM:SS] Topic title: brief description`

### Conversation formatting (OpenAI GPT-4o-mini)
Speaker labels are assigned in this priority order:
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
- Rendered in Notion as bold blue hyperlinks using rich_text with link + annotations

### Members-only videos
- The script reads cookies directly from **Google Chrome** using `yt-dlp --cookies-from-browser chrome`
- You must be logged into YouTube in Chrome and be a channel member
- No manual cookie export needed — cookies are always fresh
- Also passes `--remote-components ejs:github` for YouTube JS challenge solving

### Notion API details
- Notion's API accepts max 100 blocks per request
- The script creates the page with the first 100 blocks, then appends remaining in batches
- Paragraph text is chunked to max 1900 characters (Notion's block limit is 2000)
- Sentence boundary splitting is preferred over mid-word splits

---

## Prompt for AI Replication

If you want to ask an AI coding assistant (e.g. OpenAI Codex, ChatGPT, Claude, Cursor) to build this tool from scratch, use this prompt:

---

> Build a Python CLI script called `yt2notion.py` that takes a YouTube URL (prompted via `input()`) and creates a Notion page with the video's transcript and summary. Here are the exact requirements:
>
> **Metadata (yt-dlp):**
> - Use `yt-dlp --dump-json --no-playlist` to fetch title, thumbnail URL, and language
> - Pass `--cookies-from-browser chrome` and `--remote-components ejs:github` for all yt-dlp calls
> - Pass `--ffmpeg-location` pointing to the ffmpeg binary from `imageio_ffmpeg.get_ffmpeg_exe()`
>
> **Transcript fetching (3-tier strategy):**
> - Tier 1: Use `youtube-transcript-api` (v1.2.4+). Instantiate `YouTubeTranscriptApi()`, call `.list(video_id)` to get all transcripts, prefer manually created ones (`is_generated=False`) over auto-generated. Call `.fetch()` on the chosen transcript. Store segments as `{"text": str, "start": float}`.
> - Tier 2: If Tier 1 fails (e.g. members-only video), use `yt-dlp --skip-download --write-sub --write-auto-sub --sub-langs all --sub-format vtt` with `--cookies-from-browser chrome`. Parse the downloaded `.vtt` file: split by blank lines, extract timestamps (regex `HH:MM:SS.mmm --> HH:MM:SS.mmm`), strip VTT tags (`<c>`, `<HH:MM:SS.mmm>`, `</c>`) with `re.sub(r"<[^>]+>", "", line)`, deduplicate overlapping cues via a `seen_texts` set. Prefer non-`.auto.` files (manual subs).
> - Tier 3: If no subtitles found, download audio with `yt-dlp -f bestaudio[ext=m4a]/bestaudio`. Convert to 16kHz mono WAV using ffmpeg (`-ar 16000 -ac 1 -c:a pcm_s16le`). Transcribe with `whisper.cpp` CLI: `whisper-cli -m MODEL_PATH -f WAV_PATH -l auto --output-json -of OUTPUT_PREFIX`. Parse the output JSON: iterate `data["transcription"]`, extract `item["text"]` and parse `item["timestamps"]["from"]` (format `HH:MM:SS.mmm`) to seconds.
> - whisper.cpp binary at `~/.local/whisper-cpp/whisper-cli`, model at `~/.local/whisper-cpp/models/ggml-large-v3.bin`
>
> **Summarization (OpenAI GPT-4o-mini):**
> - Convert segments to timestamped string: `[MM:SS] text`
> - Send to GPT-4o-mini with `max_tokens=1024`
> - Prompt must instruct: detect the dominant language of the transcript and write the entire summary in that same language
> - Extract 5–10 key topics, each with a timestamp and 1–2 sentence description
> - Format: `[MM:SS] Topic title: brief description` (one per line)
>
> **Conversation formatting (OpenAI GPT-4o-mini):**
> - Send the timestamped transcript to GPT-4o-mini with `max_tokens=16384`
> - Speaker labeling priority: (1) real names if mentioned, (2) role-based labels inferred from context (Host/Guest for interviews, Interviewer/Guest for Q&A, Instructor for lectures, Narrator for documentaries, Host 1/Host 2 for dual hosts), (3) Person 1/Person 2 as last resort, (4) no labels for single-speaker videos (clean paragraphs only)
> - Remove timestamps, merge consecutive same-speaker lines, keep ALL content (no summarizing)
>
> **Notion page structure (using `notion-client`):**
> - Create a page in a Notion database (ID from .env)
> - Blocks in order: bookmark (YouTube URL), image (thumbnail), divider, heading_2 "Summary", bulleted_list_items of key topics, divider, heading_2 "Full Transcript", paragraphs of conversation text
> - Each `[MM:SS]` timestamp in the summary bullets must be a clickable YouTube link (`?v=ID&t=Xs`), rendered as bold blue text using Notion rich_text with `"link": {"url": yt_link}` and `"annotations": {"bold": true, "color": "blue"}`
> - Chunk paragraph text to max 1900 characters, splitting at sentence boundaries (`. `) then word boundaries
> - Use batched API calls: first 100 blocks in `pages.create()`, remaining in `blocks.children.append()` batches of 100
>
> **Configuration:**
> - Load `OPENAI_API_KEY`, `NOTION_API_KEY`, `NOTION_DATABASE_ID` from a `.env` file using `python-dotenv`
> - Add `~/.local/node/bin` to PATH at startup for yt-dlp JS runtime support
> - Get ffmpeg path via `imageio_ffmpeg.get_ffmpeg_exe()` with fallback to `"ffmpeg"`
>
> **Dependencies:** `openai`, `notion-client`, `python-dotenv`, `yt-dlp`, `youtube-transcript-api`, `imageio-ffmpeg`
> **System dependencies:** `whisper.cpp` (compiled with Metal/CUDA), `Node.js`

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
| `whisper.cpp not found` | Follow the whisper.cpp installation steps above |
| `Whisper model not found` | Download the ggml-large-v3.bin model |
| Notion page missing content | Check that your integration is connected to the database |

---

## File Structure

```
~/yt2notion/
├── yt2notion.py          # Main script (single file, ~460 lines)
├── .env                  # API keys (not committed)
├── README.md             # This file
└── cookies.txt           # (optional, legacy — now uses Chrome cookies directly)
```

## External paths used

```
~/.local/whisper-cpp/whisper-cli                    # whisper.cpp binary
~/.local/whisper-cpp/models/ggml-large-v3.bin       # Whisper large-v3 model (2.9GB)
~/.local/node/bin/node                              # Node.js (for yt-dlp)
```
