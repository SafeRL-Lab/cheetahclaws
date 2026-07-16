# Voice, Video, and TTS

## Voice Input

<div align=center>
<img src="../media/demos/voice_demo.gif" width="850"/>
</div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Voice Input: speak your prompt → offline Whisper transcription → AI responds</center>
</div>

CheetahClaws v3.05 adds a fully offline voice-to-prompt pipeline. Speak your request — it is transcribed and submitted as if you had typed it.

### Quick start

```bash
# 1. Install a recording backend (choose one)
pip install sounddevice        # recommended: cross-platform, no extra binary
# sudo apt install alsa-utils  # Linux arecord fallback
# sudo apt install sox         # SoX rec fallback

# 2. Install a local STT backend (recommended — works offline, no API key)
pip install faster-whisper numpy

# 3. Start CheetahClaws and speak
cheetahclaws
[myproject] ❯ /voice
  🎙  Listening… (speak now, auto-stops on silence, Ctrl+C to cancel)
  🎙  ████
✓  Transcribed: "fix the authentication bug in user.py"
[auto-submitting…]
```

### STT backends (tried in order)

| Backend | Install | Notes |
|---|---|---|
| `faster-whisper` | `pip install faster-whisper` | **Recommended** — local, offline, fastest, GPU optional |
| `openai-whisper` | `pip install openai-whisper` | Local, offline, original OpenAI model |
| OpenAI Whisper API | set `OPENAI_API_KEY` | Cloud, requires internet + API key |

Override the Whisper model size with `NANO_CLAUDE_WHISPER_MODEL` (default: `base`):

```bash
export NANO_CLAUDE_WHISPER_MODEL=small   # better accuracy, slower
export NANO_CLAUDE_WHISPER_MODEL=tiny    # fastest, lightest
```

### Recording backends (tried in order)

| Backend | Install | Notes |
|---|---|---|
| `sounddevice` | `pip install sounddevice` | **Recommended** — cross-platform, Python-native |
| `arecord` | `sudo apt install alsa-utils` | Linux ALSA, no pip needed |
| `sox rec` | `sudo apt install sox` / `brew install sox` | Built-in silence detection |

### Keyterm boosting

Before each recording, CheetahClaws extracts coding vocabulary from:
- **Git branch** (e.g. `feat/voice-input` → "feat", "voice", "input")
- **Project root name** (e.g. "cheetahclaws")
- **Recent source file stems** (e.g. `authentication_handler.py` → "authentication", "handler")
- **Global coding terms**: `MCP`, `grep`, `TypeScript`, `OAuth`, `regex`, `gRPC`, …

These are passed as Whisper's `initial_prompt` so the STT engine prefers correct spellings of coding terms.

### Commands

| Command | Description |
|---|---|
| `/voice` | Record voice and auto-submit the transcript as your next prompt |
| `/voice status` | Show which recording and STT backends are available, plus the active microphone |
| `/voice lang <code>` | Set transcription language (`en`, `zh`, `ja`, `de`, `fr`, … default: `auto`) |
| `/voice device` | List all available input microphones and select one interactively; persisted for the session |

### Selecting a microphone

On systems with multiple audio interfaces (USB headsets, virtual devices, etc.) you can pick the exact input device:

```
[myproject] ❯ /voice device
  🎙  Available input devices:
    0. Built-in Microphone
    1. USB Headset (USB Audio)  ◀  (currently selected)
    2. Virtual Input (BlackHole)
  Select device # (Enter to cancel): 1
✓  Microphone set to: [1] USB Headset (USB Audio)
```

The selected device is shown in `/voice status` and used for all subsequent recordings until you change it or restart.

### How it compares to Claude Code

| | Claude Code | CheetahClaws v3.05 |
|---|---|---|
| STT service | Anthropic private WebSocket (`voice_stream`) | `faster-whisper` / `openai-whisper` / OpenAI API |
| Requires Anthropic OAuth | Yes | **No** |
| Works offline | No | **Yes** (with local Whisper) |
| Keyterm hints | Deepgram `keyterms` param | Whisper `initial_prompt` (git + files + vocab) |
| Language support | Server-allowlisted codes | Any language Whisper supports |

---


## Video Content Factory

<div align=center>
<img src="../media/demos/video_demo.gif" width="850"/>
</div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Video Factory: topic → AI story → TTS → images → subtitles → final .mp4</center>
</div>

`/video` is an AI-powered viral video pipeline. Give it a topic — or your own script — and it produces a fully narrated, illustrated, subtitle-burned `.mp4` ready to upload.

```
[AI mode]     Topic → AI Story → TTS Voice → Images → PIL Subtitles → Final Video
[Script mode] Your Text → TTS Voice → Images → PIL Subtitles (same text) → Final Video
```

### Quick start (zero-cost path)

```bash
# Install free dependencies
pip install edge-tts Pillow imageio-ffmpeg
sudo apt install ffmpeg          # or: brew install ffmpeg / conda install ffmpeg

# Launch interactive wizard
[myproject] ❯ /video
```

The wizard walks you through every setting with `Enter = Auto` defaults at every step. Type `b` to go back, `q` to quit at any point.

### Wizard walkthrough

```
╭─ 🎬 Video Content Factory ─────────────────────╮
│  Enter=Auto on every step  ·  b=back  ·  q=quit │
╰─────────────────────────────────────────────────╯

[0] Content mode
  1. Auto         (AI generates story from your topic)
  2. Custom script (you provide the text — TTS reads it as narration + subtitles)

[1] Topic / idea        ← skip if using custom script
[2] Source folder       ← optional: images / audio / video / text files
[3] Language            ← auto-detects from topic; supports custom language entry
[4] Style / Niche       ← 10 viral niches + auto-viral + custom style
[5] Format              ← Landscape 16:9 (YouTube) or Short 9:16 (TikTok / Reels)
[6] Duration            ← 30s · 1 min · 2 min · 3 min · 5 min · custom
[7] Voice (TTS)         ← auto / Edge (free) / Gemini / ElevenLabs
[8] Images              ← auto / web-search / gemini-web / placeholder
[9] Video Quality       ← auto / high / medium / low / minimal
[10] Subtitles          ← Auto (Whisper) / Story text / Custom text / None
[11] Output path        ← default: ./video_output/
```

#### Content mode: Custom script

Select **"2. Custom script"** to provide your own narration text instead of having the AI generate a story:

```
[0] Content mode
  Pick mode: 2

  Paste your narration text (type END on a new line when done):
  CheetahClaws is a lightweight Python AI coding assistant
  that supports any model — Claude, GPT, Gemini, or local Ollama.
  END
  → Script: 18 words
```

The TTS engine reads the script aloud. The same text is split into timed subtitle entries and burned into the video with PIL. No Whisper, no AI story generation — works fully offline.

Steps skipped in script mode: Topic, Style/Niche, Duration (auto-derived from word count).

### Pipeline steps

| Step | What happens |
|---|---|
| **1. Story / Script** | AI generates viral story (AI mode) OR uses your text directly (script mode) |
| **2. Voice (TTS)** | Edge TTS / Gemini TTS / ElevenLabs narrates the text |
| **3. Subtitles** | PIL renders subtitles as transparent PNGs; ffmpeg overlays them — works for any language |
| **4. Images** | Gemini Web (Imagen 3) → web search (Pexels / Wikimedia) → placeholder |
| **5. Assembly** | zoompan clips + audio → two-pass encode with PIL subtitle burn |

### Subtitle engine

Subtitles are rendered with **Pillow + NotoSansSC font** — not libass. This means:

- Chinese, Japanese, Korean, Cyrillic, Arabic, Thai all render correctly
- Font is downloaded once to `~/.cheetahclaws/fonts/` on first run (~8 MB)
- Two-pass approach: fast `-c:v copy` assembly, then PIL PNG overlays via `filter_complex`
- Falls back to no subtitles if PIL fails — never crashes the pipeline

**Subtitle source options** (wizard step 10):

| Option | How | Best for |
|---|---|---|
| Auto | Whisper transcription (`faster-whisper`) | When exact word timing matters |
| Story text | Same text TTS reads, timed proportionally | All languages; no Whisper needed |
| Custom text | Paste your own text | Translations, alternate language |
| None | Skip subtitles | Music videos, no-sub content |

### Image backends (vision input)

<div align=center>
<img src="../media/demos/image_demo.gif" width="850"/>
</div>
<div align=center>
<center style="color:#000000;text-decoration:underline">/image: paste UI screenshot → AI flags issues; paste code screenshot → AI spots bugs</center>
</div>

| Engine | How | Cost | Quality |
|---|---|---|---|
| `gemini-web` | Playwright + Imagen 3 via Gemini web | **Free** | High |
| `web-search` | Pexels → Wikimedia Commons → Picsum | **Free** | Medium |
| `placeholder` | Gradient slides with prompt text | **Free** | N/A |
| `auto` | gemini-web → web-search → placeholder | — | Best available |

**Gemini Web images (recommended free path):**

One-time login (session is saved):

```bash
cd ../v-content-creator
python -c "from gemini_image_gen import verify_login_interactive; verify_login_interactive()"
```

**Web search images** work out-of-the-box with no login or API key. The model generates optimized search queries from the story/script content. Sources tried in order: Pexels → Wikimedia Commons → Lorem Picsum (always succeeds).

**AI source image selection:** when `--source <dir>` contains more images than needed, the model reads filenames and story content to rank and select the most relevant ones. Keyword-scoring fallback if the model is unavailable.

### TTS backends

| Engine | How | Cost | Quality |
|---|---|---|---|
| `gemini` | Gemini TTS API (`GEMINI_API_KEY`) | Free tier | Good |
| `elevenlabs` | ElevenLabs REST (`ELEVENLABS_API_KEY`) | Paid | Excellent |
| `edge` | Microsoft Edge TTS (`pip install edge-tts`) | **Free** | Good |
| `auto` | Try gemini → elevenlabs → edge | — | Best available |

Language-appropriate voices are auto-selected (e.g. `zh-CN-YunxiNeural` for Chinese, `ja-JP-KeitaNeural` for Japanese).

### Content niches (AI mode)

10 built-in viral content niches, weighted toward the most viral:

| Niche ID | Name | Style |
|---|---|---|
| `misterio_real` | True Crime | Documentary, investigative |
| `confesiones` | Dark Confessions | Intimate, vulnerable |
| `suspenso_cotidiano` | Everyday Suspense | Mundane → disturbing |
| `ciencia_ficcion` | Sci-Fi / Black Mirror | Near-future, tech noir |
| `drama_humano` | Human Drama | Emotional, raw |
| `terror_psicologico` | Psychological Horror | Insidious, ambiguous |
| `folklore_latam` | Latin American Folklore | Magical realism |
| `venganza` | Revenge / Poetic Justice | Calculated, satisfying |
| `supervivencia` | Survival Stories | Adrenaline, extreme |
| `misterio_digital` | Digital Mystery | Internet creepy, cyber horror |

Story generation uses a 3-tier fallback: structured prompt → simplified structured → free-form, ensuring a story is always produced even with small local models.

### Source materials (`--source`)

Pass `--source <dir>` (or enter path in the wizard) to pre-load your own materials:

| File type | Behaviour |
|---|---|
| Images (`.jpg`, `.png`, …) | Used directly instead of AI/web-search images; model selects most relevant |
| Audio (`.mp3`, `.wav`) | Used as narration, skipping TTS |
| Video (`.mp4`, `.mov`, …) | Audio track extracted and used as narration; frames extracted as images |
| Text (`.txt`, `.md`, …) | Read and injected as story context / topic direction |

A single file (e.g. a README or script) can also be passed — it is read and injected as context.

### Output files

```
video_output/
├── video_20260407_153000_my_title.mp4        # Final video
└── video_20260407_153000_my_title_info.json  # Metadata (title, niche, word count, engines)

video_tmp/batch_20260407_153000/story/
├── story.txt     # Story or script text
├── audio.mp3     # TTS narration
├── subs.srt      # Subtitle file (if generated)
└── images/       # img_00.png … img_07.png
```

### Requirements summary

| Requirement | Install | Notes |
|---|---|---|
| `ffmpeg` | `sudo apt install ffmpeg` or `pip install imageio-ffmpeg` | Required |
| `Pillow` | `pip install Pillow` | Required for subtitle rendering + images |
| `edge-tts` | `pip install edge-tts` | Free TTS (recommended) |
| `faster-whisper` | `pip install faster-whisper` | Auto subtitle transcription (optional) |
| `playwright` | `pip install playwright && playwright install chromium` | Gemini Web images (optional) |
| `GEMINI_API_KEY` | env var | Gemini TTS + story generation |
| `ELEVENLABS_API_KEY` | env var | ElevenLabs TTS (optional) |

---


## TTS Content Factory

<div align=center>
<img src="../media/demos/voice_demo.gif" width="850"/>
</div>
<div align=center>
<center style="color:#000000;text-decoration:underline">TTS Factory: choose voice style → AI writes script → synthesize → .mp3 output</center>
</div>

`/tts` is an AI-powered audio generation wizard. Give it a topic — or paste your own script — and it produces a narrated MP3 in any voice style.

### Quick start

```bash
# Install free TTS backend (no API key needed)
pip install edge-tts

# Launch interactive wizard
[myproject] ❯ /tts
```

The wizard walks through every setting with `Enter = Auto` at every step. Type `b` to go back, `q` to quit.

### Wizard walkthrough

```
╭─ 🎙 TTS Content Factory ────────────────────────────────╮
│  Enter=Auto on every step  ·  b=back  ·  q=quit         │
╰─────────────────────────────────────────────────────────╯

[0] Content mode
  1. Auto         (AI generates script from your topic)
  2. Custom text  (paste your own script → TTS reads every word)

[1] Voice style   ← narrator / newsreader / storyteller / ASMR / motivational /
                     documentary / children / podcast / meditation / custom
[2] Duration      ← Auto~1 min / 30s / 1m / 2m / 3m / 5m / custom  (AI mode only)
[3] TTS Engine    ← Auto / Edge (free) / Gemini / ElevenLabs
[4] Voice         ← Auto (style preset) / individual Gemini or Edge voice
[5] Output folder ← default: ./tts_output/
```

Output files:

```
tts_output/
├── tts_1712345678.mp3          # synthesized audio
└── tts_1712345678_script.txt   # companion script text
```

### Voice style presets

| Style | Description | Default Gemini voice | Default Edge voice |
|---|---|---|---|
| Narrator | Calm, authoritative | Charon | en-US-GuyNeural |
| Newsreader | Professional, neutral | Aoede | en-US-AriaNeural |
| Storyteller | Dramatic, immersive | Fenrir | en-US-DavisNeural |
| ASMR | Soft, intimate, relaxing | Aoede | en-US-JennyNeural |
| Motivational | Energetic, inspiring | Puck | en-US-TonyNeural |
| Documentary | Informative, thoughtful | Charon | en-GB-RyanNeural |
| Children | Warm, playful | Kore | en-US-AnaNeural |
| Podcast | Conversational, casual | Puck | en-US-GuyNeural |
| Meditation | Slow, peaceful | Aoede | en-US-JennyNeural |
| Custom | Describe your own style | Charon | en-US-GuyNeural |

### TTS backends

| Engine | How | Cost | Quality |
|---|---|---|---|
| `gemini` | Gemini TTS API (`GEMINI_API_KEY`) | Free tier | Good |
| `elevenlabs` | ElevenLabs REST (`ELEVENLABS_API_KEY`) | Paid | Excellent |
| `edge` | Microsoft Edge TTS (`pip install edge-tts`) | **Free** | Good |
| `auto` | Try gemini → elevenlabs → edge | — | Best available |

**CJK auto-voice:** if the text is predominantly Chinese/Japanese/Korean and an English voice is selected, the backend automatically switches to `zh-CN-XiaoxiaoNeural` so every character is spoken — not silently skipped.

**Long-text chunking:** texts over 2 000 chars are split at sentence boundaries, synthesized in chunks, and concatenated with ffmpeg. The full script is always read aloud regardless of length.

### Requirements

| Requirement | Install | Notes |
|---|---|---|
| `edge-tts` | `pip install edge-tts` | Free TTS (always-available fallback) |
| `ffmpeg` | `sudo apt install ffmpeg` or `pip install imageio-ffmpeg` | Required for multi-chunk concat |
| `GEMINI_API_KEY` | env var | Gemini TTS (optional) |
| `ELEVENLABS_API_KEY` | env var | ElevenLabs TTS (optional) |

Check status: `/tts status`

### Also in SSJ mode

`/tts` is available as option **12** in the SSJ Developer Mode menu, so you can chain it with brainstorm, worker, and video workflows in a single session.

---

