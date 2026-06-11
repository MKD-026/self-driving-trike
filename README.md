# Bike TTS Voice Module

Offline, embedded-friendly text-to-speech for the self-driving trike. Exposes a
simple `speak(text)` function that synthesizes a string with **[Piper](https://github.com/OHF-Voice/piper1-gpl)**
and plays it through the system speaker. This is the bottom audio layer for the
bike's situational-awareness voice (the layer that tells a blind rider what the
bike is doing).

- **Fully offline** — no network at runtime.
- **Fast** — model loads once (~1 s), then synthesizes at ~25× real-time
  (measured RTF ≈ 0.04 on a desktop CPU). Runs on a Raspberry Pi.
- **One speaker, one cue at a time** — playback is serialized so two
  announcements never talk over each other.

---

## 1. Requirements

**System**
- Python **3.10+**
- A working audio output and one of these players (first found is used):
  `paplay` · `aplay` · `pw-play` · `ffplay`
  - Debian/Ubuntu/Raspberry Pi OS: `sudo apt install -y pulseaudio-utils alsa-utils`
  - (`paplay`/`aplay` come from those two packages.)

**Python**
- See [`requirements.txt`](requirements.txt) (`piper-tts==1.4.2`, which pulls in
  `onnxruntime` + `numpy`).

## 2. Setup (replicate at your end)

```bash
# 1. clone the repo and check out this branch
git checkout tts-voice          # (or whatever the branch is named)
cd <path-to-this-folder>

# 2. create an isolated Python env (conda or venv — either is fine)
python -m venv .venv && source .venv/bin/activate
#   …or:  conda create -n trike-tts python=3.11 -y && conda activate trike-tts

# 3. install Python deps
pip install -r requirements.txt

# 4. download the Piper voice model (~60 MB, NOT in git)
chmod +x download_voice.sh
./download_voice.sh             # fetches en_US-lessac-medium into ./voices/
```

That's it. The voice model lands in `./voices/` and `tts.py` finds it automatically.

## 3. Usage

**From Python:**
```python
import tts

tts.speak("turning left ahead")            # blocking: speaks, returns when done
tts.speak("braking", blocking=False)       # non-blocking: plays in background
tts.say_async("on Elm Street")             # alias for non-blocking
tts.synth_to_wav("stopping", "stop.wav")   # render a WAV (pre-cache fixed phrases)
tts.warmup()                               # pre-load the model (optional)
```

**From the shell:**
```bash
python tts.py "turn right in fifty meters"
```

## 4. API

| Function | Description |
|---|---|
| `speak(text, blocking=True)` | Synthesize and play `text`. Blocking returns when audio finishes; non-blocking returns a `Thread`. |
| `say_async(text)` | Non-blocking alias for `speak(text, blocking=False)`. |
| `synth_to_wav(text, path)` | Render `text` to a WAV file (no playback). Use to pre-cache the fixed phrase grammar at build time. |
| `warmup()` | Load the Piper model now so the first `speak()` has no load latency. |

## 5. Files

```
tts.py              the module (the speak() API)
requirements.txt    Python dependencies
download_voice.sh   fetches the Piper voice model into ./voices/
voices/             voice model(s) — created by the script, git-ignored
.gitignore          keeps the 60 MB model and scratch WAVs out of git
README.md           this file
```

## 6. Notes / next steps

- **Change voice:** `./download_voice.sh en_US-amy-medium`, then set
  `VOICE_NAME` near the top of `tts.py`. Browse voices at
  <https://rhasspy.github.io/piper-samples/>.
- **Pre-caching:** because the fixed announcement grammar never changes, render
  each phrase once with `synth_to_wav()` and play the cached WAVs at runtime —
  that's the zero-latency path for urgent (P0) cues on the bike.
- This module is just the audio output. The arbiter on top of it (priority
  queue, message coalescing, periodic heartbeat, speech budget) is what decides
  *what* and *when* to speak.
