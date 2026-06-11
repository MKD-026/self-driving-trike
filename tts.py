"""
Piper-based text-to-speech for the bike's situational-awareness voice.

Public API:
    speak("turn left ahead")          # blocking: synthesize + play, returns when done
    speak("braking", blocking=False)  # fire-and-forget: play in a background thread
    say_async("on Elm Street")        # alias for non-blocking
    synth_to_wav("stopping", path)    # just render a WAV (e.g. to pre-cache the fixed grammar)
    warmup()                          # load the model once up front (first speak is then instant)

Design notes (see DESIGN.md):
  - One Piper voice, loaded once and reused (cheap on embedded after warmup).
  - A single playback lock so two cues never overlap on the speaker. A new
    blocking speak() waits its turn; that's the simplest "one speech channel".
  - Runtime synthesis is only needed for dynamic strings. For the fixed grammar,
    pre-render once with synth_to_wav() and play the cached WAV instead.
"""
from __future__ import annotations

import threading
import wave
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from piper import PiperVoice

# ── configuration ────────────────────────────────────────────────────────────
VOICE_DIR = Path(__file__).resolve().parent / "voices"
VOICE_NAME = "en_US-lessac-medium"
VOICE_ONNX = VOICE_DIR / f"{VOICE_NAME}.onnx"

# pick whatever audio player exists on this box (Pi / desktop)
_PLAYERS = ["paplay", "aplay", "pw-play", "ffplay"]

_voice: PiperVoice | None = None
_voice_lock = threading.Lock()     # guards lazy model load
_play_lock = threading.Lock()      # guards the speaker — one cue at a time


def _get_voice() -> PiperVoice:
    """Load the Piper voice once and cache it."""
    global _voice
    if _voice is None:
        with _voice_lock:
            if _voice is None:
                if not VOICE_ONNX.exists():
                    raise FileNotFoundError(
                        f"Piper voice not found: {VOICE_ONNX}\n"
                        f"Download it with:\n"
                        f"  python -m piper.download_voices {VOICE_NAME}\n"
                        f"(run inside {VOICE_DIR})"
                    )
                _voice = PiperVoice.load(str(VOICE_ONNX))
    return _voice


def warmup() -> None:
    """Load the model now so the first real speak() has no load latency."""
    _get_voice()


def _synth_wav_bytes(text: str) -> bytes:
    """Render text to in-memory WAV bytes."""
    voice = _get_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        voice.synthesize_wav(text, wav)
    return buf.getvalue()


def synth_to_wav(text: str, path: str | Path) -> Path:
    """Render text to a WAV file on disk (use to pre-cache fixed phrases)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    voice = _get_voice()
    with wave.open(str(path), "wb") as wav:
        voice.synthesize_wav(text, wav)
    return path


def _player_cmd(path: str) -> list[str]:
    """Build the playback command for whichever player exists, given a WAV path.

    We play a real file (not stdin) because WAV needs a seekable source — piping
    bytes through stdin makes players like paplay fail to parse the header.
    """
    for p in _PLAYERS:
        if shutil.which(p):
            if p == "ffplay":
                return [p, "-nodisp", "-autoexit", "-loglevel", "quiet", path]
            if p == "aplay":
                return [p, "-q", path]
            return [p, path]
    raise RuntimeError(f"No audio player found (looked for: {_PLAYERS})")


def _play_wav_bytes(wav_bytes: bytes) -> None:
    """Write WAV to a temp file and play it. Serialized by _play_lock."""
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="tts_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes)
        cmd = _player_cmd(tmp)
        with _play_lock:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise RuntimeError(f"audio player {cmd[0]} exited {proc.returncode}")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def speak(text: str, blocking: bool = True) -> threading.Thread | None:
    """
    Speak `text` aloud through the bike's speaker.

    blocking=True  (default): synthesize + play, return when the audio finishes.
    blocking=False: render+play on a background thread, return the Thread handle.

    Multiple cues never overlap: playback is serialized by an internal lock.
    """
    text = (text or "").strip()
    if not text:
        return None

    def _run():
        _play_wav_bytes(_synth_wav_bytes(text))

    if blocking:
        _run()
        return None

    t = threading.Thread(target=_run, name="tts-speak", daemon=True)
    t.start()
    return t


def say_async(text: str) -> threading.Thread | None:
    """Non-blocking alias for speak(text, blocking=False)."""
    return speak(text, blocking=False)


if __name__ == "__main__":
    import sys
    msg = " ".join(sys.argv[1:]) or "Bike voice online. Turning left ahead."
    print(f"speaking: {msg!r}")
    warmup()
    speak(msg)
    print("done.")
