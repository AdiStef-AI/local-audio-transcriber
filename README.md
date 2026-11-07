# Local audio transcriber (Windows / PowerShell)

Transcribe an audio file to text on your own computer, with optional "who is speaking" labels, and cut long recordings into pieces. **No API key, no account, nothing is uploaded.**

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (OpenAI's open-source Whisper model) and [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for speaker detection.

## Files
| File | Purpose |
|---|---|
| `run_mp3-to-txt.ps1` | Main command. Creates a Python venv on first run, then transcribes. |
| `transcribe.py` | The engine called by the wrapper. |
| `mp3-clip.ps1` | Cuts a time range out of an audio file without re-encoding. |
| `docs/NOTES.md` | Model comparison, accuracy findings, speed numbers. |

## Requirements
- Windows 64-bit, PowerShell, Python 3 on PATH (developed on 3.14).
- Internet on first run (packages, and Whisper models downloaded from Hugging Face), unless you pre-populate `wheels/` and `models/`.

## Quick start
```powershell
.\run_mp3-to-txt.ps1 -h                                   # full help
.\run_mp3-to-txt.ps1 call.mp3                             # plain transcript
.\run_mp3-to-txt.ps1 call.mp3 -Speakers "Alice,Bob"       # label two speakers
.\run_mp3-to-txt.ps1 call.mp3 -Model medium               # more accurate, slower
.\mp3-clip.ps1 long.mp3 -Begin 10:00 -End 25:30           # cut a piece first
```
If PowerShell blocks scripts: `powershell -ExecutionPolicy Bypass -File .\run_mp3-to-txt.ps1 ...`

## Speaker labels
- `-Speakers "A,B,C"` names the voices in order of first speech and sets the speaker count.
- `-NumSpeakers N` sets the count explicitly. Guessing the count (neither option) is unreliable.
- `-Embedding` picks the voice model: `eres2net` (default), `titanet-small`, `titanet-large`, `wespeaker`.
- The speaker models are not included. Download them into `models/diarization/` from the [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) (`speaker-segmentation-models` and `speaker-recongition-models`):
  `sherpa-onnx-pyannote-segmentation-3-0` (extract the tarball) plus the embedding `.onnx` files named in `transcribe.py`.

## Offline install
Run `pip download faster-whisper imageio-ffmpeg numpy sherpa-onnx -d wheels` once; the wrapper then installs from `wheels/` with no internet. Wheels are tied to the Python version and OS. Whisper models go in `models/` (the first run downloads them there).

## Privacy
Everything runs locally. Do not commit recordings or transcripts (the `.gitignore` blocks common audio formats and `*.transcript.txt`).

## Limits
Speaker labels are good for 2-4 speakers with clear audio. Short interjections and overlapping speech are sometimes mislabelled, and names or jargon can be misheard. Check the output. See `docs/NOTES.md`.
