# Notes: models, accuracy and speed

## What the models are
All Whisper models are open source from **OpenAI** (2022, MIT license). The versions used here are converted by **Systran** to the CTranslate2 format that `faster-whisper` runs: same model, faster and lighter.

| Model | Size | Speed on CPU | Accuracy |
|---|---|---|---|
| tiny / base | 75 / 145 MB | Very fast | Poor on accents and names |
| small | ~460 MB | Moderate (13 min of audio in a few minutes) | Good for clear English |
| medium | ~1.5 GB | Measured ~24 min for 13 min of audio | Mixed vs small on one test (below) |
| large-v3 | ~3 GB | Slowest; best with a GPU | Best in general |

Larger models reduce misheard words; they do not add speaker labels.

## small vs medium (one 13-minute two-person call)
About 90% word-for-word identical. Medium gave cleaner punctuation and fixed several words, but introduced errors of its own (dropped a phrase, wrong acronym spelling, a spurious "Exactly") and took ~24 minutes vs a few for small. For clear audio, small was enough. Proper nouns were wrong in both; check them by ear.

## Speaker labels
Done with sherpa-onnx: a pyannote segmentation model plus a voice-embedding model, then clustering.
- Two-person, 13-minute call: 2 speakers found; about 3 of ~260 lines mislabelled (short interjections).
- Names are assigned by order of first speech, not by voice. If labels come out reversed, swap the names.
- Without a speaker count the tool over-split one file into 54 "speakers". Always give `-Speakers` or `-NumSpeakers`.
- Lines are split word by word at each change of speaker, because Whisper segments can be longer than a turn. This needs word timestamps and slightly changes how some segments are worded.

### Embedding models (3 people, 3-minute clip, 3 speakers requested)
| Embedding | Voices found |
|---|---|
| titanet-small | 2 (two people merged) |
| titanet-large | 2 |
| wespeaker | 2 |
| eres2net (3D-Speaker) | 3 |

One clip only, so this is not proof it is better in general. `eres2net` is the default. It was ~2.7x slower than titanet-small on speaker detection (103 s vs 38 s for 180 s of audio, one run each, CPU). Rough extrapolation: ~34 min vs ~13 min per hour of audio (not measured on long files).

## Cloud alternatives
This tool runs entirely locally and is slower than real time on a CPU. Hosted speech-to-text services are faster and usually do speaker labels, but they need an API key, cost money, and upload the audio. A GPU-enabled local install is the other way to speed it up.

## mp3-clip.ps1
Copies audio between two timestamps without re-encoding (seconds to run, no quality loss, cut accurate to a few hundredths of a second). Never modifies the original or overwrites an existing output. Times: `90`, `1:30`, `1:02:30`. Transcript timestamps of a clip are relative to the clip start; add `-Begin` to locate the position in the original.
