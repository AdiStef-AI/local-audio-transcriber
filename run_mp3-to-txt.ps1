<#
.SYNOPSIS
    Transcribe an audio/video file to a timestamped text file, locally (no API key),
    optionally labelling who is speaking.

.DESCRIPTION
    Wraps tools\transcribe.py. On first run it creates a Python virtual
    environment (tools\venv) and installs faster-whisper, imageio-ffmpeg, numpy
    and sherpa-onnx, from tools\wheels when that folder exists (offline).

    How it works:
      1. ffmpeg (bundled with imageio-ffmpeg) decodes your file to 16 kHz mono audio.
      2. Optional: sherpa-onnx finds who speaks when (speaker diarization).
      3. faster-whisper (an open-source Whisper model) turns the speech into text.
      4. Each segment is written as "[mm:ss] Speaker 1: text" to the output file.

    Everything runs on your machine. Nothing is uploaded.

.PARAMETER InputFile
    Audio or video file (mp3, m4a, wav, mp4, ...).

.PARAMETER OutputFile
    Where to write the transcript. Default: <InputFile name>.transcript.txt next to the input.

.PARAMETER Model
    Whisper model: tiny, base, small (default), medium, large-v3.

.PARAMETER Lang
    Spoken language code. Default: en.

.PARAMETER Diarize
    Label each line with the speaker (Speaker 1, Speaker 2, ...).

.PARAMETER NumSpeakers
    How many people speak. Giving it turns labelling on and is much more reliable
    than letting the tool guess. 0 (default) = guess.

.PARAMETER Speakers
    Names for the speakers, in order of first speech, e.g. "Alice,Bob".
    Turns labelling on.

.PARAMETER Help
    Show this help. Also: -h

.EXAMPLE
    .\run_mp3-to-txt.ps1 ..\inputs\call.mp3

.EXAMPLE
    .\run_mp3-to-txt.ps1 call.mp3 -NumSpeakers 2 -Speakers "Alice,Bob"

.EXAMPLE
    .\run_mp3-to-txt.ps1 -h
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$InputFile,
    [Parameter(Position = 1)][string]$OutputFile,
    [ValidateSet('tiny', 'base', 'small', 'medium', 'large-v3')][string]$Model = 'small',
    [string]$Lang = 'en',
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [switch]$Diarize,
    [int]$NumSpeakers = 0,
    [string]$Speakers = '',
    [ValidateSet('titanet-small', 'titanet-large', 'wespeaker', 'eres2net')][string]$Embedding = 'eres2net',
    [Alias('h')][switch]$Help
)

$ErrorActionPreference = 'Stop'

function Show-Help {
    Write-Host @"

run_mp3-to-txt.ps1 - local speech-to-text (no API key, nothing uploaded)

USAGE
  .\run_mp3-to-txt.ps1 <InputFile> [OutputFile] [-Model <name>] [-Lang <code>]
                       [-Diarize] [-NumSpeakers <n>] [-Speakers "Alice,Bob"]
  .\run_mp3-to-txt.ps1 -h

ARGUMENTS
  InputFile      Audio/video file (mp3, m4a, wav, mp4, ...). Required.
  OutputFile     Transcript path. Default: <InputFile>.transcript.txt beside the input.
  -Model         tiny | base | small (default) | medium | large-v3
                 Bigger = more accurate, slower.
  -Lang          Spoken language code (default: en).
  -Diarize       Label who is speaking: "Speaker 1:", "Speaker 2:", ...
  -NumSpeakers   Number of people speaking (e.g. 2). Turns labelling on.
                 Strongly recommended: much more reliable than auto-guessing.
  -Speakers      Names in order of first speech, e.g. "Alice,Bob".
                 Turns labelling on AND sets the speaker count to the number of
                 names (unless -NumSpeakers is also given). The first voice heard gets the first name,
                 so check the first lines and swap the names if they are reversed.
  -Embedding     Voice model used to tell speakers apart:
                 eres2net (default) | titanet-small | titanet-large | wespeaker
                 eres2net separated 3 voices on a test where the others merged
                 two of them. If it merges or splits voices wrongly on your
                 audio, try -Embedding titanet-small.
  -Device        auto (default) | cpu | cuda
                 auto uses an NVIDIA GPU for Whisper when one works, otherwise
                 the CPU. cuda forces the GPU and stops with an error if it
                 cannot be used. Speaker detection always runs on the CPU.
  -h, -Help      Show this help.

EXAMPLES
  .\run_mp3-to-txt.ps1 ..\inputs\call.mp3
  .\run_mp3-to-txt.ps1 call.mp3 out.txt -Model medium
  .\run_mp3-to-txt.ps1 call.mp3 -NumSpeakers 2 -Speakers "Alice,Bob"

OUTPUT
  Without labels:  [03:27] text spoken at 3 min 27 s
  With labels:     [03:27] Alice: text spoken at 3 min 27 s

PROGRESS
  Two progress bars show percent done, elapsed time and ETA: one for speaker
  detection (if enabled) and one for transcription. The ETA is rough at first.
  At the end a SUMMARY shows: file size, audio length, speech time, words,
  speakers found (share of speech, lines and words each), time per phase,
  total time, and speed as a multiple of real time.

HOW IT WORKS
  1. First run only: creates tools\venv and installs faster-whisper,
     imageio-ffmpeg, numpy and sherpa-onnx from tools\wheels (offline) or
     from the internet if tools\wheels is missing.
  2. ffmpeg decodes your file to 16 kHz mono audio.
  3. Speaker labels (optional): sherpa-onnx runs a pyannote segmentation model
     and a NeMo TitaNet voice-embedding model, then clusters voices. Models
     are in tools\models\diarization. It tells voices apart; it does not
     know who anyone is, hence generic names unless you pass -Speakers.
  4. Whisper transcribes on your CPU. Models are in tools\models
     (small ~460 MB, medium ~1.5 GB; others download once on first use).
  5. Each text segment gets the speaker whose voice overlaps it most.

LIMITS OF SPEAKER LABELS
  - Works best with 2-4 speakers and clear audio.
  - Short interjections ("yes, yes") and overlapping speech can be mislabelled.
  - Lines are split at each change of speaker, word by word. Boundaries can be
    off by a word or two, and a short interjection may be absorbed.
  - Without -NumSpeakers the tool may find too many or too few voices.

GPU (NVIDIA)
  Needs a recent NVIDIA driver (check with: nvidia-smi). No CUDA Toolkit is
  needed: the CUDA libraries come from pip (nvidia-cublas-cu12,
  nvidia-cudnn-cu12) and are installed automatically when nvidia-smi exists.
  The summary at the end shows "device: CUDA" or "device: CPU".

PORTING TO ANOTHER MACHINE
  Copy the whole tools folder. Everything lives there: scripts, models
  (so no re-download), wheels and venv. The venv holds absolute paths and is
  NOT portable: delete tools\venv on the new machine and the first run
  rebuilds it from tools\wheels, with no internet needed.
  The new machine needs the SAME Python version (3.14) and Windows 64-bit,
  because the wheels are built for that combination. Python itself is not
  bundled; install it first.

NOTES
  - Needs Python 3 on PATH ("python --version" to check).
  - Check the output for misheard names and technical terms.
  - If PowerShell says "running scripts is disabled", run it like this:
      powershell -ExecutionPolicy Bypass -File .\run_mp3-to-txt.ps1 <InputFile> ...
    (applies to this one run only; no system setting is changed)
  - The engine is tools\transcribe.py; edit it to change behaviour.

"@
}

if ($Help -or -not $InputFile) {
    Show-Help
    if (-not $Help) { Write-Host "Error: no InputFile given.`n" -ForegroundColor Yellow; exit 1 }
    exit 0
}

if (-not (Test-Path -LiteralPath $InputFile)) {
    Write-Host "Error: input file not found: $InputFile" -ForegroundColor Red
    exit 1
}
$InputFile = (Resolve-Path -LiteralPath $InputFile).Path
if (-not $OutputFile) {
    $OutputFile = Join-Path (Split-Path $InputFile) ([IO.Path]::GetFileNameWithoutExtension($InputFile) + '.transcript.txt')
}

$here = $PSScriptRoot
$venv = Join-Path $here 'venv'
$py = Join-Path $venv 'Scripts\python.exe'
$wheels = Join-Path $here 'wheels'
$packages = @('faster-whisper', 'imageio-ffmpeg', 'numpy', 'sherpa-onnx')
# NVIDIA GPU present (driver installed)? Then also install the CUDA libraries Whisper needs.
$hasGpu = ($Device -ne 'cpu') -and [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasGpu) { $packages += @('nvidia-cublas-cu12', 'nvidia-cudnn-cu12') }

function Install-Packages {
    if (Test-Path $wheels) {
        Write-Host 'Installing from local tools\wheels (offline).' -ForegroundColor Cyan
        & $py -m pip install --quiet --no-index --find-links $wheels @packages
    } else {
        & $py -m pip install --quiet @packages
    }
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
}

if (-not (Test-Path $py)) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host 'Error: Python not found on PATH. Install Python 3 first.' -ForegroundColor Red
        exit 1
    }
    Write-Host 'First run: creating virtual environment and installing packages...' -ForegroundColor Cyan
    & python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
    Install-Packages
} else {
    # Existing venv from an older version may lack sherpa-onnx.
    & $py -c "import sherpa_onnx" 2>$null
    $missing = ($LASTEXITCODE -ne 0)
    if ($hasGpu) { & $py -c "import nvidia.cudnn, nvidia.cublas" 2>$null; if ($LASTEXITCODE -ne 0) { $missing = $true } }
    if ($missing) { Write-Host 'Adding missing packages...' -ForegroundColor Cyan; Install-Packages }
}

$argsList = @($InputFile, $OutputFile, '--model', $Model, '--lang', $Lang)
if ($Diarize) { $argsList += '--diarize' }
if ($NumSpeakers -gt 0) { $argsList += @('--num-speakers', $NumSpeakers) }
if ($Speakers) { $argsList += @('--speakers', $Speakers) }
$argsList += @('--embedding', $Embedding, '--device', $Device)

Write-Host "Transcribing: $InputFile (model: $Model)" -ForegroundColor Cyan
& $py (Join-Path $here 'transcribe.py') @argsList
$code = $LASTEXITCODE
if ($code -ne 0) { Write-Host 'Transcription failed.' -ForegroundColor Red; exit $code }
Write-Host "Done: $OutputFile" -ForegroundColor Green
exit 0
