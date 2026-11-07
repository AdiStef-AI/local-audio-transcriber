<#
.SYNOPSIS
    Cut a piece out of an audio file between two timestamps (no re-encoding, so it is fast and lossless).

.DESCRIPTION
    Uses the ffmpeg binary already installed in tools\venv (via imageio-ffmpeg),
    or ffmpeg on PATH if the venv is missing. Run run_mp3-to-txt.ps1 once first
    so the venv exists.

    Timestamps can be written as seconds (90), mm:ss (1:30) or hh:mm:ss (1:02:30).

.PARAMETER InputFile
    Source audio file (mp3, m4a, wav, ...).

.PARAMETER Begin
    Start of the clip. Default: 0 (the start of the file).

.PARAMETER End
    End of the clip. Default: the end of the file.

.PARAMETER OutputFile
    Where to save the clip. Default: <name>_clip_<begin>-<end>.<ext> next to the input.

.PARAMETER Help
    Show this help. Also: -h

.EXAMPLE
    .\mp3-clip.ps1 call.mp3 -Begin 5:00 -End 20:30

.EXAMPLE
    .\mp3-clip.ps1 call.mp3 -Begin 1:02:00 -End 1:30:00 short.mp3

.EXAMPLE
    .\mp3-clip.ps1 call.mp3 -End 10:00      # first 10 minutes
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$InputFile,
    [Parameter(Position = 1)][string]$OutputFile,
    [Alias('Start')][string]$Begin = '0',
    [string]$End = '',
    [Alias('h')][switch]$Help
)

$ErrorActionPreference = 'Stop'

function Show-Help {
    Write-Host @"

mp3-clip.ps1 - cut a piece out of an audio file

USAGE
  .\mp3-clip.ps1 <InputFile> [OutputFile] [-Begin <time>] [-End <time>]
  .\mp3-clip.ps1 -h

ARGUMENTS
  InputFile    Audio file to cut (mp3, m4a, wav, ...). Required.
  OutputFile   Where to save the clip. Default: <name>_clip_<begin>-<end>.<ext>
               beside the input.
  -Begin       Start time. Default: 0 (start of file). Alias: -Start
  -End         End time. Default: end of file.
  -h, -Help    Show this help.

TIME FORMATS
  90        90 seconds
  1:30      1 min 30 s
  1:02:30   1 h 2 min 30 s

EXAMPLES
  .\mp3-clip.ps1 call.mp3 -Begin 5:00 -End 20:30
  .\mp3-clip.ps1 call.mp3 -Begin 1:02:00 -End 1:30:00 short.mp3
  .\mp3-clip.ps1 call.mp3 -End 10:00        (first 10 minutes)
  .\mp3-clip.ps1 call.mp3 -Begin 45:00      (from 45:00 to the end)

HOW IT WORKS
  ffmpeg copies the audio between the two times without re-encoding, so it
  takes seconds even for a long file and there is no quality loss. The cut
  lands on an mp3 frame boundary, so it can be off by a few hundredths of a
  second. The original file is never changed, and an existing output file is
  never overwritten (the script stops instead).

TIP
  Clip first, then transcribe the clip:
    .\mp3-clip.ps1 long.mp3 -Begin 10:00 -End 25:00 part.mp3
    .\run_mp3-to-txt.ps1 part.mp3 -Speakers "Alice,Bob"
  Timestamps in the transcript are then relative to the clip, so add the Begin
  time to get the position in the original file.

NOTES
  - If PowerShell says "running scripts is disabled", run it like this:
      powershell -ExecutionPolicy Bypass -File .\mp3-clip.ps1 <InputFile> ...

"@
}

function ConvertTo-Seconds([string]$t) {
    $t = $t.Trim()
    if ($t -match '^\d+(\.\d+)?$') { return [double]$t }
    if ($t -match '^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$') {
        $h = if ($Matches[1]) { [double]$Matches[1] } else { 0 }
        return $h * 3600 + [double]$Matches[2] * 60 + [double]$Matches[3]
    }
    throw "Cannot read time '$t'. Use seconds (90), mm:ss (1:30) or hh:mm:ss (1:02:30)."
}

function Format-Time([double]$s) {
    $ts = [TimeSpan]::FromSeconds($s)
    '{0}:{1:00}:{2:00}' -f [int][math]::Floor($ts.TotalHours), $ts.Minutes, $ts.Seconds
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

try {
    $b = ConvertTo-Seconds $Begin
    $e = if ($End) { ConvertTo-Seconds $End } else { $null }
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
if ($null -ne $e -and $e -le $b) {
    Write-Host "Error: -End ($End) must be later than -Begin ($Begin)." -ForegroundColor Red
    exit 1
}

# Find ffmpeg: venv first, then PATH.
$ffmpeg = $null
$venvPy = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
if (Test-Path $venvPy) {
    $ffmpeg = & $venvPy -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())" 2>$null
    if ($LASTEXITCODE -ne 0) { $ffmpeg = $null }
}
if (-not $ffmpeg -and (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { $ffmpeg = 'ffmpeg' }
if (-not $ffmpeg) {
    Write-Host 'Error: ffmpeg not found. Run run_mp3-to-txt.ps1 once to build tools\venv, or install ffmpeg.' -ForegroundColor Red
    exit 1
}

if (-not $OutputFile) {
    $tag = '{0}-{1}' -f ((Format-Time $b) -replace ':', '.'), $(if ($null -ne $e) { (Format-Time $e) -replace ':', '.' } else { 'end' })
    $OutputFile = Join-Path (Split-Path $InputFile) ([IO.Path]::GetFileNameWithoutExtension($InputFile) + "_clip_$tag" + [IO.Path]::GetExtension($InputFile))
}
if (Test-Path -LiteralPath $OutputFile) {
    Write-Host "Error: output file already exists, not overwriting: $OutputFile" -ForegroundColor Red
    exit 1
}

$inv = [Globalization.CultureInfo]::InvariantCulture
$ffArgs = @('-hide_banner', '-loglevel', 'error', '-ss', $b.ToString($inv), '-i', $InputFile)
if ($null -ne $e) { $ffArgs += @('-t', ($e - $b).ToString($inv)) }
$ffArgs += @('-c', 'copy', $OutputFile)

Write-Host "Clipping $(Format-Time $b) -> $(if ($null -ne $e) { Format-Time $e } else { 'end' })" -ForegroundColor Cyan
& $ffmpeg @ffArgs
if ($LASTEXITCODE -ne 0) { Write-Host 'ffmpeg failed.' -ForegroundColor Red; exit $LASTEXITCODE }
if (-not (Test-Path -LiteralPath $OutputFile) -or (Get-Item -LiteralPath $OutputFile).Length -eq 0) {
    Write-Host 'Error: nothing was written. Is -Begin past the end of the file?' -ForegroundColor Red
    exit 1
}
Write-Host ("Done: {0} ({1:N1} MB)" -f $OutputFile, ((Get-Item -LiteralPath $OutputFile).Length / 1MB)) -ForegroundColor Green
exit 0
