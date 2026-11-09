# Setting up on a fresh Windows 11 PC with an NVIDIA GPU

The GPU only speeds up the Whisper transcription step. Speaker detection always runs on the CPU.

## 1. Install these three things (in this order)
| # | What | Where | Notes |
|---|---|---|---|
| 1 | **NVIDIA graphics driver** (recent) | nvidia.com/drivers | Check it worked: open PowerShell and run `nvidia-smi`. It should print your GPU. |
| 2 | **Python 3.14, 64-bit** | python.org/downloads/windows | Tick **"Add python.exe to PATH"** in the installer. It must be 3.14: the bundled packages are built for it. |
| 3 | **Microsoft Visual C++ Redistributable 2015-2022, x64** | aka.ms/vc14/vc_redist.x64.exe | Needed by the speech libraries. If the installer offers only **Repair** and **Uninstall**, it is already installed: click Close and move on. Do not uninstall it. |

You do **not** need: the CUDA Toolkit, a manual cuDNN install, ffmpeg, Git, or internet access (when you use the full zip with `wheels/` and `models/`).

## 2. Copy and unpack
Unzip to a short path such as `C:\audio-transcriber`.

## 3. First run
Open PowerShell in that folder:
```powershell
cd C:\audio-transcriber
powershell -ExecutionPolicy Bypass -File .\run_mp3-to-txt.ps1 -h        # shows help
powershell -ExecutionPolicy Bypass -File .\run_mp3-to-txt.ps1 C:\path\call.mp3 -Speakers "Alice,Bob"
```
The first run builds a Python environment from the bundled `wheels/` (a few minutes; the NVIDIA libraries are over 1 GB). Later runs start immediately.

## 4. Check the GPU is used
The summary at the end shows `device: CUDA`. If it says `device: CPU`:
- run `nvidia-smi`; if it fails, reinstall the driver and reboot;
- run with `-Device cuda` to force the GPU and see the real error message;
- make sure you did not delete `wheels/` before the first run (the NVIDIA libraries come from there).

## Troubleshooting
- **"No matching distribution found" during setup:** wrong Python version. Install 3.14, or delete the `wheels` folder and rerun with internet access.
- **Scripts blocked:** use the `-ExecutionPolicy Bypass` form shown above.
- **Moving the folder later:** delete `venv/` (it holds absolute paths); the next run rebuilds it.

## Status
The GPU path follows the faster-whisper documentation for Windows, but it was written on a PC without an NVIDIA GPU and has not been run on one. The automatic CPU fallback was tested.
