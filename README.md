# Eye Cursor Control

Local Windows eye-to-cursor prototype using a laptop webcam.

This project is built for the practical setup we discussed:

- Train/calibrate from your own data.
- Use open-source face/eye landmarks, not paid eye-tracking APIs.
- Run real-time cursor control locally on the laptop.
- Use a GPU server only for optional offline experiments.

Laptop-webcam tracking will not be as accurate as an IR eye tracker. Expect it to work best for large UI targets after calibration, with keyboard or voice commands for clicking.

## Requirements

- Windows 11
- Python 3.11 or 3.12
- Built-in laptop webcam
- Good lighting
- Recommended: 16 GB RAM, though the baseline can run with 8 GB

Your machine already has Python 3.12 available through:

```powershell
py -3.12
```

## Setup

From this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

If MediaPipe install fails, confirm you are using Python 3.12, not 3.13 or 3.14.

## Check The Machine

```powershell
.\.venv\Scripts\eye-cursor.exe doctor
```

This checks Python version, imports, camera access, and screen size.

## Debug Webcam Tracking

```powershell
.\.venv\Scripts\eye-cursor.exe debug
```

Press `q` or `Esc` to quit. You should see the webcam image, face box, eye points, FPS, and feature quality.

If `doctor` says the camera cannot open:

- Check Windows Settings > Privacy & security > Camera.
- Enable camera access and desktop-app camera access.
- Close Teams/Zoom/Browser tabs that may already hold the camera.
- Check Device Manager if the device status is not normal.
- Try `--camera 1`, `--camera 2`, etc. if you later attach an external webcam.

## Calibrate

Use the laptop screen only for v1. Sit naturally, keep the laptop still, and look directly at each target.

```powershell
.\.venv\Scripts\eye-cursor.exe calibrate --profile default
```

Defaults:

- 7 by 5 target grid
- 8 shuffled rounds
- about 5 minutes
- feature-only data saved by default

Optional longer calibration:

```powershell
.\.venv\Scripts\eye-cursor.exe calibrate --profile default --rounds 12
```

Calibration files are saved under:

```text
profiles/default/
```

Each calibration run is timestamped, so old runs are not overwritten.

## Train

```powershell
.\.venv\Scripts\eye-cursor.exe train --profile default
```

The baseline model is a calibrated personal regressor. It is intentionally small so it can run locally without a GPU.

The command reports median and 90th-percentile pixel error. A usable laptop-webcam v1 is usually:

- median error below about 120 px
- 90th-percentile error below about 250 px

## Run Cursor Control

```powershell
.\.venv\Scripts\eye-cursor.exe run --profile default
```

Hotkeys:

- `Ctrl+Alt+G`: pause/resume gaze control
- `Ctrl+Alt+Q`: emergency stop
- `Ctrl+Alt+Enter`: left click
- `Ctrl+Alt+Backspace`: right click

The cursor freezes if face tracking is lost or if the prediction looks invalid.

Useful runtime options:

```powershell
.\.venv\Scripts\eye-cursor.exe run --profile default --show
.\.venv\Scripts\eye-cursor.exe run --profile default --camera 1
.\.venv\Scripts\eye-cursor.exe run --profile default --min-cutoff 0.8 --beta 0.08
```

## Server Workflow

Do not host live cursor inference on the GPU server. The cursor should run locally to avoid latency.

Use the server for offline training only:

1. Calibrate locally.
2. Upload the project and latest calibration to the SSH server.
3. Submit a SLURM job.
4. Fetch `model.onnx`, `model_metadata.json`, and `metrics.json` back into the local profile.
5. Run the cursor locally with the fetched model.

Install local ONNX inference support before using a server-trained model:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[onnx]"
```

Submit a remote training job:

```powershell
.\scripts\submit_remote_training.ps1 `
  -SshTarget "user@your-server" `
  -RemoteRoot "/home/user/eye-cursor-jobs" `
  -Profile default
```

The script stages:

- `src/`
- `pyproject.toml`
- `requirements-server.txt`
- the latest local calibration as `calibration.npz`
- `slurm/train_eye_cursor.sbatch`

After the SLURM job finishes, fetch the model:

```powershell
.\scripts\fetch_remote_training.ps1 `
  -SshTarget "user@your-server" `
  -RemoteJobDir "/home/user/eye-cursor-jobs/eye_cursor_default_YYYYMMDD_HHMMSS" `
  -Profile default
```

Then run locally:

```powershell
.\.venv\Scripts\eye-cursor.exe run --profile default
```

If your cluster needs modules or a specific partition/account, edit:

```text
slurm/train_eye_cursor.sbatch
```

The first working version can still use the included local `train` command. It does not require a GPU.

## Practical Tips

- Put the laptop on a stable desk.
- Recalibrate after moving the laptop or changing screen angle.
- Avoid backlighting.
- Use Windows display scaling consistently between calibration and runtime.
- If the cursor drifts, increase calibration rounds before changing model code.
- Keep keyboard or voice clicking for v1. Blink clicking is easy to trigger accidentally.
