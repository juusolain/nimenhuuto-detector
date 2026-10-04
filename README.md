# Nimenhuuto detector

A continuously running Python service that listens to a microphone, recognizes Finnish speech locally, and sends a Telegram message when it hears **“nimenhuuto”**. It is designed for a noisy queue: audio is analyzed in overlapping windows, likely transcription variations such as `nimen huuto` are accepted, and a cooldown prevents one shout from generating several alerts.

Only a fixed alert and its timestamp are sent over the network. Audio, surrounding speech, and transcriptions remain on the machine. Live transcripts are not written to normal `INFO` logs.

## Requirements

- Python 3.10–3.12
- A microphone placed as close as practical to where the shout occurs
- A Telegram bot token and destination chat ID
- Internet access on the first run to download the Whisper model, and later to call Telegram

On Linux, PortAudio may need to be installed first (for example, `sudo apt install libportaudio2`). Windows normally receives the required audio components through the Python wheel.

## Install

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

On Linux/macOS, activate with `source .venv/bin/activate` and copy the file with `cp .env.example .env`.

## Configure Telegram

1. Message [`@BotFather`](https://t.me/BotFather), create a bot, and copy its token into `TELEGRAM_BOT_TOKEN` in `.env`.
2. Send any message to the new bot.
3. Query updates from PowerShell without putting the token in browser history, then find `message.chat.id` and put that number in `TELEGRAM_CHAT_ID`:

   ```powershell
   $token = Read-Host "Telegram bot token"
   Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/getUpdates"
   Remove-Variable token
   ```

   For a group, add the bot to the group, send a message there, and use the negative group chat ID shown by `getUpdates`.

Do not commit `.env`; it is ignored by Git.

## Select and test the microphone

List audio devices:

```powershell
nimenhuuto --list-devices
```

Set `AUDIO_DEVICE` in `.env` to the desired input device index or part of its name. Leaving it blank uses the operating system default. You can also temporarily choose one with `nimenhuuto --device 2`.

First run without Telegram delivery:

```powershell
nimenhuuto --dry-run
```

Say “nimenhuuto” near the microphone. A successful detection is logged as `DRY RUN detection`. Stop with Ctrl+C. The first run downloads the selected Whisper model and can take several minutes.

You can also test a saved WAV, MP3, or other FFmpeg-supported recording:

```powershell
nimenhuuto --test-audio samples\nimenhuuto.wav --dry-run
```

Exit status is `0` when the target was detected, `1` when it was not, and `2` for a setup/runtime error.

## Run

```powershell
nimenhuuto
```

Run the command from the directory containing `.env`. For unattended use, set that directory as the working directory in Windows Task Scheduler, systemd, Docker, or another process supervisor that restarts it after a machine reboot or unexpected failure. Telegram delivery runs in the background and is retried during temporary network outages, so it does not pause microphone capture.

## Accuracy tuning

Real recordings from the installation location are essential. Collect examples of both the shout and normal queue noise, then tune these `.env` values:

| Setting | Default | Effect |
|---|---:|---|
| `WHISPER_MODEL` | `small` | Use `medium` for better recognition if the machine is fast enough; use `base` for lower CPU load. |
| `MATCH_THRESHOLD` | `0.86` | Text-similarity requirement. Raise toward `0.92` to reduce false alerts; lower toward `0.80` if real shouts are missed. |
| `MIN_LOG_PROBABILITY` | `-0.8` | Whisper confidence requirement. Raise toward `-0.5` to reject more uncertain recognition; lower toward `-1.0` if real shouts are rejected. |
| `TARGET_PHRASES` | `nimenhuuto,nimen huuto` | Add verified recurring mistranscriptions, comma-separated. |
| `MIN_AUDIO_DBFS` | `-50` | Windows quieter than this are skipped. Raise it if silence wastes CPU; lower it if distant calls are skipped. |
| `WINDOW_SECONDS` | `6` | Longer windows provide context but add latency and CPU use. |
| `STRIDE_SECONDS` | `2` | Detection runs this often. Lower values reduce latency but increase CPU use. |
| `ALERT_COOLDOWN_SECONDS` | `600` | Prevents duplicate alerts from overlapping windows and repeated calls. |

Saved-file tests log their transcript. During live monitoring, surrounding speech is intentionally not logged; detections log text-match and approximate ASR confidence instead. Start with `small`, keep the microphone away from speakers and machinery, and test at the actual distance. If the log says inference skipped windows, select a smaller model or use a faster computer. No speech recognizer can guarantee reliable results if the microphone cannot capture the shout above the ambient noise; microphone placement is usually more valuable than threshold changes.

### GPU (optional)

For a supported NVIDIA setup, install the CUDA libraries required by CTranslate2 and set:

```dotenv
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

CPU with `int8` is the portable default.

## Tests

```powershell
python -m unittest discover -s tests
```
