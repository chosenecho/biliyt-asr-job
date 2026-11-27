# biliyt-asr-job

Non-invasive daily orchestrator that connects [`biliyt`](https://github.com/chosenecho/biliyt-archiver) and `qwen3-asr-pipeline` via systemd — **without modifying either repository**.

## What It Does

On each scheduled run (default: `08:03:00` daily, offset from an hourly `:00` `biliyt-sync.timer`):

1. **Derives an ephemeral `biliyt` config** (`runtime/runtime_config.yaml`) from `/opt/biliyt/config.yaml` (read-only):
   - Anchors relative paths (`download_dir`, `state_dir`, `temp_dir`, `cookies_file`) to `/opt/biliyt`.
   - Enforces `media.max_height: 360` so `yt-dlp` downloads low-resolution video while keeping the **best audio stream** (`bv*[height<=360]+ba/...`).
   - Disables temporary video metadata/chapter/thumbnail embedding to skip extra remuxing.
   - Automatically tracks target additions/removals and cookie updates in `/opt/biliyt/config.yaml`.
2. **Downloads newly published videos** via `biliyt sync -c runtime/runtime_config.yaml --only videos`:
   - Shares `/opt/biliyt/state/archives/<target_id>.txt`, so deleted videos are never re-downloaded.
   - Automatically waits and retries if `biliyt` exits with code `3` (hourly sync holds `state/.sync.lock`).
3. **Extracts compact MP3 audio** via `qwen3-asr-pipeline/asr_pipeline.py extract --preset asr-mp3`:
   - Outputs 16kHz mono 48kbps MP3s under `/opt/qwen3-asr-pipeline/data/audio/<target_id>/`.
   - Runs on every newly downloaded video regardless of whether platform AI subtitles exist.
4. **Warms up Modal & runs ASR** via `qwen3-asr-pipeline/asr_pipeline.py transcribe --endpoint modal --save-json`:
   - Checks `pipeline.db` first and **skips Modal warm-up completely** when there is no pending audio.
   - Uses a `120s` health-check timeout (with retry) to absorb ~30s Modal GPU container cold starts.
   - Saves `.txt`, `.srt`, and raw `.json` transcripts under `/opt/qwen3-asr-pipeline/data/transcripts/<target_id>/`.
5. **Safely deletes temporary video files** while **keeping MP3 audio and all transcripts**:
   - Reads `pipeline.db` in read-only mode and only unlinks a video when **all** checks pass:
     - `asr_status == 'done'`
     - Video file is a regular file (not a symlink) strictly inside `biliyt`'s `download_dir`
     - Extracted `.mp3` exists and is non-empty
     - `.txt`, `.srt`, and `.json` (when `--save-json` is enabled) all exist and are non-empty
   - Failed ASR items retain their original video file for automatic retry on the next run.

## Directory Layout

```text
/opt/biliyt/                    # Untouched
/opt/qwen3-asr-pipeline/        # Untouched (MP3, transcripts, pipeline.db live here)

/opt/biliyt-asr-job/            # This repository
├── run_daily_asr.py
├── job.example.yaml
├── systemd/
│   ├── biliyt-asr.service
│   └── biliyt-asr.timer
└── runtime/                    # Git-ignored ephemeral config, lockfile, logs
```

## Requirements

- Python 3.10+ with `PyYAML`
- Existing working installations of `biliyt` and `qwen3-asr-pipeline` (with `.env` configured)

## Quick Start

```bash
git clone https://github.com/chosenecho/biliyt-asr-job.git /opt/biliyt-asr-job
cd /opt/biliyt-asr-job

# Optional: customize paths or concurrency
cp job.example.yaml job.yaml

# Dry-run check (no downloads, no ASR calls, no deletions)
python3 run_daily_asr.py --dry-run

# Run manually once
python3 run_daily_asr.py
```

## Install systemd Timer

```bash
sudo cp /opt/biliyt-asr-job/systemd/biliyt-asr.service /etc/systemd/system/
sudo cp /opt/biliyt-asr-job/systemd/biliyt-asr.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now biliyt-asr.timer

# Inspect schedule and logs
systemctl list-timers biliyt-asr.timer
tail -f /opt/biliyt-asr-job/runtime/logs/daily_asr.log
```

## Useful CLI Flags

```bash
# Run for a single target
python3 run_daily_asr.py --target bili_example

# Retry ASR & cleanup without re-running biliyt sync
python3 run_daily_asr.py --skip-sync

# Disable JSON output (only .txt + .srt)
python3 run_daily_asr.py --no-save-json
```

## Testing

```bash
python3 -m pytest -v
```
