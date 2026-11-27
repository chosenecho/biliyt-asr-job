#!/usr/bin/env python3
"""Non-invasive daily orchestrator connecting biliyt and qwen3-asr-pipeline.

Workflow:
1. Read biliyt's config.yaml (read-only) and generate an isolated runtime YAML
   anchoring relative paths to biliyt's directory and forcing low-res video
   with best audio (`media.max_height <= 360`).
2. Run `biliyt sync -c <runtime_config> --only videos` (waiting on exit code 3
   if the hourly sync timer currently holds `state/.sync.lock`).
3. Run `asr_pipeline.py extract` on target directories containing videos
   (produces compact 16kHz mono MP3 in qwen3-asr-pipeline/data/audio/).
4. If any target group has pending/failed audio awaiting ASR, warm up the
   configured ASR endpoint (handles ~30s Modal cold start) and run
   `asr_pipeline.py transcribe --save-json`.
5. Delete original downloaded video files ONLY after verifying in pipeline.db
   that `asr_status = 'done'` AND the kept MP3 + `.txt` + `.srt` (+ `.json`)
   all exist and are non-empty on disk.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".flv", ".avi", ".m4v", ".ts"}
BILIYT_LOCK_EXIT_CODE = 3
BILIYT_PARTIAL_FAILURE_EXIT_CODE = 2

DEFAULTS: dict[str, Any] = {
    "biliyt_config": "/opt/biliyt/config.yaml",
    "biliyt_bin": "/usr/local/bin/biliyt",
    "asr_dir": "/opt/qwen3-asr-pipeline",
    "python_bin": sys.executable or "python3",
    "runtime_dir": str(REPO_ROOT / "runtime"),
    "max_height": 360,
    "audio_preset": "asr-mp3",
    "extract_workers": 4,
    "endpoint": "modal",
    "asr_concurrency": 4,
    "save_json": True,
    "warmup_timeout": 120,
    "warmup_retries": 2,
    "asr_timeout": 300,
    "task_timeout": 7200,
    "lock_wait_seconds": 300,
    "lock_retry_interval": 15,
}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@dataclass(frozen=True)
class TargetSpec:
    id: str
    platform: str
    download_dir: Path
    enabled: bool


@dataclass
class CleanupSummary:
    deleted: int = 0
    reclaimed_bytes: int = 0
    already_gone: int = 0
    kept_unsafe_or_incomplete: int = 0


class JobLockError(RuntimeError):
    """Raised when another run_daily_asr.py process already holds the job lock."""


@contextmanager
def acquire_job_lock(lock_path: Path) -> Iterator[Path]:
    """Acquire an exclusive non-blocking process lock across POSIX and Windows."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+b")
    locked = False
    try:
        fh.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise JobLockError(f"Another daily ASR job is already running (lock: {lock_path})") from exc

        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} started_at={datetime.now().isoformat(timespec='seconds')}\n".encode("utf-8"))
        fh.flush()
        yield lock_path
    finally:
        if locked:
            try:
                fh.seek(0)
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        fh.close()


def _resolve_against(raw: str, base_dir: Path) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def infer_platform(source: str) -> str:
    text = source.strip().lower()
    if (
        text.isdigit()
        or "bilibili.com" in text
        or "b23.tv" in text
        or re.fullmatch(r"bv[0-9a-z]+", text) is not None
    ):
        return "bilibili"
    return "youtube"


def build_runtime_biliyt_config(
    source_config_path: Path,
    runtime_config_path: Path,
    *,
    max_height: int = 360,
    target_filter: set[str] | None = None,
) -> tuple[Path, list[TargetSpec]]:
    """Derive a temporary biliyt YAML config without modifying the original repo/config.

    - Anchors all relative paths (download_dir, state_dir, temp_dir, cookies_file)
      to `source_config_path.parent`.
    - Enforces `media.max_height <= max_height` so yt-dlp downloads low-res video
      while still selecting the best audio stream (`bv*[height<=360]+ba/...`).
    - Disables temporary video metadata/chapter/thumbnail embedding to skip extra remuxing.
    """
    src = source_config_path.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"biliyt config not found: {src}")

    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML root in {src}: expected mapping")

    base_dir = src.parent
    derived: dict[str, Any] = copy.deepcopy(raw)

    paths_cfg = derived.setdefault("paths", {})
    if not isinstance(paths_cfg, dict):
        paths_cfg = {}
        derived["paths"] = paths_cfg

    download_root = _resolve_against(str(paths_cfg.get("download_dir") or "./downloads"), base_dir)
    state_dir = _resolve_against(str(paths_cfg.get("state_dir") or "./state"), base_dir)
    paths_cfg["download_dir"] = str(download_root)
    paths_cfg["state_dir"] = str(state_dir)

    raw_temp = paths_cfg.get("temp_dir")
    if isinstance(raw_temp, str) and raw_temp.strip():
        paths_cfg["temp_dir"] = str(_resolve_against(raw_temp, base_dir))

    # Anchor platform cookies_file paths
    platforms_cfg = derived.get("platforms")
    if isinstance(platforms_cfg, dict):
        for pcfg in platforms_cfg.values():
            if not isinstance(pcfg, dict):
                continue
            auth = pcfg.get("auth")
            if isinstance(auth, dict):
                cfile = auth.get("cookies_file")
                if isinstance(cfile, str) and cfile.strip():
                    auth["cookies_file"] = str(_resolve_against(cfile, base_dir))

    # Force low-res video (yt-dlp still picks best audio `ba`) and skip remux embeds
    media_cfg = derived.setdefault("media", {})
    if not isinstance(media_cfg, dict):
        media_cfg = {}
        derived["media"] = media_cfg
    media_cfg["max_height"] = int(max_height)
    media_cfg["embed_metadata"] = False
    media_cfg["embed_chapters"] = False
    media_cfg["embed_thumbnail"] = False

    selected_targets: list[TargetSpec] = []
    targets_raw = derived.get("targets") or []
    if isinstance(targets_raw, list):
        for t in targets_raw:
            if not isinstance(t, dict):
                continue
            # Anchor target-level cookies_file if set
            t_auth = t.get("auth")
            if isinstance(t_auth, dict):
                cfile = t_auth.get("cookies_file")
                if isinstance(cfile, str) and cfile.strip():
                    t_auth["cookies_file"] = str(_resolve_against(cfile, base_dir))

            # Clamp target-level media.max_height if overridden
            t_media = t.get("media")
            if isinstance(t_media, dict):
                cur_h = t_media.get("max_height")
                if cur_h is None or (isinstance(cur_h, int) and cur_h > max_height):
                    t_media["max_height"] = int(max_height)
                t_media["embed_metadata"] = False
                t_media["embed_chapters"] = False
                t_media["embed_thumbnail"] = False

            tid = str(t.get("id") or "").strip()
            if not tid:
                continue
            enabled = bool(t.get("enabled", True))
            if target_filter:
                if tid not in target_filter:
                    continue
            elif not enabled:
                continue

            platform = str(t.get("platform") or "").strip().lower() or infer_platform(str(t.get("source") or ""))
            subdir = t.get("output_subdir")
            if isinstance(subdir, str) and subdir.strip():
                t_dir = (download_root / subdir.strip()).resolve()
            else:
                t_dir = (download_root / platform / tid).resolve()
            selected_targets.append(TargetSpec(id=tid, platform=platform, download_dir=t_dir, enabled=enabled))

    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = runtime_config_path.with_name(f".{runtime_config_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(yaml.safe_dump(derived, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if os.name == "posix":
        os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, runtime_config_path)

    return download_root, selected_targets


def find_video_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    found: list[Path] = []
    for p in directory.rglob("*"):
        if not p.is_file() or p.is_symlink():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            found.append(p)
    return sorted(found)


def _open_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def get_pending_asr_groups(db_path: Path, allowed_groups: set[str] | None = None) -> dict[str, int]:
    """Return `{source_group: count}` for extracted audio rows whose ASR is not yet done."""
    if not db_path.is_file():
        return {}
    conn = _open_readonly_sqlite(db_path)
    try:
        rows = conn.execute(
            """
            SELECT source_group, COUNT(*) AS cnt
            FROM media_items
            WHERE audio_status = 'done' AND asr_status != 'done'
            GROUP BY source_group
            ORDER BY source_group
            """
        ).fetchall()
        result: dict[str, int] = {}
        for row in rows:
            grp = str(row["source_group"])
            if allowed_groups is not None and grp not in allowed_groups:
                continue
            cnt = int(row["cnt"])
            if cnt > 0:
                result[grp] = cnt
        return result
    finally:
        conn.close()


def _is_non_empty_file(raw_path: Any) -> bool:
    if not raw_path or not isinstance(raw_path, str):
        return False
    p = Path(raw_path)
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def cleanup_transcribed_videos(
    db_path: Path,
    download_root: Path,
    *,
    allowed_groups: set[str] | None = None,
    require_json: bool = True,
    dry_run: bool = False,
) -> CleanupSummary:
    """Safely delete original video files after verifying MP3 + transcript outputs exist.

    Original MP3 audio and all transcript artifacts (.txt, .srt, .json) are preserved.
    """
    summary = CleanupSummary()
    if not db_path.is_file():
        return summary

    resolved_root = download_root.resolve()
    conn = _open_readonly_sqlite(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, source_group, video_id, video_path, audio_path, txt_path, srt_path, json_path, asr_status
            FROM media_items
            WHERE asr_status = 'done'
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        grp = str(row["source_group"])
        if allowed_groups is not None and grp not in allowed_groups:
            continue

        raw_video = row["video_path"]
        if not raw_video:
            continue
        vpath = Path(str(raw_video))
        if not vpath.exists():
            summary.already_gone += 1
            continue

        if vpath.is_symlink() or not vpath.is_file() or vpath.suffix.lower() not in VIDEO_EXTENSIONS:
            log(f"[cleanup] SKIP non-video or symlink: {vpath}")
            summary.kept_unsafe_or_incomplete += 1
            continue

        try:
            resolved_video = vpath.resolve()
            if not resolved_video.is_relative_to(resolved_root):
                log(f"[cleanup] SKIP outside download_root ({resolved_root}): {resolved_video}")
                summary.kept_unsafe_or_incomplete += 1
                continue
        except OSError as exc:
            log(f"[cleanup] SKIP cannot resolve {vpath}: {exc}")
            summary.kept_unsafe_or_incomplete += 1
            continue

        missing_artifacts: list[str] = []
        if not _is_non_empty_file(row["audio_path"]):
            missing_artifacts.append("mp3(audio_path)")
        if not _is_non_empty_file(row["txt_path"]):
            missing_artifacts.append("txt_path")
        if not _is_non_empty_file(row["srt_path"]):
            missing_artifacts.append("srt_path")
        if require_json and not _is_non_empty_file(row["json_path"]):
            missing_artifacts.append("json_path")

        if missing_artifacts:
            log(f"[cleanup] KEEP {vpath.name} (missing/empty: {', '.join(missing_artifacts)})")
            summary.kept_unsafe_or_incomplete += 1
            continue

        size_bytes = vpath.stat().st_size
        if dry_run:
            log(f"[cleanup][dry-run] WOULD DELETE {vpath} ({size_bytes / (1024 * 1024):.1f} MiB)")
        else:
            vpath.unlink()
            log(f"[cleanup] DELETED {vpath} ({size_bytes / (1024 * 1024):.1f} MiB) | kept MP3 + transcripts")
        summary.deleted += 1
        summary.reclaimed_bytes += size_bytes

    return summary


Runner = Callable[[list[str], Path | None], int]


def default_runner(cmd: list[str], cwd: Path | None = None) -> int:
    log(f"$ {' '.join(cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    return proc.returncode


def run_biliyt_sync_with_lock_retry(
    biliyt_bin: str,
    runtime_config: Path,
    *,
    target_ids: list[str] | None = None,
    dry_run: bool = False,
    lock_wait_seconds: int = 300,
    lock_retry_interval: int = 15,
    runner: Runner = default_runner,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    cmd = [biliyt_bin, "sync", "-c", str(runtime_config), "--only", "videos"]
    for tid in target_ids or []:
        cmd.extend(["--target", tid])
    if dry_run:
        cmd.append("--dry-run")

    waited = 0
    while True:
        rc = runner(cmd, None)
        if rc != BILIYT_LOCK_EXIT_CODE:
            return rc
        if waited >= lock_wait_seconds:
            log(f"[sync] Timed out after waiting {waited}s for biliyt .sync.lock")
            return rc
        sleep_for = min(lock_retry_interval, lock_wait_seconds - waited)
        log(f"[sync] biliyt sync lock busy (exit 3); retrying in {sleep_for}s ({waited}/{lock_wait_seconds}s)...")
        sleeper(sleep_for)
        waited += sleep_for


def warmup_asr_endpoint(
    python_bin: str,
    asr_script: Path,
    asr_dir: Path,
    *,
    endpoint: str,
    timeout: int,
    retries: int,
    runner: Runner = default_runner,
    sleeper: Callable[[float], None] = time.sleep,
) -> bool:
    attempts = max(1, retries)
    cmd = [
        python_bin,
        str(asr_script),
        "endpoints",
        "--endpoint",
        endpoint,
        "--check",
        "--timeout",
        str(timeout),
    ]
    for attempt in range(1, attempts + 1):
        log(f"[asr] Warming up endpoint '{endpoint}' (attempt {attempt}/{attempts}, timeout={timeout}s)...")
        rc = runner(cmd, asr_dir)
        if rc == 0:
            log(f"[asr] Endpoint '{endpoint}' is healthy and ready.")
            return True
        if attempt < attempts:
            log(f"[asr] Endpoint health check returned {rc}; retrying in 5s...")
            sleeper(5)
    log(f"[asr] WARNING: Endpoint '{endpoint}' health check failed after {attempts} attempt(s).")
    return False


def load_job_config(path: Path | None) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path is None:
        default_local = REPO_ROOT / "job.yaml"
        if default_local.is_file():
            path = default_local
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"Job config file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid job config YAML in {path}: expected mapping")
        for k, v in raw.items():
            if v is not None:
                cfg[k] = v
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Daily orchestrator: biliyt low-res video download -> MP3 extract -> Qwen3-ASR -> safe video cleanup."
    )
    p.add_argument("--config", type=Path, default=None, help="Optional job.yaml override file")
    p.add_argument("--biliyt-config", type=Path, default=None, help="Path to biliyt config.yaml")
    p.add_argument("--biliyt-bin", type=str, default=None, help="Path to biliyt executable")
    p.add_argument("--asr-dir", type=Path, default=None, help="Path to qwen3-asr-pipeline repo")
    p.add_argument("--python-bin", type=str, default=None, help="Python executable for asr_pipeline.py")
    p.add_argument("--runtime-dir", type=Path, default=None, help="Directory for ephemeral runtime config and locks")
    p.add_argument("--max-height", type=int, default=None, help="Max video height for temporary download (default: 360)")
    p.add_argument("--audio-preset", type=str, default=None, help="asr_pipeline.py audio preset (default: asr-mp3)")
    p.add_argument("--extract-workers", type=int, default=None, help="Parallel ffmpeg workers (default: 4)")
    p.add_argument("--endpoint", type=str, default=None, help="ASR endpoint name in qwen3-asr-pipeline/.env (default: modal)")
    p.add_argument("--asr-concurrency", type=int, default=None, help="Concurrent ASR workers (default: 4)")
    p.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save raw ASR JSON alongside .txt and .srt (default: enabled)",
    )
    p.add_argument("--warmup-timeout", type=int, default=None, help="Endpoint health check timeout in seconds (default: 120)")
    p.add_argument("--warmup-retries", type=int, default=None, help="Endpoint warm-up retry attempts (default: 2)")
    p.add_argument("--asr-timeout", type=int, default=None, help="Per-request ASR HTTP timeout in seconds (default: 300)")
    p.add_argument("--task-timeout", type=int, default=None, help="Async ASR task max wait in seconds (default: 7200)")
    p.add_argument("--lock-wait-seconds", type=int, default=None, help="Max seconds to wait if biliyt sync lock is held")
    p.add_argument("--lock-retry-interval", type=int, default=None, help="Retry interval while waiting for biliyt sync lock")
    p.add_argument("--target", action="append", default=None, help="Limit run to specific target ID(s) (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="Preview sync, extraction, ASR, and cleanup without mutating files")
    p.add_argument("--skip-sync", action="store_true", help="Skip biliyt video download step")
    p.add_argument("--skip-extract", action="store_true", help="Skip ffmpeg audio extraction step")
    p.add_argument("--skip-transcribe", action="store_true", help="Skip ASR transcription step")
    p.add_argument("--skip-cleanup", action="store_true", help="Skip post-ASR video deletion step")
    return p


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_job_config(args.config)
    for key in DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg


def execute_pipeline(
    settings: dict[str, Any],
    *,
    target_ids: list[str] | None = None,
    dry_run: bool = False,
    skip_sync: bool = False,
    skip_extract: bool = False,
    skip_transcribe: bool = False,
    skip_cleanup: bool = False,
    runner: Runner = default_runner,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    biliyt_config = Path(str(settings["biliyt_config"])).expanduser().resolve()
    biliyt_bin = str(settings["biliyt_bin"])
    if not Path(biliyt_bin).is_file():
        resolved_bin = shutil.which(biliyt_bin) or shutil.which("biliyt")
        if resolved_bin:
            biliyt_bin = resolved_bin
    asr_dir = Path(str(settings["asr_dir"])).expanduser().resolve()
    asr_script = asr_dir / "asr_pipeline.py"
    db_path = asr_dir / "data" / "pipeline.db"
    python_bin = str(settings["python_bin"])
    runtime_dir = Path(str(settings["runtime_dir"])).expanduser().resolve()
    runtime_config = runtime_dir / "runtime_config.yaml"
    lock_path = runtime_dir / ".job.lock"
    save_json = bool(settings["save_json"])

    if not asr_script.is_file():
        log(f"[error] asr_pipeline.py not found at {asr_script}")
        return 1

    had_errors = False
    target_filter = set(target_ids) if target_ids else None

    with acquire_job_lock(lock_path):
        log(f"=== Daily Video ASR Job started (dry_run={dry_run}, save_json={save_json}) ===")

        download_root, targets = build_runtime_biliyt_config(
            biliyt_config,
            runtime_config,
            max_height=int(settings["max_height"]),
            target_filter=target_filter,
        )
        log(
            f"[config] Derived {runtime_config} from {biliyt_config} "
            f"(max_height={settings['max_height']}, targets={[t.id for t in targets]})"
        )
        if not targets:
            log("[config] No enabled targets selected. Nothing to do.")
            return 0

        selected_ids = {t.id for t in targets}

        # Step 1: Incremental low-res video download via biliyt
        if skip_sync:
            log("[sync] Skipped (--skip-sync).")
        else:
            rc = run_biliyt_sync_with_lock_retry(
                biliyt_bin,
                runtime_config,
                target_ids=target_ids,
                dry_run=dry_run,
                lock_wait_seconds=int(settings["lock_wait_seconds"]),
                lock_retry_interval=int(settings["lock_retry_interval"]),
                runner=runner,
                sleeper=sleeper,
            )
            if rc == 0:
                log("[sync] biliyt video sync completed cleanly.")
            elif rc == BILIYT_PARTIAL_FAILURE_EXIT_CODE:
                log("[sync] WARNING: biliyt video sync finished with partial failures (exit 2); continuing pipeline.")
                had_errors = True
            else:
                log(f"[sync] ERROR: biliyt video sync exited with code {rc}; continuing for existing local files.")
                had_errors = True

        # Step 2: Extract audio (MP3) for targets with video files on disk
        if skip_extract:
            log("[extract] Skipped (--skip-extract).")
        else:
            for t in targets:
                videos = find_video_files(t.download_dir)
                if not videos:
                    log(f"[extract] Target '{t.id}': 0 video files in {t.download_dir}; skipping.")
                    continue
                if dry_run:
                    log(f"[extract][dry-run] Target '{t.id}': would extract {len(videos)} video(s) from {t.download_dir}")
                    continue
                cmd = [
                    python_bin,
                    str(asr_script),
                    "extract",
                    str(t.download_dir),
                    "--group",
                    t.id,
                    "--preset",
                    str(settings["audio_preset"]),
                    "--workers",
                    str(int(settings["extract_workers"])),
                ]
                rc = runner(cmd, asr_dir)
                if rc != 0:
                    log(f"[extract] ERROR: extract failed for '{t.id}' (exit {rc})")
                    had_errors = True

        # Step 3: Warm up Modal (only when pending audio exists) & transcribe
        if skip_transcribe:
            log("[asr] Skipped (--skip-transcribe).")
        else:
            pending_groups = get_pending_asr_groups(db_path, allowed_groups=selected_ids)
            if not pending_groups:
                log("[asr] No extracted audio awaiting transcription; skipping endpoint warm-up & ASR.")
            elif dry_run:
                for grp, cnt in pending_groups.items():
                    log(
                        f"[asr][dry-run] Group '{grp}': would transcribe {cnt} item(s) "
                        f"via endpoint '{settings['endpoint']}' (save_json={save_json})"
                    )
            else:
                healthy = warmup_asr_endpoint(
                    python_bin,
                    asr_script,
                    asr_dir,
                    endpoint=str(settings["endpoint"]),
                    timeout=int(settings["warmup_timeout"]),
                    retries=int(settings["warmup_retries"]),
                    runner=runner,
                    sleeper=sleeper,
                )
                if not healthy:
                    had_errors = True

                for grp, cnt in pending_groups.items():
                    log(f"[asr] Transcribing group '{grp}' ({cnt} pending item(s))...")
                    cmd = [
                        python_bin,
                        str(asr_script),
                        "transcribe",
                        "--endpoint",
                        str(settings["endpoint"]),
                        "--group",
                        grp,
                        "--concurrency",
                        str(int(settings["asr_concurrency"])),
                        "--timeout",
                        str(int(settings["asr_timeout"])),
                        "--task-timeout",
                        str(int(settings["task_timeout"])),
                    ]
                    if save_json:
                        cmd.append("--save-json")
                    rc = runner(cmd, asr_dir)
                    if rc != 0:
                        log(f"[asr] ERROR: transcribe failed for group '{grp}' (exit {rc})")
                        had_errors = True

        # Step 4: Safe video cleanup (keep MP3 + transcripts)
        if skip_cleanup:
            log("[cleanup] Skipped (--skip-cleanup).")
        else:
            cs = cleanup_transcribed_videos(
                db_path,
                download_root,
                allowed_groups=selected_ids,
                require_json=save_json,
                dry_run=dry_run,
            )
            log(
                f"[cleanup] Summary: deleted={cs.deleted} "
                f"({cs.reclaimed_bytes / (1024 * 1024):.1f} MiB), "
                f"already_gone={cs.already_gone}, kept_incomplete={cs.kept_unsafe_or_incomplete}"
            )

        status = "COMPLETED WITH WARNINGS/ERRORS" if had_errors else "COMPLETED"
        log(f"=== Daily Video ASR Job {status} ===")
        return 1 if had_errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = resolve_settings(args)
    try:
        return execute_pipeline(
            settings,
            target_ids=args.target,
            dry_run=args.dry_run,
            skip_sync=args.skip_sync,
            skip_extract=args.skip_extract,
            skip_transcribe=args.skip_transcribe,
            skip_cleanup=args.skip_cleanup,
        )
    except JobLockError as exc:
        log(f"[lock] {exc}")
        return 3
    except Exception as exc:
        log(f"[fatal] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
