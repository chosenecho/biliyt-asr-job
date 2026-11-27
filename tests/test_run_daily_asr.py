"""Offline unit and end-to-end orchestration tests for run_daily_asr.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import run_daily_asr as job
import yaml


def _init_pipeline_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE media_items (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            source_group      TEXT NOT NULL,
            video_id          TEXT NOT NULL,
            title             TEXT NOT NULL,
            video_path        TEXT NOT NULL UNIQUE,
            video_rel_path    TEXT NOT NULL,
            video_size_bytes  INTEGER NOT NULL DEFAULT 0,
            video_mtime       REAL NOT NULL DEFAULT 0,
            duration_sec      REAL,
            audio_status      TEXT NOT NULL DEFAULT 'pending',
            audio_preset      TEXT,
            audio_path        TEXT,
            audio_format      TEXT,
            audio_size_bytes  INTEGER,
            audio_error       TEXT,
            audio_started_at  TEXT,
            audio_finished_at TEXT,
            asr_status        TEXT NOT NULL DEFAULT 'pending',
            asr_mode          TEXT,
            asr_server_url    TEXT,
            asr_task_id       TEXT,
            asr_progress      REAL NOT NULL DEFAULT 0.0,
            asr_message       TEXT,
            txt_path          TEXT,
            srt_path          TEXT,
            json_path         TEXT,
            asr_error         TEXT,
            asr_attempts      INTEGER NOT NULL DEFAULT 0,
            asr_started_at    TEXT,
            asr_updated_at    TEXT,
            asr_finished_at   TEXT,
            created_at        TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
            updated_at        TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z'
        )
        """
    )
    conn.commit()
    return conn


def _write_sample_biliyt_config(biliyt_dir: Path) -> Path:
    cfg = {
        "paths": {
            "download_dir": "./downloads",
            "state_dir": "./state",
        },
        "media": {
            "max_height": 1080,
            "embed_metadata": True,
            "embed_chapters": True,
            "embed_thumbnail": True,
        },
        "platforms": {
            "bilibili": {
                "auth": {"cookies_file": "./cookies/bili.txt"},
                "archive": {"videos": False},
            }
        },
        "targets": [
            {
                "id": "bili_100",
                "platform": "bilibili",
                "source": "https://space.bilibili.com/100",
                "enabled": True,
            },
            {
                "id": "bili_200",
                "source": "200",
                "enabled": True,
                "media": {"max_height": 2160},
            },
            {
                "id": "yt_off",
                "platform": "youtube",
                "source": "@example",
                "enabled": False,
            },
        ],
    }
    cfg_path = biliyt_dir / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_build_runtime_biliyt_config_preserves_source_and_anchors_paths(tmp_path: Path) -> None:
    biliyt_dir = tmp_path / "biliyt"
    src_cfg = _write_sample_biliyt_config(biliyt_dir)
    original_text = src_cfg.read_text(encoding="utf-8")

    runtime_cfg = tmp_path / "job" / "runtime_config.yaml"
    download_root, targets = job.build_runtime_biliyt_config(src_cfg, runtime_cfg, max_height=360)

    # Original config untouched
    assert src_cfg.read_text(encoding="utf-8") == original_text

    assert download_root == (biliyt_dir / "downloads").resolve()
    assert [t.id for t in targets] == ["bili_100", "bili_200"]
    assert targets[1].platform == "bilibili"

    derived = yaml.safe_load(runtime_cfg.read_text(encoding="utf-8"))
    assert derived["paths"]["download_dir"] == str((biliyt_dir / "downloads").resolve())
    assert derived["paths"]["state_dir"] == str((biliyt_dir / "state").resolve())
    assert derived["platforms"]["bilibili"]["auth"]["cookies_file"] == str(
        (biliyt_dir / "cookies" / "bili.txt").resolve()
    )
    assert derived["media"]["max_height"] == 360
    assert derived["media"]["embed_metadata"] is False
    assert derived["targets"][1]["media"]["max_height"] == 360


def test_sync_retries_on_lock_exit_code_3(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []
    returns = iter([3, 3, 0])

    def fake_runner(cmd: list[str], cwd: Path | None) -> int:
        calls.append(cmd)
        return next(returns)

    rc = job.run_biliyt_sync_with_lock_retry(
        "biliyt",
        tmp_path / "runtime_config.yaml",
        lock_wait_seconds=60,
        lock_retry_interval=10,
        runner=fake_runner,
        sleeper=sleeps.append,
    )
    assert rc == 0
    assert len(calls) == 3
    assert sleeps == [10, 10]


def test_cleanup_deletes_only_fully_verified_videos_and_keeps_mp3(tmp_path: Path) -> None:
    download_root = tmp_path / "biliyt" / "downloads"
    target_dir = download_root / "bilibili" / "bili_100"
    target_dir.mkdir(parents=True)

    # Case 1: Fully transcribed with mp3 + txt + srt + json -> MUST DELETE video, KEEP mp3
    v_ok = target_dir / "2026-11-27 - Ok [BV1ok].mp4"
    v_ok.write_bytes(b"video-bytes-ok")
    a_ok = tmp_path / "asr" / "data" / "audio" / "bili_100" / "v_ok.mp3"
    t_ok = tmp_path / "asr" / "data" / "transcripts" / "bili_100" / "v_ok.txt"
    s_ok = tmp_path / "asr" / "data" / "transcripts" / "bili_100" / "v_ok.srt"
    j_ok = tmp_path / "asr" / "data" / "transcripts" / "bili_100" / "v_ok.json"
    for p, data in [(a_ok, b"mp3"), (t_ok, b"hello"), (s_ok, b"1\n00:00:00 --> 00:00:01\nhi"), (j_ok, b"{}")]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    # Case 2: Missing json when require_json=True -> MUST KEEP video
    v_nojson = target_dir / "2026-11-27 - NoJson [BV1nj].mp4"
    v_nojson.write_bytes(b"video-bytes-nojson")
    a_nj = tmp_path / "asr" / "data" / "audio" / "bili_100" / "v_nj.mp3"
    t_nj = tmp_path / "asr" / "data" / "transcripts" / "bili_100" / "v_nj.txt"
    s_nj = tmp_path / "asr" / "data" / "transcripts" / "bili_100" / "v_nj.srt"
    for p, data in [(a_nj, b"mp3"), (t_nj, b"hello"), (s_nj, b"srt")]:
        p.write_bytes(data)

    # Case 3: ASR failed -> MUST KEEP video
    v_fail = target_dir / "2026-11-27 - Fail [BV1fl].mp4"
    v_fail.write_bytes(b"video-bytes-fail")

    # Case 4: Video outside download_root -> MUST KEEP video (path traversal safety)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"precious")

    db_path = tmp_path / "asr" / "data" / "pipeline.db"
    conn = _init_pipeline_db(db_path)
    conn.executemany(
        """
        INSERT INTO media_items (
            source_group, video_id, title, video_path, video_rel_path,
            audio_status, audio_path, asr_status, txt_path, srt_path, json_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("bili_100", "BV1ok", "Ok", str(v_ok), v_ok.name, "done", str(a_ok), "done", str(t_ok), str(s_ok), str(j_ok)),
            ("bili_100", "BV1nj", "NoJson", str(v_nojson), v_nojson.name, "done", str(a_nj), "done", str(t_nj), str(s_nj), None),
            ("bili_100", "BV1fl", "Fail", str(v_fail), v_fail.name, "done", str(a_ok), "failed", None, None, None),
            ("bili_100", "BV1out", "Out", str(outside), outside.name, "done", str(a_ok), "done", str(t_ok), str(s_ok), str(j_ok)),
        ],
    )
    conn.commit()
    conn.close()

    summary = job.cleanup_transcribed_videos(db_path, download_root, require_json=True, dry_run=False)
    assert summary.deleted == 1
    assert summary.reclaimed_bytes == len(b"video-bytes-ok")
    assert summary.kept_unsafe_or_incomplete == 2  # nojson + outside (failed isn't queried)
    assert not v_ok.exists()
    assert a_ok.exists()  # MP3 preserved
    assert t_ok.exists() and s_ok.exists() and j_ok.exists()
    assert v_nojson.exists()
    assert v_fail.exists()
    assert outside.exists()


def test_execute_pipeline_skips_warmup_when_no_pending_asr(tmp_path: Path) -> None:
    biliyt_dir = tmp_path / "biliyt"
    src_cfg = _write_sample_biliyt_config(biliyt_dir)
    asr_dir = tmp_path / "asr"
    asr_dir.mkdir()
    (asr_dir / "asr_pipeline.py").write_text("# stub\n", encoding="utf-8")
    _init_pipeline_db(asr_dir / "data" / "pipeline.db").close()

    commands: list[list[str]] = []

    def fake_runner(cmd: list[str], cwd: Path | None) -> int:
        commands.append(cmd)
        return 0

    settings: dict[str, Any] = dict(job.DEFAULTS)
    settings.update(
        {
            "biliyt_config": src_cfg,
            "biliyt_bin": "biliyt",
            "asr_dir": asr_dir,
            "python_bin": "python3",
            "runtime_dir": tmp_path / "runtime",
        }
    )

    rc = job.execute_pipeline(settings, runner=fake_runner, sleeper=lambda _: None)
    assert rc == 0
    # Only biliyt sync ran; extract & warmup & transcribe were skipped cleanly
    assert len(commands) == 1
    assert commands[0][1] == "sync"
    assert "--only" in commands[0] and "videos" in commands[0]


def test_execute_pipeline_full_flow_with_save_json_and_cleanup(tmp_path: Path) -> None:
    biliyt_dir = tmp_path / "biliyt"
    src_cfg = _write_sample_biliyt_config(biliyt_dir)
    asr_dir = tmp_path / "asr"
    asr_dir.mkdir()
    (asr_dir / "asr_pipeline.py").write_text("# stub\n", encoding="utf-8")
    db_path = asr_dir / "data" / "pipeline.db"
    _init_pipeline_db(db_path).close()

    video_file = biliyt_dir / "downloads" / "bilibili" / "bili_100" / "2026-11-27 - Demo [BV1demo].mp4"
    audio_file = asr_dir / "data" / "audio" / "bili_100" / "demo.mp3"
    txt_file = asr_dir / "data" / "transcripts" / "bili_100" / "demo.txt"
    srt_file = asr_dir / "data" / "transcripts" / "bili_100" / "demo.srt"
    json_file = asr_dir / "data" / "transcripts" / "bili_100" / "demo.json"

    commands: list[list[str]] = []

    def fake_runner(cmd: list[str], cwd: Path | None) -> int:
        commands.append(cmd)
        if "sync" in cmd:
            video_file.parent.mkdir(parents=True, exist_ok=True)
            video_file.write_bytes(b"fake-mp4-data")
        elif "extract" in cmd:
            audio_file.parent.mkdir(parents=True, exist_ok=True)
            audio_file.write_bytes(b"fake-mp3-data")
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                INSERT INTO media_items (
                    source_group, video_id, title, video_path, video_rel_path,
                    audio_status, audio_path, asr_status
                ) VALUES (?, ?, ?, ?, ?, 'done', ?, 'pending')
                """,
                ("bili_100", "BV1demo", "Demo", str(video_file), video_file.name, str(audio_file)),
            )
            conn.commit()
            conn.close()
        elif "transcribe" in cmd:
            assert "--save-json" in cmd
            for p, data in [(txt_file, b"text"), (srt_file, b"srt"), (json_file, b'{"text":"text"}')]:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                UPDATE media_items
                SET asr_status='done', txt_path=?, srt_path=?, json_path=?
                WHERE video_id='BV1demo'
                """,
                (str(txt_file), str(srt_file), str(json_file)),
            )
            conn.commit()
            conn.close()
        return 0

    settings: dict[str, Any] = dict(job.DEFAULTS)
    settings.update(
        {
            "biliyt_config": src_cfg,
            "biliyt_bin": "biliyt",
            "asr_dir": asr_dir,
            "python_bin": "python3",
            "runtime_dir": tmp_path / "runtime",
            "save_json": True,
        }
    )

    rc = job.execute_pipeline(settings, runner=fake_runner, sleeper=lambda _: None)
    assert rc == 0
    assert [c[1] if c[0] == "biliyt" else c[2] for c in commands] == [
        "sync",
        "extract",
        "endpoints",
        "transcribe",
    ]
    assert not video_file.exists()
    assert audio_file.exists()
    assert txt_file.exists() and srt_file.exists() and json_file.exists()
