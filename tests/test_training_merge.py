"""Merging another install's training telemetry (Windows <-> Ubuntu) by run id. Temp databases only."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from dual_gpu_setup.training import merge as M
from dual_gpu_setup.training.store import TrainingStore


def new_db(path: Path) -> Path:
    TrainingStore(path).close()
    return path


def db_rows(path: Path, sql: str, args=()):
    c = sqlite3.connect(path)
    try:
        return c.execute(sql, args).fetchall()
    finally:
        c.close()


def add_run(path: Path, rid: str, name: str, start: float, end: float | None, steps: int = 3, status="completed",
            hw_every: float = 1.0, extra_config: dict | None = None) -> None:
    """One run with steps/events/rank rows and 1 Hz two-GPU hardware + system samples across its window."""
    c = sqlite3.connect(path)
    c.execute("PRAGMA foreign_keys = ON")
    c.execute("INSERT INTO training_runs (id, name, status, created_at, started_at, finished_at, last_step, last_step_at,"
              " last_heartbeat_at, config_json, world_size, agg_windows) VALUES (?,?,?,?,?,?,?,?,?,?,2,0)",
              (rid, name, status, start, start, end, steps, end or start, end or start,
               json.dumps({"data_dir": "D:\\data\\packed", **(extra_config or {})})))
    for s in range(1, steps + 1):
        c.execute("INSERT INTO training_steps (run_id, step, ts, loss, nonpad_tokens) VALUES (?,?,?,?,?)",
                  (rid, s, start + s, 9.0 - s, 1000 * s))
    c.execute("INSERT INTO training_events (run_id, eid, ts, kind) VALUES (?,?,?,?)", (rid, "e1", start + 1, "start"))
    for rank, bus in enumerate(("0000:03:00.0", "0000:0b:00.0")):
        c.execute("INSERT INTO training_ranks (run_id, rank, pci_bus, last_step) VALUES (?,?,?,?)", (rid, rank, bus, steps))
        c.execute("INSERT INTO training_rank_steps (run_id, step, rank, ts, pci_bus) VALUES (?,?,?,?,?)",
                  (rid, 1, rank, start + 1, bus))
    t = start
    while end is not None and t <= end:
        for bus in ("0000:03:00.0", "0000:0b:00.0"):
            c.execute("INSERT INTO hardware_samples (ts, pci_bus, lane, gpu_busy_pct, power_w, fan_rpm) VALUES (?,?,?,?,?,?)",
                      (t, bus, "r9700" if bus.startswith("0000:03") else "9070xt", 90.0, 250.0, None))
        c.execute("INSERT OR IGNORE INTO system_samples (ts, cpu_pct) VALUES (?,?)", (t, 12.0))
        t += hw_every
    c.commit()
    c.close()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def pair(tmp_path):
    dest, src = new_db(tmp_path / "ubuntu.sqlite3"), new_db(tmp_path / "windows.sqlite3")
    return dest, src


def test_imports_runs_with_all_children_and_hardware_window(pair):
    dest, src = pair
    add_run(src, "a" * 32, "win_smoke", 1000.0, 1010.0)
    add_run(src, "b" * 32, "win_stage1", 2000.0, 2020.0, steps=5)
    add_run(dest, "c" * 32, "ubuntu_run", 5000.0, 5010.0)
    before_steps = db_rows(dest, "select count(*) from training_steps")[0][0]
    r = M.merge_training_db(dest, src)
    assert r.ok and [x["name"] for x in r.runs_imported] == ["win_smoke", "win_stage1"]
    assert {x[0] for x in db_rows(dest, "select name from training_runs")} == {"win_smoke", "win_stage1", "ubuntu_run"}
    assert db_rows(dest, "select count(*) from training_steps")[0][0] == before_steps + 3 + 5
    assert db_rows(dest, "select count(*) from training_events where run_id=?", ("a" * 32,))[0][0] == 1
    assert db_rows(dest, "select count(*) from training_rank_steps where run_id=?", ("b" * 32,))[0][0] == 2
    assert db_rows(dest, "select count(*) from training_ranks")[0][0] == 6
    # 11 s window at 1 Hz x 2 GPUs (+ the dest's own run) and the 21 s run
    assert db_rows(dest, "select count(*) from hardware_samples where ts between 1000 and 1010")[0][0] == 22
    assert r.rows["training_runs"] == 2 and r.rows["hardware_samples"] > 0


def test_autoincrement_ids_never_collide(pair):
    dest, src = pair
    add_run(dest, "c" * 32, "local", 5000.0, 5005.0, steps=4)
    add_run(src, "a" * 32, "remote", 1000.0, 1005.0, steps=4)
    M.merge_training_db(dest, src)
    ids = [x[0] for x in db_rows(dest, "select id from training_steps")]
    assert len(ids) == len(set(ids)) == 8
    steps = db_rows(dest, "select step, loss from training_steps where run_id=? order by step", ("a" * 32,))
    assert steps == [(1, 8.0), (2, 7.0), (3, 6.0), (4, 5.0)]


def test_merge_is_idempotent(pair):
    dest, src = pair
    add_run(src, "a" * 32, "w", 1000.0, 1010.0)
    first = M.merge_training_db(dest, src)
    snapshot = {t: db_rows(dest, f"select count(*) from {t}")[0][0]
                for t in ("training_runs", "training_steps", "hardware_samples", "system_samples", "training_events")}
    second = M.merge_training_db(dest, src)
    assert len(first.runs_imported) == 1 and second.runs_imported == [] and len(second.runs_skipped) == 1
    assert second.rows == {}
    assert snapshot == {t: db_rows(dest, f"select count(*) from {t}")[0][0] for t in snapshot}


def test_existing_run_id_is_never_overwritten(pair):
    dest, src = pair
    add_run(dest, "a" * 32, "local_name", 1000.0, 1010.0, steps=3, status="completed")
    add_run(src, "a" * 32, "REMOTE_name", 1000.0, 1010.0, steps=9, status="failed")
    r = M.merge_training_db(dest, src)
    assert r.runs_imported == [] and len(r.runs_skipped) == 1
    assert db_rows(dest, "select name, status, last_step from training_runs") == [("local_name", "completed", 3)]


def test_hardware_samples_are_deduplicated_and_windowed(pair):
    dest, src = pair
    add_run(src, "a" * 32, "w", 1000.0, 1004.0)
    c = sqlite3.connect(src)    # a sample outside any run window must not be imported
    c.execute("INSERT INTO hardware_samples (ts, pci_bus, lane, gpu_busy_pct) VALUES (9999, '0000:03:00.0', 'r9700', 1)")
    c.commit(); c.close()
    c = sqlite3.connect(dest)   # the destination already holds one of the same samples
    c.execute("INSERT INTO hardware_samples (ts, pci_bus, lane, gpu_busy_pct) VALUES (1002, '0000:03:00.0', 'r9700', 55)")
    c.commit(); c.close()
    M.merge_training_db(dest, src)
    rows = db_rows(dest, "select count(*) from hardware_samples where pci_bus='0000:03:00.0' and ts=1002")
    assert rows == [(1,)], "duplicate (pci_bus, ts) must be skipped"
    assert db_rows(dest, "select gpu_busy_pct from hardware_samples where pci_bus='0000:03:00.0' and ts=1002") == [(55.0,)]
    assert db_rows(dest, "select count(*) from hardware_samples where ts=9999") == [(0,)]
    assert db_rows(dest, "select count(*) from hardware_samples where ts between 1000 and 1004")[0][0] == 10


def test_unavailable_sensors_stay_null_not_zero(pair):
    dest, src = pair
    add_run(src, "a" * 32, "w", 1000.0, 1003.0)
    M.merge_training_db(dest, src)
    assert db_rows(dest, "select count(*) from hardware_samples where fan_rpm is not null")[0][0] == 0
    assert db_rows(dest, "select count(*) from hardware_samples where fan_rpm = 0")[0][0] == 0


def test_source_is_never_modified(pair):
    dest, src = pair
    add_run(src, "a" * 32, "w", 1000.0, 1010.0)
    c = sqlite3.connect(src); c.execute("PRAGMA wal_checkpoint(TRUNCATE)"); c.close()
    before = sha(src)
    M.merge_training_db(dest, src)
    M.merge_training_db(dest, src, dry_run=True)
    assert sha(src) == before


def test_dry_run_changes_nothing_and_predicts_the_real_import(pair):
    dest, src = pair
    add_run(src, "a" * 32, "w", 1000.0, 1010.0)
    add_run(dest, "c" * 32, "local", 5000.0, 5003.0)
    c = sqlite3.connect(dest); c.execute("PRAGMA wal_checkpoint(TRUNCATE)"); c.close()
    counts = lambda: {t: db_rows(dest, f"select count(*) from {t}")[0][0] for t in  # noqa: E731
                      ("training_runs", "training_steps", "hardware_samples")}
    before, digest = counts(), sha(dest)
    dry = M.merge_training_db(dest, src, dry_run=True)
    assert counts() == before and sha(dest) == digest and dry.dry_run
    real = M.merge_training_db(dest, src)
    assert dry.rows["training_runs"] == real.rows["training_runs"] == 1
    assert dry.rows["hardware_samples"] == real.rows["hardware_samples"]
    assert dry.rows["training_steps"] == real.rows["training_steps"]


def test_dry_run_with_missing_destination_creates_nothing(tmp_path):
    src = new_db(tmp_path / "w.sqlite3")
    add_run(src, "a" * 32, "w", 1000.0, 1003.0)
    dest = tmp_path / "does_not_exist_yet" / "training.sqlite3"
    r = M.merge_training_db(dest, src, dry_run=True)
    assert r.ok and len(r.runs_imported) == 1 and not dest.exists() and any("does not exist" in w for w in r.warnings)


def test_each_run_is_all_or_nothing(pair, monkeypatch):
    dest, src = pair
    add_run(src, "a" * 32, "bad", 1000.0, 1003.0)
    add_run(src, "b" * 32, "good", 2000.0, 2003.0)
    real = M._import_run

    def flaky(conn, run, report, *rest):
        real(conn, run, report, *rest)                # rows are in, then the run fails part-way
        if run["name"] == "bad":
            raise sqlite3.IntegrityError("simulated failure after partial copy")
    monkeypatch.setattr(M, "_import_run", flaky)
    r = M.merge_training_db(dest, src)
    assert not r.ok and len(r.errors) == 1 and "bad" in r.errors[0]
    assert [x[0] for x in db_rows(dest, "select name from training_runs")] == ["good"]
    assert db_rows(dest, "select count(*) from training_steps where run_id=?", ("a" * 32,)) == [(0,)]
    assert db_rows(dest, "select count(*) from hardware_samples where ts between 1000 and 1003") == [(0,)]


def test_newer_schema_source_is_refused(pair):
    dest, src = pair
    add_run(src, "a" * 32, "w", 1000.0, 1003.0)
    c = sqlite3.connect(src); c.execute("PRAGMA user_version = 99"); c.commit(); c.close()
    r = M.merge_training_db(dest, src)
    assert not r.ok and "newer" in r.errors[0] and db_rows(dest, "select count(*) from training_runs") == [(0,)]


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 35, 0), reason="needs ALTER TABLE DROP COLUMN")
def test_older_source_schema_merges_on_common_columns(pair):
    dest, src = pair
    add_run(src, "a" * 32, "old", 1000.0, 1003.0)
    c = sqlite3.connect(src)
    c.execute("ALTER TABLE training_runs DROP COLUMN tags_json")
    c.commit(); c.close()
    r = M.merge_training_db(dest, src)
    assert r.ok and db_rows(dest, "select name, tags_json from training_runs") == [("old", None)]


def test_bad_inputs(tmp_path):
    dest = tmp_path / "d.sqlite3"
    assert "not found" in M.merge_training_db(dest, tmp_path / "nope.sqlite3").errors[0]
    same = new_db(tmp_path / "x.sqlite3")
    assert "same file" in M.merge_training_db(same, same).errors[0]
    empty = tmp_path / "empty.sqlite3"
    sqlite3.connect(empty).close()
    assert "training_runs" in M.merge_training_db(dest, empty).errors[0]


def test_windows_paths_are_kept_verbatim_as_history(pair):
    dest, src = pair
    add_run(src, "a" * 32, "w", 1000.0, 1003.0, extra_config={"out": "F:\\SLM\\runs"})
    M.merge_training_db(dest, src)
    cfg = json.loads(db_rows(dest, "select config_json from training_runs")[0][0])
    assert cfg["out"] == "F:\\SLM\\runs" and cfg["data_dir"] == "D:\\data\\packed"


def test_merged_runs_are_usable_by_the_store_and_foreign_keys_hold(pair):
    dest, src = pair
    add_run(src, "a" * 32, "w", 1000.0, 1010.0)
    M.merge_training_db(dest, src)
    c = sqlite3.connect(dest)
    assert c.execute("PRAGMA foreign_key_check").fetchall() == [] and c.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    c.close()
    store = TrainingStore(dest, clock=lambda: 1011.0)
    run = store.get_run("a" * 32)
    assert run["name"] == "w" and run["status"] == "completed"
    assert len(store.hardware_rows("a" * 32)) == 22
    store.close()


def test_two_sources_and_cli(pair, tmp_path, capsys):
    dest, src = pair
    src2 = new_db(tmp_path / "other.sqlite3")
    add_run(src, "a" * 32, "w1", 1000.0, 1003.0)
    add_run(src2, "b" * 32, "w2", 2000.0, 2003.0)
    assert M.main(["--dest", str(dest), "--from", str(src), "--from", str(src2)]) == 0
    assert "imported 1 run(s)" in capsys.readouterr().out
    assert db_rows(dest, "select count(*) from training_runs") == [(2,)]
    assert M.main(["--dest", str(dest), "--from", str(src), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["runs_skipped"][0]["name"] == "w1"
    assert M.main(["--dest", str(dest), "--from", str(tmp_path / "missing.sqlite3")]) == 1


def test_app_exposes_import_from_flag():
    app = pytest.importorskip("app")
    ns = app.build_parser().parse_args(["--import-from", "a.sqlite3", "--import-from", "b.sqlite3"])
    assert ns.import_from == ["a.sqlite3", "b.sqlite3"]
    assert app.build_parser().parse_args([]).import_from == []
