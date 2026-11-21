"""Windows + Ubuntu history in one dashboard: OS tags survive the import, and the All / Windows / Ubuntu filter."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual_gpu_setup import datasafe  # noqa: E402
from dual_gpu_setup.exchange import import_usage_db  # noqa: E402
from dual_gpu_setup.hardware_log import HardwareLog  # noqa: E402
from dual_gpu_setup.osfilter import normalize_os  # noqa: E402
from dual_gpu_setup.training.merge import merge_training_db  # noqa: E402
from dual_gpu_setup.training.store import TrainingStore  # noqa: E402

from test_hardware_log import Clock, gpu_sample, make_runs_db  # noqa: E402


def _raw_sample(store, ts, host_os, bus="0000:03:00.0"):
    with store.connect() as c:
        c.execute("INSERT INTO hardware_samples (ts, pci_bus, lane, temp_junction_c, power_w, host_os) VALUES (?,?,?,?,?,?)",
                  (ts, bus, "r9700", 70.0, 200.0, host_os))


def _os_counts(path):
    with sqlite3.connect(path) as c:
        return dict(c.execute("SELECT COALESCE(host_os, 'NULL'), COUNT(*) FROM hardware_samples GROUP BY 1").fetchall())


@pytest.mark.parametrize("given,expected", [(None, None), ("", None), ("all", None), ("Windows", "Windows"),
                                            ("ubuntu", "Linux"), ("LINUX", "Linux")])
def test_os_names(given, expected):
    assert normalize_os(given) == expected


def test_unknown_os_name_is_an_error_not_an_empty_page():
    with pytest.raises(ValueError):
        normalize_os("macos")


def test_untagged_windows_samples_arrive_tagged_and_local_untagged_rows_get_this_os(tmp_path):
    windows = TrainingStore(tmp_path / "w" / "training.sqlite3", clock=Clock(1000.0))
    windows.create_run({"name": "smoke", "host_os": {"system": "Windows"}})
    _raw_sample(windows, 1001.0, None)                  # recorded before rows were tagged
    _raw_sample(windows, 5000.0, None)                  # outside any run: only --all-telemetry brings it
    linux = TrainingStore(tmp_path / "l" / "training.sqlite3", clock=Clock(9000.0))
    _raw_sample(linux, 9001.0, None)
    _raw_sample(linux, 9002.0, "Linux")
    before = datasafe.table_counts(linux.path)

    report = merge_training_db(linux.path, windows.path, all_telemetry=True, dest_os="Linux")
    assert report.ok and report.source_os == "Windows"
    assert report.rows_tagged == {"hardware_samples": 1}
    assert _os_counts(linux.path) == {"Linux": 2, "Windows": 2}
    assert datasafe.verify_counts_not_lower(before, datasafe.table_counts(linux.path)) == []

    again = merge_training_db(linux.path, windows.path, all_telemetry=True, dest_os="Linux")
    assert again.ok and not again.rows.get("hardware_samples") and _os_counts(linux.path) == {"Linux": 2, "Windows": 2}
    assert _os_counts(windows.path) == {"NULL": 2}      # the source is never modified


def test_a_source_schema_without_the_os_column_is_tagged_from_its_runs(tmp_path):
    windows = TrainingStore(tmp_path / "w" / "training.sqlite3", clock=Clock(1000.0))
    windows.create_run({"name": "smoke", "host_os": {"system": "Windows"}})
    _raw_sample(windows, 1001.0, None)
    with sqlite3.connect(windows.path) as c:            # an older Windows install's schema
        for index in [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='hardware_samples' "
                                              "AND sql LIKE '%host_os%'")]:
            c.execute(f"DROP INDEX {index}")
        c.execute("ALTER TABLE hardware_samples DROP COLUMN host_os")
    linux = TrainingStore(tmp_path / "l" / "training.sqlite3")
    assert merge_training_db(linux.path, windows.path, all_telemetry=True, dest_os="Linux").ok
    assert _os_counts(linux.path) == {"Windows": 1}


def test_local_untagged_rows_are_left_alone_once_foreign_runs_are_already_merged(tmp_path):
    linux = TrainingStore(tmp_path / "l" / "training.sqlite3")
    linux.create_run({"name": "old-import", "host_os": {"system": "Windows"}})   # merged by an older version
    _raw_sample(linux, 9001.0, None)
    other = TrainingStore(tmp_path / "x" / "training.sqlite3")
    report = merge_training_db(linux.path, other.path, dest_os="Linux")
    assert report.ok and not report.rows_tagged and any("left untagged" in w for w in report.warnings)
    assert _os_counts(linux.path) == {"NULL": 1}


def test_hardware_log_filters_every_view_by_os(tmp_path):
    clock = Clock(10_000.0)
    store = TrainingStore(tmp_path / "t.sqlite3", clock=clock)
    log = HardwareLog(store, "Linux", lambda: {"r9700": {"model": "q"}}, clock=clock)
    log.on_gpu_samples([gpu_sample(clock.now)])
    _raw_sample(store, 5_000.0, "Windows")
    _raw_sample(store, 5_001.0, "Windows")

    assert sum(g["samples"] for g in log.summary(0, 20_000, host_os="Windows")["gpus"].values()) == 2
    assert sum(g["samples"] for g in log.summary(0, 20_000, host_os="Linux")["gpus"].values()) == 1
    assert sum(g["samples"] for g in log.summary(0, 20_000)["gpus"].values()) == 3
    assert sum(g["samples"] for g in log.history(0, 20_000, host_os="Windows")["gpus"].values()) == 2
    assert log.history(0, 20_000, host_os="Windows")["host"] == []           # Linux host rows are not Windows'
    _, body = log.export(0, 20_000, "jsonl", host_os="Windows")
    assert body.count("\n") == 2 and "Linux" not in body
    _, system_rows = log.export(0, 20_000, "csv", table="system_samples", host_os="Windows")   # no OS column: not an error
    rng = log.data_range("Windows")
    assert (rng["first_ts"], rng["last_ts"], rng["samples"]) == (5_000.0, 5_001.0, 2)
    assert set(log.data_range()["per_os"]) == {"Windows", "Linux"}


def test_run_history_and_dashboard_filter_by_os(tmp_path):
    from dual_gpu_setup.exchange import import_runs_db

    dest = make_runs_db(tmp_path / "l" / "chatbot.sqlite3", "Linux", ["l1"])
    import_runs_db(dest.path, make_runs_db(tmp_path / "w" / "chatbot.sqlite3", "Windows", ["w1", "w2"]).path)
    assert {r["id"] for r in dest.recent(host_os="Windows")} == {"w1", "w2"}
    assert dest.count_runs(host_os="Linux") == 1 and dest.count_runs() == 3
    board = dest.dashboard(host_os="Windows")
    assert board["summary"]["total_runs"] == 2 and [r["os"] for r in board["operating_systems"]] == ["Windows"]
    assert dest.dashboard()["summary"]["total_runs"] == 3


def test_training_runs_filter_by_os(tmp_path):
    store = TrainingStore(tmp_path / "t.sqlite3")
    store.create_run({"name": "win", "host_os": {"system": "Windows"}})
    store.create_run({"name": "lin", "host_os": {"system": "Linux"}})
    assert [r["name"] for r in store.list_runs(host_os="Windows")["runs"]] == ["win"]
    assert store.list_runs(host_os="Linux")["total"] == 1 and store.list_runs()["total"] == 2


def test_usage_rows_are_added_and_existing_ones_kept(tmp_path):
    def usage_db(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as c:
            c.execute("CREATE TABLE project_usage (project_id TEXT NOT NULL, connection_type TEXT NOT NULL, usage_date TEXT NOT NULL,"
                      " requests INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL"
                      " DEFAULT 0, updated_at TEXT NOT NULL, PRIMARY KEY (project_id, connection_type, usage_date))")
            c.executemany("INSERT INTO project_usage VALUES (?,?,?,?,?,?,?)", rows)
        return path

    dest = usage_db(tmp_path / "l" / "u.sqlite3", [("p", "api", "2026-11-01", 5, 1, 1, "x")])
    src = usage_db(tmp_path / "w" / "u.sqlite3", [("p", "api", "2026-11-01", 99, 9, 9, "y"), ("p", "api", "2026-10-01", 3, 1, 1, "y")])
    assert import_usage_db(dest, src)["imported"] == {"project_usage": 1}
    with sqlite3.connect(dest) as c:
        assert c.execute("SELECT requests FROM project_usage WHERE usage_date = '2026-11-01'").fetchone()[0] == 5   # kept
        assert c.execute("SELECT COUNT(*) FROM project_usage").fetchone()[0] == 2
