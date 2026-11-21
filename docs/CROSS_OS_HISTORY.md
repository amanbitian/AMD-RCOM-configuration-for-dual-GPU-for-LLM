# One history on Windows and Linux

Both installs keep their own databases on their own native disk and show **everything either of them ever recorded**:
inference runs (with full prompts, outputs and per-run samples), training runs, the hardware log, saved alerts and events.
Nothing is deleted, nothing existing is rewritten, and every row says which OS it came from (`origin_os` on runs,
`host_os` on telemetry).

**Never put a live database on the partition both systems write to.** A SQLite file on NTFS opened from Linux (and its
WAL side files, and Windows Fast Startup) is how databases get corrupted. `/api/health` warns when `--data-dir` is on an
NTFS/FAT partition. Share *files* instead.

## Share continuously: `--exchange-dir`

Point both installs at one shared folder (for example on the NTFS data partition that Linux mounts):

```bash
python app.py --config dual_gpu.toml --data-dir ~/dual-gpu-data/chat_runs --exchange-dir /media/you/shared/dual-gpu-exchange
```

On start, then hourly, and at shutdown each install

- writes `<exchange>/<os>/runs-YYYY-MM-DD.<os>.jsonl.gz` for finished runs that originated on it (and their metric samples);
- writes `<exchange>/<os>/training-<os>.sqlite3`, a consistent snapshot (SQLite backup API, a single file) of its training
  and hardware-log database, at most every six hours and always at shutdown;
- imports every other OS's files it has not imported yet. Rows already present are skipped; imported rows are never
  exported again, so nothing ping-pongs. A damaged or newer-format file is reported in `/api/health` and skipped.

Files are written to a temporary name and renamed, so the other side never reads half a file, and nothing in the folder
is ever deleted.

## One-off import of an existing database

```bash
# the Windows history (18,080 runs) into the Linux install; the source is opened read-only and never changed
python app.py --config dual_gpu.toml --data-dir ~/dual-gpu-data/chat_runs \
    --import-runs-from /media/you/Projects/Dual_gpu_setup/chat_runs/chatbot.sqlite3 \
    --import-from      /media/you/Projects/Dual_gpu_setup/chat_runs/training.sqlite3
```

A verified backup of every database is taken first (`<data-dir>/backups`), and the import aborts if any table lost rows.
Runs with no recorded OS predate OS capture, ran on Windows, and are tagged so with `origin_os_inferred = 1`.
Models are matched by `model_fingerprint` (size plus a hash of the first, middle and last MiB), not by path.
`--import-runs-from` also adds the per-project daily usage rows from the `project_usage.sqlite3` next to it.

Telemetry keeps its OS through the import: a Windows GPU sample recorded before rows were tagged arrives as `Windows`
(read from the source's runs). Before the first foreign row is added, this install's own untagged rows are tagged
with this install's OS. That replaces only an empty value, and it is skipped if another OS's runs are already present.

If the Windows install's service may still be writing, copy the whole `chat_runs` folder (each `.sqlite3` together with its
`-wal` and `-shm` files) to the Linux disk first and import from the copy. The copy then includes every committed row,
and the files on the shared partition are never opened.

**Done on 20 Nov 2026.** The Windows history was copied byte for byte to `~/dual-gpu-data/windows-snapshot-20261120`
(databases, logs and the 183 deployment folders) and imported from there:
- **Inference:** 18,080 runs and 876,922 metric samples.
- **Training:** 4 runs, plus 9,315 GPU samples and 3,105 system samples.
- **Other rows:** 183 deployments, 117 clients, 7 projects, 10 agent sessions and 19 daily usage rows.

No table lost rows. The pre-import backups are `<data-dir>/backups/*.pre-windows-import-20261120.bak`, and the full
report is `IMPORT_REPORT.txt` in the snapshot folder.

## The OS toggle

The dashboard header has **All / Windows / Ubuntu**. It filters the overview, analytics, run history, training runs and
the saved hardware log. The choice is remembered per browser. The hardware log's **Saved data** button jumps to the time
span that has samples for the chosen OS (the Windows samples are from 14 Nov 2026, outside the default last-hour view).
Live panels (current GPU readings, loaded models) always show this machine.

## Clocks

Timestamps are epoch seconds, so a combined timeline is only right if both systems have the correct time. Linux treats
the hardware clock as UTC and Windows as local time, so after a dual boot one of them can be hours off until it
resyncs. Setting Windows to UTC (admin, your call): `reg add "HKLM\SYSTEM\CurrentControlSet\Control\TimeZoneInformation" /v RealTimeIsUniversal /d 1 /t REG_DWORD /f`.
