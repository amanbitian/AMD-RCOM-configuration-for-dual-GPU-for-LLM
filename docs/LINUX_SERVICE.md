# Running Dual-GPU Studio on native Ubuntu

Target: the AMD Radeon AI PRO R9700 (32 GB) + RX 9070 XT (16 GB) workstation, native Ubuntu + ROCm. The companion
training repository (`SLM_training`) documents the PyTorch/ROCm stack in `docs/LINUX_SETUP.md`.

> **Status.** The Linux code paths (sysfs/procfs telemetry, PCI-first lane assignment, leases) are unit-tested against
> fake `/sys` and `/proc` trees. **Real gfx1201 sensors have not been validated yet**; section 6 is the checklist for
> doing that on the actual machine. Sensors a card does not expose are stored as `null`, never `0`.

## 1. Lane identities

Keys must be exactly `r9700` and `9070xt` (the trainer addresses GPUs by them; CI in both repositories checks this).
**PCI bus is the primary identity**; `match` (name) and `vram_gb` are fallbacks for lanes without a `pci_bus`.

```bash
cp example.dual_gpu.ubuntu.toml dual_gpu.toml
python scripts/detect_linux_lanes.py          # prints [[lanes]] with the real pci_bus values -> paste into dual_gpu.toml
lspci -nn | grep -i 'AMD/ATI'                 # cross-check the same addresses
```

Then set `[discovery].backend_server_binaries.rocm` to a ROCm build of `llama-server`
(`/opt/llama.cpp/build/bin/llama-server --list-devices` must list both GPUs) and `models_dir`.

## 2. Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .                 # package + dual-gpu CLI; also lets SLM_training import dual_gpu_setup (no DGPU_REPO needed)
sudo usermod -aG video,render,adm $USER    # adm or systemd-journal lets the sampler read amdgpu faults from the journal
```

Keep the database, logs and artifacts on Linux-native storage (`--data-dir ~/dual-gpu-data/chat_runs`), not an NTFS mount.

## 3. Run as a service

[`deploy/dual-gpu.service`](../deploy/dual-gpu.service) is a systemd **user** unit: binds `127.0.0.1:8090`, restarts on
failure, SIGTERM then SIGKILL after 60 s, writable paths limited to the data directory and `runs/`.

```bash
mkdir -p ~/.config/systemd/user ~/dual-gpu-data
cp deploy/dual-gpu.service ~/.config/systemd/user/ && $EDITOR ~/.config/systemd/user/dual-gpu.service
systemctl --user daemon-reload && systemctl --user enable --now dual-gpu.service
loginctl enable-linger "$USER"
journalctl --user -u dual-gpu.service -f
```

Do not bind to `0.0.0.0`/`--lan`: the API has no authentication.

## 4. Validate

```bash
python app.py --config ./dual_gpu.toml --check
python launch_service.py --start --config ./dual_gpu.toml --no-browser      # (skip if systemd already started it)
curl -s http://127.0.0.1:8090/api/health
curl -s http://127.0.0.1:8090/api/hardware | python -m json.tool | head -60
curl -s http://127.0.0.1:8090/api/leases
```

Expected: `/api/health` → `"status": "ok"`; `/api/hardware` → `hardware.available: true` and two entries in `gpus` whose `lane`
fields are `r9700` and `9070xt` (not `null`); `/api/leases` → `{"leases": [...]}`.

## 5. What the dashboard records per GPU (1 Hz)

utilisation, dedicated VRAM, GTT, board power, edge/junction/memory temperature, core and memory clock, fan, PCIe link
width/speed (and max), throttle state where exposed, kernel reset / page-fault / ring-timeout / PCIe events (from the
kernel log), and integrated energy (Wh) per run. Anything unavailable is `null` with a reason in `unavailable`.

## 6. Real-hardware sensor validation (run once on the workstation)

```bash
pip install pytest && python -m pytest -m rocm_hardware -q       # reads the real /sys; skipped on other machines
```

Then eyeball one training window in the dashboard against `amd-smi monitor` / `sensors`: power within ~10 %, VRAM and
temperatures matching, PCIe width equal to `lspci -vv` LnkSta, and a **no-sensor → null** check for any field `amd-smi`
also does not report. Record anything that differs; the tests cannot know what a real RDNA4 board exposes.

## 7. Rollback

`systemctl --user disable --now dual-gpu.service`; the repository files added for Linux are additive
(`deploy/`, `scripts/detect_linux_lanes.py`, this document, `tests/test_linux_deployment.py`, `.github/workflows/ci.yml`).
The only edited file is `example.dual_gpu.ubuntu.toml` (lane keys `gpu0/gpu1` → `r9700/9070xt`); any local
`dual_gpu.toml` that still uses `gpu0/gpu1` must be renamed together with the `lane` references in its `[[models]]`.

## 8. Bringing your Windows run history over

Windows and Ubuntu keep separate databases; merge instead of sharing one live file. See "Using one dashboard history
across Windows and Ubuntu" in [TRAINING_TELEMETRY.md](TRAINING_TELEMETRY.md): `python app.py ... --import-from <windows
training.sqlite3>` (read-only on the source, idempotent) or `python -m dual_gpu_setup.training.merge --dry-run ...`.
