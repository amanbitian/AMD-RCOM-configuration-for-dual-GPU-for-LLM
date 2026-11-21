#!/usr/bin/env bash
# Install Dual-GPU Studio as a systemd *user* service from this checkout (the live service used to be started by hand
# with nohup: no restart on failure, no journal, no clean stop). Re-running is safe. Nothing is started until you say so.
#
#   scripts/install_linux_service.sh [--data-dir ~/dual-gpu-data/chat_runs] [--config ~/dual-gpu-data/dual_gpu.toml] \
#                                    [--python ~/venvs/slm-rocm/bin/python] [--exchange-dir /media/you/shared/dual-gpu-exchange] \
#                                    [--enable]
#
# --exchange-dir shares run history and telemetry with the Windows install through plain files (see docs/CROSS_OS_HISTORY.md).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$HOME/dual-gpu-data/chat_runs"; config="$HOME/dual-gpu-data/dual_gpu.toml"; python="$HOME/venvs/slm-rocm/bin/python"
exchange=""; enable=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) data_dir="$2"; shift 2;; --config) config="$2"; shift 2;; --python) python="$2"; shift 2;;
    --exchange-dir) exchange="$2"; shift 2;; --enable) enable=1; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[ -x "$python" ] || { echo "python not found: $python" >&2; exit 1; }
[ -f "$config" ] || { echo "config not found: $config" >&2; exit 1; }
unit_dir="$HOME/.config/systemd/user"; mkdir -p "$unit_dir" "$data_dir"
args="--config $config --host 127.0.0.1 --port 8090 --data-dir $data_dir --no-browser"
writable="$data_dir $(dirname "$data_dir")"
if [ -n "$exchange" ]; then args="$args --exchange-dir $exchange"; writable="$writable $exchange"; fi
sed -e "s|^WorkingDirectory=.*|WorkingDirectory=$repo|" \
    -e "s|^ExecStart=.*|ExecStart=$python app.py $args|" \
    -e "s|^ReadWritePaths=.*|ReadWritePaths=$writable $repo/runs|" \
    "$repo/deploy/dual-gpu.service" > "$unit_dir/dual-gpu.service"
echo "wrote $unit_dir/dual-gpu.service"
systemctl --user daemon-reload
if [ "$enable" = 1 ]; then
  echo "A hand-started service on port 8090 must be stopped first (it would hold the port)."
  systemctl --user enable --now dual-gpu.service
  loginctl enable-linger "$USER" || true
  systemctl --user status dual-gpu.service --no-pager | head -12
else
  echo "Not started. When training is idle and the old nohup process is stopped:  systemctl --user enable --now dual-gpu.service"
fi
