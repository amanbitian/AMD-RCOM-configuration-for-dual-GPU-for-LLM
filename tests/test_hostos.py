from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual_gpu_setup.hostos import ProcessIdentity, get_host  # noqa: E402
from dual_gpu_setup.hostos import linux as lx  # noqa: E402
from dual_gpu_setup.hostos.base import HostOS  # noqa: E402

STAT = "{pid} ({comm}) S 1 {pid} {pid} 0 -1 4194560 {minflt} 0 {majflt} 0 {utime} {stime} 0 0 20 0 1 0 {start} 1000 100 18446744073709551615 0 0 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0\n"


def write_proc(root: Path, pid: int, *, comm="llama-server", utime=100, stime=50, start=4242, minflt=10, majflt=2,
               rss_kb=2048, anon_kb=1024, hwm_kb=4096) -> None:
    d = root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "stat").write_text(STAT.format(pid=pid, comm=comm, utime=utime, stime=stime, start=start, minflt=minflt, majflt=majflt))
    (d / "status").write_text(f"Name:\t{comm}\nVmRSS:\t{rss_kb} kB\nVmHWM:\t{hwm_kb} kB\nRssAnon:\t{anon_kb} kB\n")
    (d / "comm").write_text(comm + "\n")


# ---------------------------------------------------------------- Linux (fake /proc)


def test_stat_parser_handles_spaces_and_parens_in_comm(tmp_path):
    text = STAT.format(pid=7, comm="my (odd) name", utime=5, stime=6, start=99, minflt=1, majflt=2)
    parsed = lx.parse_stat(text)
    assert parsed["utime"] == 5 and parsed["stime"] == 6 and parsed["starttime"] == 99 and parsed["minflt"] == 1


def test_linux_process_probe_cpu_memory_identity_and_forget(tmp_path):
    write_proc(tmp_path, 10, utime=100, stime=100)
    probe = lx.LinuxProcessProbe(tmp_path)
    first = probe.metrics(10, now=0.0)
    assert first["cpu_pct"] is None and first["rss_bytes"] == 2048 * 1024 and first["private_bytes"] == 1024 * 1024
    assert first["page_fault_count"] == 12 and first["peak_rss_bytes"] == 4096 * 1024
    write_proc(tmp_path, 10, utime=100 + lx._CLK_TCK, stime=100)  # one CPU-second in one wall second
    second = probe.metrics(10, now=1.0)
    assert second["cpu_pct"] == pytest.approx(100.0 / (os.cpu_count() or 1), abs=0.01)
    identity = probe.identity(10)
    assert identity.pid == 10 and identity.start_token == 4242
    assert probe.metrics(999, now=1.0) == {} and probe.identity(999) is None
    probe.forget(10)
    assert probe.metrics(10, now=2.0)["cpu_pct"] is None


def test_linux_zombie_process_has_no_identity(tmp_path):
    write_proc(tmp_path, 11)
    path = tmp_path / "11" / "stat"
    path.write_text(path.read_text().replace(") S ", ") Z "))
    assert lx.LinuxProcessProbe(tmp_path).identity(11) is None


def test_linux_system_probe(tmp_path):
    (tmp_path / "meminfo").write_text("MemTotal: 1000 kB\nMemAvailable: 250 kB\nSwapTotal: 100 kB\nSwapFree: 40 kB\n")
    (tmp_path / "stat").write_text("cpu  100 0 100 700 100 0 0 0 0 0\n")
    probe = lx.LinuxSystemProbe(tmp_path)
    first = probe.metrics()
    assert first["cpu_pct"] is None and first["ram_used_bytes"] == 750 * 1024 and first["ram_used_pct"] == 75.0
    assert first["pagefile_used_bytes"] == 60 * 1024
    (tmp_path / "stat").write_text("cpu  200 0 200 800 200 0 0 0 0 0\n")  # +200 busy, +200 idle (idle+iowait)
    assert probe.metrics()["cpu_pct"] == pytest.approx(50.0)


def test_linux_gpu_probe_dedupes_shared_drm_clients(tmp_path):
    fdinfo = tmp_path / "20" / "fdinfo"
    fdinfo.mkdir(parents=True)
    drm = ("drm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t{cid}\n"
           "drm-memory-vram:\t{vram} KiB\ndrm-memory-gtt:\t{gtt} KiB\ndrm-engine-gfx:\t{ns} ns\n")
    (fdinfo / "5").write_text(drm.format(cid=1, vram=1048576, gtt=2048, ns=0))
    (fdinfo / "6").write_text(drm.format(cid=1, vram=1048576, gtt=2048, ns=0))  # same client: counted once
    (fdinfo / "7").write_text("pos:\t0\nflags:\t02\n")  # not a DRM fd
    probe = lx.LinuxGpuProbe(tmp_path)
    first = probe.process_metrics([20])[20]
    assert first["dedicated_gb"] == pytest.approx(1.0) and first["shared_gb"] == pytest.approx(2048 * 1024 / 1024**3)
    assert first["utilization_pct"] == 0.0
    assert probe.process_metrics([404]) == {}


def test_listening_socket_inode_lookup_and_server_detection(tmp_path):
    net = tmp_path / "net"
    net.mkdir()
    header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
    (net / "tcp").write_text(header + "   0: 0100007F:1F91 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000 0 555 1 0\n"
                             "   1: 0100007F:1F92 00000000:0000 01 00000000:00000000 00:00000000 00000000  1000 0 666 1 0\n")
    (net / "tcp6").write_text(header)
    assert lx._listening_inodes(tmp_path, 8081) == {"555"}  # 0x1F91; established (01) entries ignored
    assert lx._listening_inodes(tmp_path, 8082) == set()
    write_proc(tmp_path, 30, comm="llama-server")
    write_proc(tmp_path, 31, comm="python")
    control = lx.LinuxProcessControl(tmp_path)
    assert control.is_llama_server(30) and not control.is_llama_server(31)
    assert control.spawn_kwargs() == {"start_new_session": True}


# ------------------------------------------------------------- terminate_verified


class _Proc:
    def __init__(self, identity):
        self._identity = identity

    def identity(self, pid):
        return self._identity

    def metrics(self, pid, now):
        return {}

    def forget(self, pid):
        pass


class _Control:
    server_binary_name = "llama-server"

    def __init__(self):
        self.terminated = []

    def terminate_tree(self, pid, grace_s=10.0):
        self.terminated.append(pid)

    def pid_listening_on(self, port):
        return None

    def is_llama_server(self, pid):
        return False

    def spawn_kwargs(self):
        return {}


def fake_host(identity):
    control = _Control()
    return HostOS("fake", _Proc(identity), None, None, control), control


def test_terminate_verified_only_kills_the_original_process():
    host, control = fake_host(ProcessIdentity(100, start_token=7, exe="python"))
    assert host.terminate_verified(ProcessIdentity(100, 7)) is True and control.terminated == [100]
    host, control = fake_host(ProcessIdentity(100, start_token=8))  # PID recycled by a new process
    assert host.terminate_verified(ProcessIdentity(100, 7)) is False and control.terminated == []
    host, control = fake_host(None)  # already gone
    assert host.terminate_verified(ProcessIdentity(100, 7)) is False and control.terminated == []


# ------------------------------------------------------------------ Windows (real)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows host implementation")
def test_windows_probes_read_this_process():
    host = get_host()
    assert host.name == "windows" and host.control.server_binary_name == "llama-server.exe"
    identity = host.process.identity(os.getpid())
    assert identity.pid == os.getpid() and identity.start_token > 0 and identity.exe
    assert host.process.identity(os.getpid()).start_token == identity.start_token
    metrics = host.process.metrics(os.getpid(), now=1.0)
    assert metrics["rss_bytes"] > 0 and metrics["page_fault_count"] is not None
    system = host.system.metrics()
    assert system["ram_total_bytes"] > 0 and 0 <= system["ram_used_pct"] <= 100
    assert host.gpu.process_metrics([]) == {}
    assert host.control.spawn_kwargs()["creationflags"] != 0
    assert host.terminate_verified(ProcessIdentity(os.getpid(), identity.start_token + 1)) is False  # refuses; we survive


@pytest.mark.skipif(sys.platform != "win32", reason="Windows host implementation")
def test_windows_gpu_counter_output_parsing():
    from dual_gpu_setup.hostos.windows import parse_gpu_counter_output

    parsed = parse_gpu_counter_output("123 1073741824 2147483648 250\nbad line\n456 0 0 12.5\n")
    assert parsed[123] == {"dedicated_gb": 1.0, "shared_gb": 2.0, "utilization_pct": 100.0}  # clamped to 100
    assert parsed[456]["utilization_pct"] == 12.5 and len(parsed) == 2
