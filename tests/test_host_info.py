from __future__ import annotations

import json

from dual_gpu_setup import host_info
from dual_gpu_setup.tasks import _shell_command


def test_os_info_records_linux_distribution(monkeypatch):
    monkeypatch.setattr(host_info.sys, "platform", "linux")
    monkeypatch.setattr(host_info.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host_info.platform, "release", lambda: "6.8.0")
    monkeypatch.setattr(host_info.platform, "version", lambda: "kernel build")
    monkeypatch.setattr(host_info.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host_info.platform, "freedesktop_os_release", lambda: {
        "PRETTY_NAME": "Ubuntu 24.04.3 LTS", "ID": "ubuntu", "VERSION_ID": "24.04",
    })

    info = host_info.os_info()

    assert info["system"] == "Linux"
    assert info["distribution"] == "Ubuntu 24.04.3 LTS"
    assert info["distribution_id"] == "ubuntu"
    assert json.loads(host_info.os_log_line())["distribution_version"] == "24.04"


def test_detached_process_kwargs_use_a_new_posix_session(monkeypatch):
    monkeypatch.setattr(host_info.sys, "platform", "linux")
    assert host_info.detached_process_kwargs() == {"start_new_session": True, "close_fds": True}


def test_detached_process_kwargs_use_windows_flags(monkeypatch):
    monkeypatch.setattr(host_info.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(host_info.subprocess, "DETACHED_PROCESS", 0x8, raising=False)
    monkeypatch.setattr(host_info.sys, "platform", "win32")
    options = host_info.detached_process_kwargs()
    expected = 0x200 | 0x8
    assert options == {"creationflags": expected, "close_fds": True}


def test_shell_command_uses_posix_quoting_on_linux(monkeypatch):
    monkeypatch.setattr("dual_gpu_setup.tasks.sys.platform", "linux")
    assert _shell_command(["python3", "my script.py", "it's-safe"]) == "python3 'my script.py' 'it'\"'\"'s-safe'"


def test_linux_uses_the_ubuntu_example_by_default():
    assert host_info.default_config_name("linux") == "example.dual_gpu.ubuntu.toml"
    assert host_info.default_config_name("win32") == "example.dual_gpu.toml"
