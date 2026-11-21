<#
.SYNOPSIS
  Start Dual-GPU Studio automatically at logon on Windows (Task Scheduler), restarting it if it exits.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install_windows_task.ps1 -Python C:\venvs\dgpu\Scripts\python.exe `
      -Config D:\dual-gpu\dual_gpu.toml -DataDir D:\dual-gpu\chat_runs -ExchangeDir D:\shared\dual-gpu-exchange
  Remove it:  Unregister-ScheduledTask -TaskName "Dual-GPU Studio" -Confirm:$false
.NOTES
  Keep -DataDir on a native NTFS disk of this Windows install, never on a partition Linux also writes to; use
  -ExchangeDir (plain files) to share history with the Linux install. Not started until you run the task or log in.
#>
param(
  [Parameter(Mandatory = $true)][string]$Python,
  [Parameter(Mandatory = $true)][string]$Config,
  [string]$DataDir = (Join-Path $env:USERPROFILE "dual-gpu-data\chat_runs"),
  [string]$ExchangeDir = "",
  [int]$Port = 8090
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not (Test-Path $Python)) { throw "python not found: $Python" }
if (-not (Test-Path $Config)) { throw "config not found: $Config" }
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$arguments = "app.py --config `"$Config`" --host 127.0.0.1 --port $Port --data-dir `"$DataDir`" --no-browser"
if ($ExchangeDir) { $arguments += " --exchange-dir `"$ExchangeDir`"" }
$action = New-ScheduledTaskAction -Execute $Python -Argument $arguments -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "Dual-GPU Studio" -Action $action -Trigger $trigger -Settings $settings `
  -Description "Dual-GPU Studio service (llama-server lanes, telemetry, GPU leases)" -Force | Out-Null
Write-Host "Registered task 'Dual-GPU Studio'. Start it now with:  Start-ScheduledTask -TaskName 'Dual-GPU Studio'"
Write-Host "Windows: set the hardware clock to UTC so a dual boot does not shift timestamps by hours:"
Write-Host '  reg add "HKLM\SYSTEM\CurrentControlSet\Control\TimeZoneInformation" /v RealTimeIsUniversal /d 1 /t REG_DWORD /f   (admin; this changes a system setting, run it yourself)'
