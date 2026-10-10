# Local Network Setup

This guide explains how to run Dual GPU Studio on the Windows GPU machine and access its web interface from a Mac on the same local network.

The models, GPU orchestration, SQLite database, and `llama-server` processes remain on the Windows machine. The Mac only opens the web interface and sends chat requests to the Windows application.

## Network layout

```text
Mac web browser
    |
    |  Local network: TCP 8090
    v
Windows PC: Dual GPU Studio
    |
    +-- llama-server lane A: 127.0.0.1:8081
    +-- llama-server lane B: 127.0.0.1:8082
    +-- Local GPUs, models, logs, and SQLite run history
```

Only the Dual GPU Studio web port needs to be reachable from the Mac. The underlying model-server ports can remain bound to `127.0.0.1`.

## Requirements

- The Windows PC and Mac must be connected to the same trusted Ethernet or Wi-Fi network.
- The Windows network profile should be set to **Private**.
- Guest Wi-Fi or wireless client isolation must not block communication between devices.
- The configured LM Studio runtime and model files must be accessible on the Windows PC.
- The Windows PC must remain awake while the Mac is using the application.

## 1. Start the application for LAN access

Local-network listening is now enabled by default. From the project directory on Windows, run:

```powershell
python .\app.py --config .\example.dual_gpu.toml
```

The terminal prints the Windows-local address and one or more detected local-network addresses. Open a displayed **Local network** URL on the Mac.

The equivalent explicit command is:

```powershell
python .\app.py `
  --config .\example.dual_gpu.toml `
  --host 0.0.0.0 `
  --port 8090 `
  --no-browser
```

What these options mean:

- `--host 0.0.0.0` makes the web application listen on all Windows network interfaces.
- `--port 8090` exposes the web application on TCP port 8090.
- `--no-browser` prevents Windows from opening a local browser automatically.
- `--config` selects the GPU, model, and runtime configuration.

Keep this terminal open. Press `Ctrl+C` when you want to stop the application and unload model servers managed by it.

Passing `--host 127.0.0.1` intentionally disables LAN access. A Mac cannot connect while the application is bound only to `127.0.0.1`.

## 2. Find the Windows IP address

Run this on Windows:

```powershell
ipconfig
```

Find the active Ethernet or Wi-Fi adapter and note its **IPv4 Address**, for example:

```text
IPv4 Address. . . . . . . . . . . : 192.168.1.50
```

Ignore disconnected adapters and addresses beginning with `127.` or `169.254.`.

You can also use this PowerShell command to list likely LAN addresses:

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object {
    $_.IPAddress -notlike "127.*" -and
    $_.IPAddress -notlike "169.254.*"
  } |
  Select-Object InterfaceAlias, IPAddress
```

## 3. Allow the port through Windows Firewall

Open PowerShell **as Administrator** and create a rule restricted to private local subnets:

```powershell
New-NetFirewallRule `
  -DisplayName "Dual GPU Studio 8090" `
  -Direction Inbound `
  -Protocol TCP `
  -LocalPort 8090 `
  -Action Allow `
  -Profile Private `
  -RemoteAddress LocalSubnet
```

This rule:

- allows inbound TCP traffic only on port 8090;
- applies only when Windows considers the network Private;
- restricts access to local-subnet devices.

Check the rule:

```powershell
Get-NetFirewallRule -DisplayName "Dual GPU Studio 8090" |
  Get-NetFirewallPortFilter
```

If you later change the application port, create a matching firewall rule for the new port.

To remove the rule when LAN access is no longer required:

```powershell
Remove-NetFirewallRule -DisplayName "Dual GPU Studio 8090"
```

## 4. Connect from the Mac

Open Safari, Chrome, or Firefox on the Mac and visit:

```text
http://<windows-ip-address>:8090
```

For the example address above:

```text
http://192.168.1.50:8090
```

The Overview, Chat, and Run History pages should load. All run data continues to be stored on the Windows PC in:

```text
chat_runs/chatbot.sqlite3
```

Nothing is stored permanently on the Mac by the application, apart from normal browser data.

## 5. Verify the connection

First verify the application locally on Windows:

```text
http://127.0.0.1:8090
```

Then verify the LAN address from the Mac:

```text
http://<windows-ip-address>:8090
```

From macOS Terminal, you can test whether the web server responds:

```bash
curl http://192.168.1.50:8090/api/status
```

A successful response is JSON containing fields such as `state`, `mode`, `servers`, and `metrics`.

## Optional: use a stable address

The Windows IP address may change when the router renews its DHCP lease. For a stable bookmark, use one of these approaches:

- Configure a DHCP reservation for the Windows PC in the router.
- Assign a suitable static IP using your network administrator's addressing plan.
- Try the Windows computer name if local name resolution works, for example:

```text
http://windows-computer-name:8090
```

A router-side DHCP reservation is generally safer than manually choosing an arbitrary static address.

## Security warning

The current application has no built-in authentication and serves plain HTTP.

- Use it only on a trusted private network.
- Do not configure router port forwarding for port 8090.
- Do not expose the application directly to the internet.
- Do not use `-Profile Any` or `-RemoteAddress Any` unless you understand the exposure.
- Avoid untrusted public, office, hotel, or guest networks.
- Keep the model-server ports bound to `127.0.0.1`; expose only the web application port.
- Stop the application when it is not required.

For access from outside the home/local network, use a secure VPN such as a private WireGuard or Tailscale network and add application authentication before considering broader exposure.

## Troubleshooting

### The page works on Windows but not on the Mac

Check that:

- the service was started with `--host 0.0.0.0` and the terminal shows a `Local network:` URL (the secure default bind is `127.0.0.1`);
- the Mac is using the current Windows IPv4 address;
- both machines are connected to the same network;
- the Windows network profile is Private;
- the firewall rule exists and uses the same port as the app;
- a VPN on either machine is not routing or filtering local traffic;
- the Wi-Fi access point does not enable client/AP isolation.

### Confirm that Windows is listening

Run on Windows:

```powershell
Get-NetTCPConnection -LocalPort 8090 -State Listen
```

The local address should normally appear as `0.0.0.0` when LAN listening is enabled.

### Test basic connectivity from the Mac

Run:

```bash
ping 192.168.1.50
```

Some Windows firewall configurations block ping even when TCP port 8090 works, so also test with `curl`:

```bash
curl -v http://192.168.1.50:8090/api/status
```

### The Windows address keeps changing

Create a DHCP reservation in the router for the Windows PC and update the Mac bookmark to use that reserved address.

### The UI loads but models do not start

LAN connectivity is working in this case. Check the Windows application terminal and confirm:

- model files exist at the configured paths;
- the LM Studio runtime can be discovered;
- GPU device matching is correct;
- lane ports are not occupied by unrelated applications.

You can validate device discovery on Windows with:

```powershell
dual-gpu --config .\example.dual_gpu.toml inspect
```

## Recommended launch command

For normal trusted-LAN use:

```powershell
python .\app.py `
  --config .\example.dual_gpu.toml `
  --host 0.0.0.0 `
  --port 8090 `
  --data-dir .\chat_runs `
  --no-browser
```

Then open the following address on the Mac:

```text
http://<windows-ip-address>:8090
```
