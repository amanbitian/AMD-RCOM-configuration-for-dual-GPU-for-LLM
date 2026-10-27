# Local Network Setup

This guide explains how to run Dual GPU Studio on a Windows or Ubuntu GPU host and access its web interface from another device on the same local network.

The models, GPU orchestration, SQLite database, and `llama-server` processes remain on the GPU host. The client device only opens the web interface and sends requests to that host.

## Network layout

```text
Client web browser
    |
    |  Local network: TCP 8090
    v
Windows or Ubuntu GPU host: Dual GPU Studio
    |
    +-- llama-server lane A: 127.0.0.1:8081
    +-- llama-server lane B: 127.0.0.1:8082
    +-- Local GPUs, models, logs, and SQLite run history
```

Only the Dual GPU Studio web port needs to be reachable from the client. The underlying model-server ports can remain bound to `127.0.0.1`.

## Requirements

- The GPU host and client must be connected to the same trusted Ethernet or Wi-Fi network.
- On Windows, the network profile should be set to **Private**.
- Guest Wi-Fi or wireless client isolation must not block communication between devices.
- The configured `llama-server` runtime and model files must be accessible on the GPU host. Ubuntu should use the native llama.cpp binary configured in `backend_server_binaries`; Windows may also use the LM Studio fallback.
- The GPU host must remain awake while the client is using the application.

## 1. Start the application for LAN access

From the project directory on Windows, run:

```powershell
python .\app.py --config .\example.dual_gpu.toml --host 0.0.0.0
```

On Ubuntu, use the edited native llama.cpp config:

```bash
python3 app.py --config ./dual_gpu.toml --host 0.0.0.0 --port 8090 --no-browser
```

The terminal prints the local address and one or more detected local-network addresses. Open a displayed **Local network** URL on the client.

The equivalent explicit command is:

```powershell
python .\app.py `
  --config .\example.dual_gpu.toml `
  --host 0.0.0.0 `
  --port 8090 `
  --no-browser
```

What these options mean:

- `--host 0.0.0.0` makes the web application listen on all host network interfaces.
- `--port 8090` exposes the web application on TCP port 8090.
- `--no-browser` prevents the host from opening a local browser automatically.
- `--config` selects the GPU, model, and runtime configuration.

Keep this terminal open. Press `Ctrl+C` when you want to stop the application and unload model servers managed by it.

If the service was started in the background with `py launch_service.py`, there is no terminal to close. Use
the dashboard's **Service** page from the client: **Restart service** or **Stop service**, **Unload** for each GPU,
and the settings. Starting it again after **Stop service** requires access to the GPU host
(`py launch_service.py --host 0.0.0.0` on Windows or
`python3 launch_service.py --config ./dual_gpu.toml --host 0.0.0.0` on Ubuntu).
Anyone who can open the UI can use these controls; see the security notes in
[app_overview.md](app_overview.md#12-security-posture).

Passing `--host 127.0.0.1` intentionally disables LAN access; another device cannot connect while the
application is bound only to loopback.

## 2. Find the host IP address

On Windows:

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

On Ubuntu:

```bash
ip -brief address
```

Use the non-loopback IPv4 address on the active Ethernet or Wi-Fi interface.

## 3. Allow the port through the host firewall

### Windows Firewall

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

### Ubuntu UFW

If UFW is active, allow only the trusted local subnet (replace the example CIDR if needed):

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8090 proto tcp
sudo ufw status
```

Remove that rule later with the matching `ufw delete allow ...` command. Do not open port 8090
globally on an internet-facing interface.

## 4. Connect from the Mac

Open Safari, Chrome, or Firefox on the Mac and visit:

```text
http://<host-ip-address>:8090
```

For the example address above:

```text
http://192.168.1.50:8090
```

The Overview, Chat, and Run History pages should load. All run data continues to be stored on the GPU host in:

```text
chat_runs/chatbot.sqlite3
```

Nothing is stored permanently on the client by the application, apart from normal browser data.

## 5. Verify the connection

First verify the application locally on the GPU host:

```text
http://127.0.0.1:8090
```

Then verify the LAN address from the client:

```text
http://<host-ip-address>:8090
```

From the client terminal, you can test whether the web server responds:

```bash
curl http://192.168.1.50:8090/api/status
```

A successful response is JSON containing fields such as `state`, `mode`, `servers`, and `metrics`.

## Optional: use a stable address

The host IP address may change when the router renews its DHCP lease. For a stable bookmark, use one of these approaches:

- Configure a DHCP reservation for the GPU host in the router.
- Assign a suitable static IP using your network administrator's addressing plan.
- Try the host name if local name resolution works, for example:

```text
http://gpu-host-name:8090
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

### The page works on the GPU host but not on the client

Check that:

- the service was started with `--host 0.0.0.0` and the terminal shows a `Local network:` URL (the secure default bind is `127.0.0.1`);
- the client is using the current host IPv4 address;
- both machines are connected to the same network;
- the Windows network profile is Private, or the Ubuntu firewall has a matching local-subnet rule;
- the firewall rule exists and uses the same port as the app;
- a VPN on either machine is not routing or filtering local traffic;
- the Wi-Fi access point does not enable client/AP isolation.

### Confirm that the host is listening

Run on Windows:

```powershell
Get-NetTCPConnection -LocalPort 8090 -State Listen
```

The local address should normally appear as `0.0.0.0` when LAN listening is enabled.

On Ubuntu, use:

```bash
ss -ltnp | grep ':8090'
```

### Test basic connectivity from the client

Run:

```bash
ping 192.168.1.50
```

Some firewall configurations block ping even when TCP port 8090 works, so also test with `curl`:

```bash
curl -v http://192.168.1.50:8090/api/status
```

### The host address keeps changing

Create a DHCP reservation in the router for the GPU host and update the client bookmark to use that reserved address.

### The UI loads but models do not start

LAN connectivity is working in this case. Check the application log or host terminal and confirm:

- model files exist at the configured paths;
- the native `llama-server` configured on Ubuntu (or the configured/LM Studio build on Windows) can be found and the model fits the free VRAM (a `Not enough free VRAM` error means another application holds the GPU);
- GPU device matching is correct;
- lane ports are not occupied by unrelated applications.

Validate device discovery with the config for that host:

```powershell
dual-gpu --config .\example.dual_gpu.toml inspect
```

```bash
dual-gpu --config ./dual_gpu.toml inspect
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
http://<host-ip-address>:8090
```

The Ubuntu equivalent is:

```bash
python3 app.py --config ./dual_gpu.toml --host 0.0.0.0 --port 8090 --data-dir ./chat_runs --no-browser
```
