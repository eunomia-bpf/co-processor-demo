# Remote Power Management Guide

Complete guide for remote startup/shutdown of this development machine.

---

## Hardware Configuration

**Motherboard:** ASRock Z890M Riptide WiFi
**CPU:** Intel Core Ultra 9 285K
**Network Card:** Realtek Killer E3000 2.5GbE Controller (rev 06)
**MAC Address:** `9c:6b:00:84:c0:74`
**Network Interface:** `enp129s0`
**Current IP:** `128.114.59.195/24`
**Broadcast Address:** `128.114.59.255`

---

## Supported Features Summary

| Feature | Status | Notes |
|---------|--------|-------|
| Wake-on-LAN (WOL) | ✅ **SUPPORTED** | Already enabled on Ethernet |
| Intel AMT/vPro | ❌ Not available | Consumer CPU without vPro |
| IPMI/BMC | ❌ Not available | Consumer motherboard |
| WiFi WOL | ⚠️ Not reliable | Not recommended for desktop use |

**Current WOL Status:**
```
Supports Wake-on: pumbg
Wake-on: g  ← Magic Packet mode ENABLED
```

---

## BIOS/UEFI Configuration (REQUIRED)

Boot into BIOS and configure these settings:

### ✅ Enable These:
- `Wake on LAN` or `Wake on PCI-E/PCI`
- `Power on by PCIE Devices`
- `Resume by PCI or PCI-E Device`

### ❌ Disable These:
- `ErP Ready` / `ErP Support` → **Disabled** (kills standby power)
- `Deep Sleep` → **Disabled**
- `Energy Efficient` S5 states → **Disabled**

### Optional:
- `Restore on AC Power Loss` → **Power On** (enables smart plug workaround)

**Verification:** After shutdown, Ethernet port LED should show faint light/blink.

---

## Linux Network Configuration

### Option A: NetworkManager (Recommended)

```bash
# Check current WOL setting
nmcli c show "Wired connection 1" | grep wake

# Enable WOL persistently
nmcli c modify "Wired connection 1" 802-3-ethernet.wake-on-lan magic

# Apply changes
nmcli c up "Wired connection 1"

# Verify
sudo ethtool enp129s0 | grep Wake-on
```

### Option B: systemd Service

```bash
sudo tee /etc/systemd/system/wol@.service > /dev/null <<'EOF'
[Unit]
Description=Configure Wake-on-LAN for %i
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/ethtool -s %i wol g

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable wol@enp129s0.service
sudo systemctl start wol@enp129s0.service
```

---

## Local WOL Testing

### Install Tools

```bash
sudo apt install wakeonlan etherwake
```

### Send Wake Packet (from another machine on same LAN)

```bash
# Method 1: wakeonlan
wakeonlan 9c:6b:00:84:c0:74

# Method 2: etherwake
sudo etherwake 9c:6b:00:84:c0:74

# Method 3: magic-wol
sudo apt install magic-wol
wol 9c:6b:00:84:c0:74
```

### Test Procedure

1. Shut down computer: `sudo systemctl poweroff`
2. Wait 30 seconds
3. From another machine on same network, send WOL packet
4. Computer should power on within 2-5 seconds

---

## Remote Wake (From Outside Network)

Since this machine is on institutional network (`128.114.59.x`), use a jump host approach.

### Setup Jump Host

Set up a small always-on device (Raspberry Pi, old laptop, router) on same network segment:

```bash
# On jump host, create wake script
cat > ~/wake-dev-machine.sh <<'EOF'
#!/bin/bash
wakeonlan 9c:6b:00:84:c0:74
echo "Magic packet sent to 9c:6b:00:84:c0:74 at $(date)"
logger "WOL packet sent to dev-machine"
EOF

chmod +x ~/wake-dev-machine.sh
```

### Wake from Anywhere

```bash
ssh user@jump-host './wake-dev-machine.sh'
```

---

## Remote Shutdown/Reboot

### Basic Commands

```bash
# Immediate shutdown
sudo systemctl poweroff

# Immediate reboot
sudo systemctl reboot

# Scheduled shutdown (10 min warning)
sudo shutdown -h +10 "System going down for maintenance"

# Cancel scheduled shutdown
sudo shutdown -c
```

### Remote Shutdown via SSH

```bash
ssh user@128.114.59.195 'sudo systemctl poweroff'
```

### Passwordless Sudo for Power Management (Optional)

```bash
sudo visudo -f /etc/sudoers.d/poweroff

# Add this line (replace 'yourusername'):
yourusername ALL=(ALL) NOPASSWD: /usr/bin/systemctl poweroff, /usr/bin/systemctl reboot
```

---

## Troubleshooting Realtek Killer E3000

The Realtek 2.5GbE controllers can have WOL quirks. If WOL doesn't work:

### Disable Energy Efficient Ethernet

```bash
# Disable EEE
sudo ethtool --set-eee enp129s0 eee off

# Disable Green Ethernet
sudo ethtool -s enp129s0 advertise 0x0000000000008FFF

# Make persistent (add to NetworkManager connection)
nmcli c modify "Wired connection 1" ethtool.feature-eee off
```

### Check Driver

```bash
# Current driver
ethtool -i enp129s0
# Should show: driver: r8169 (kernel driver)

# If r8169 has issues, try official driver
sudo apt install r8168-dkms
sudo modprobe -r r8169
sudo modprobe r8168
```

### Common Issues

1. **WOL works after reboot but not after shutdown**
   - Check BIOS: ErP must be DISABLED
   - Verify 5VSB power: Ethernet LED should stay lit

2. **Wake-on setting resets after reboot**
   - Use NetworkManager persistent config (see above)
   - Or use systemd service

3. **WOL packet sent but machine doesn't wake**
   - Verify broadcast address: `ip addr show enp129s0`
   - Try direct IP instead of broadcast
   - Check switch/router doesn't filter Magic Packets

---

## Quick Reference Commands

### Check Current WOL Status
```bash
sudo ethtool enp129s0 | grep Wake
```

### Get MAC Address
```bash
cat /sys/class/net/enp129s0/address
```

### Get IP/Broadcast Address
```bash
ip addr show enp129s0
```

### Test Network Reachability
```bash
ping -c 3 128.114.59.195
```

### Monitor System Logs for WOL Events
```bash
sudo journalctl -f -u NetworkManager
dmesg | grep -i wake
```

---

## Action Checklist

- [ ] Enter BIOS and disable ErP/Deep Sleep, enable Wake-on-LAN
- [ ] Configure NetworkManager to persist WOL setting
- [ ] Test locally - shutdown and wake from another device on network
- [ ] Set up jump host if remote wake from outside is needed
- [ ] Configure SSH for convenient remote shutdown
- [ ] Test full cycle: remote wake → work → remote shutdown
- [ ] Document jump host IP and credentials securely

---

## Security Considerations

1. **WOL has no authentication** - anyone who can send packets to your MAC can wake machine
   - Mitigation: Use on trusted network only
   - Use VPN for remote access when possible

2. **SSH access for shutdown**
   - Use key-based authentication
   - Consider fail2ban for SSH protection
   - Limit sudo privileges to specific commands

3. **Jump host security**
   - Keep updated
   - Firewall properly
   - Monitor access logs

---

## Alternative: Smart Plug Method

If WOL fails or isn't available:

1. **BIOS setting:** `Restore on AC Power Loss` → **Power On**
2. **Hardware:** Connect PC to smart plug (TP-Link Kasa, Wemo, etc.)
3. **Wake process:**
   - Ensure machine is properly shut down (OS shutdown)
   - Use smart plug app to: Turn OFF → Wait 5s → Turn ON
   - Machine will auto-start due to "Power On" BIOS setting

**Warning:** Do NOT use smart plug to force shutdown (pulling power) - this risks file system corruption.

---

## System Information

**OS:** Linux 6.15.11-061511-generic
**Date Configured:** 2025-11-10
**Kernel Driver:** r8169
**Network Manager:** NetworkManager + systemd-networkd

**Related Files:**
- Network config: `/etc/NetworkManager/system-connections/`
- systemd service: `/etc/systemd/system/wol@.service`
- BIOS: ASRock Z890M Riptide WiFi UEFI

---

## Additional Resources

- [ASRock Z890M Riptide WiFi Manual](https://www.asrock.com/mb/Intel/Z890M%20Riptide%20WiFi/)
- [Realtek Ethernet Linux Driver](https://www.realtek.com/en/component/zoo/category/network-interface-controllers-10-100-1000m-gigabit-ethernet-pci-express-software)
- [Wake-on-LAN Wikipedia](https://en.wikipedia.org/wiki/Wake-on-LAN)
- [systemd Network Configuration](https://www.freedesktop.org/software/systemd/man/systemd.network.html)

---

**Last Updated:** 2025-11-10
**Maintained By:** System Administrator
