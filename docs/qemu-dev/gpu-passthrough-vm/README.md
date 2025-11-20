# QEMU GPU Passthrough VM Setup

This directory contains scripts to run Ubuntu 24.04 VM with NVIDIA RTX 5090 GPU passthrough.

## Hardware Configuration

- **Host GPU (Intel Arrow Lake):** Runs host desktop
- **Passthrough GPU (NVIDIA RTX 5090):** Passed to VM
- **VM OS:** Ubuntu 24.04 LTS

## Files

- `setup-vfio.sh` - Configure NVIDIA GPU for VFIO passthrough (run once)
- `create-vm-disk.sh` - Create Ubuntu 24.04 VM disk image
- `start-vm.sh` - Launch VM with GPU passthrough
- `revert-gpu-to-host.sh` - Return NVIDIA GPU to host (undo VFIO)
- `config.sh` - Configuration variables
- `check-status.sh` - Check GPU and VFIO status

## Quick Start

### 1. Initial Setup (One-time)

```bash
# Configure NVIDIA GPU for VFIO passthrough
sudo ./setup-vfio.sh

# Reboot required
sudo reboot
```

### 2. Create VM Disk

```bash
# Download Ubuntu 24.04 and create VM disk (50GB)
./create-vm-disk.sh
```

### 3. Launch VM

```bash
# Start VM with GPU passthrough
sudo ./start-vm.sh
```

### 4. Access VM

- **VNC:** Connect to `localhost:5900` (password: `gpu-vm-pass`)
- **SSH:** After installation, VM will be on bridged network

### 5. Install NVIDIA Drivers in VM

Once Ubuntu is installed in the VM:

```bash
# Inside VM:
sudo apt update
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

## Revert to Host GPU Usage

To use NVIDIA GPU on host again:

```bash
sudo ./revert-gpu-to-host.sh
sudo reboot
```

## Status Check

Check current GPU driver bindings:

```bash
./check-status.sh
```

## Troubleshooting

See `TROUBLESHOOTING.md` for common issues and solutions.

## Notes

- VM requires ~16GB RAM (configurable in `config.sh`)
- ~8 CPU cores allocated (configurable)
- UEFI boot with OVMF firmware
- VirtIO drivers for best performance
