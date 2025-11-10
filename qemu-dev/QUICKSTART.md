# Quick Start Guide - QEMU Kernel Development with GPU

## Step-by-Step Setup

### 1. Initialize VM (2 minutes)
```bash
cd qemu-dev/scripts
./setup-vm.sh
```

### 2. Download OS ISO
Choose one:

**Ubuntu Server (Recommended):**
```bash
cd ../images
wget https://releases.ubuntu.com/22.04/ubuntu-22.04.3-live-server-amd64.iso
mv ubuntu-*.iso ubuntu.iso
```

**OR Debian:**
```bash
cd ../images
wget https://cdimage.debian.org/debian-cd/current/amd64/iso-cd/debian-12.2.0-amd64-netinst.iso
mv debian-*.iso debian.iso
```

### 3. Install OS (20-30 minutes)
```bash
cd ../scripts
./install-os.sh ../images/ubuntu.iso
```

**Installation tips:**
- Select "Install Ubuntu Server" (not live)
- Choose minimal installation
- Enable "Install OpenSSH server"
- Create user account
- Wait for installation to complete

### 4. First Boot (Test without GPU)
```bash
./start-vm.sh
```

**Inside the VM:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install development tools
sudo apt install -y build-essential git vim curl wget \
    linux-headers-$(uname -r) dkms

# Install NVIDIA drivers (for later GPU passthrough)
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Reboot
sudo reboot
```

### 5. Setup GPU Passthrough on Host
Exit the VM, then on the host:

```bash
sudo ./setup-gpu-passthrough.sh
# Answer 'y' when prompted to bind GPU to vfio-pci
```

### 6. Start VM with GPU
```bash
./start-vm.sh --gpu
```

**Verify GPU in VM:**
```bash
# SSH into VM
ssh -p 2222 user@localhost

# Check GPU
lspci | grep NVIDIA
nvidia-smi
```

## Kernel Development Quick Guide

### Get Kernel Source
```bash
cd ../kernel
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.8.tar.xz
tar xf linux-6.8.tar.xz
cd linux-6.8
```

### Make Your Changes
```bash
# Edit kernel source files
vim drivers/gpu/drm/...   # Example: modify GPU drivers
vim kernel/sched/...       # Example: modify scheduler
```

### Build Custom Kernel
```bash
cd ../../scripts
./build-kernel.sh ../kernel/linux-6.8 --use-host-config
```

### Test Custom Kernel
```bash
./start-vm.sh --gpu
# Your custom kernel will boot automatically
```

### Iterate
1. Make changes to kernel source
2. Rebuild: `./build-kernel.sh ../kernel/linux-6.8`
3. Test: `./start-vm.sh --gpu`
4. Repeat

## Common Commands

```bash
# Start VM without GPU
./start-vm.sh

# Start VM with GPU passthrough
./start-vm.sh --gpu

# Connect via SSH
ssh -p 2222 user@localhost

# Connect to QEMU monitor
./connect-vm.sh  # Select option 2

# Build kernel with custom config
./build-kernel.sh ../kernel/linux-6.8

# Setup GPU for passthrough
sudo ./setup-gpu-passthrough.sh

# Create VM snapshot
qemu-img snapshot -c snapshot-name ../images/kernel-dev.qcow2

# List snapshots
qemu-img snapshot -l ../images/kernel-dev.qcow2

# Restore snapshot
qemu-img snapshot -a snapshot-name ../images/kernel-dev.qcow2
```

## Typical Development Workflow

```
┌─────────────────────────────────────────────┐
│ 1. Boot VM with current kernel              │
│    ./start-vm.sh --gpu                      │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 2. Test current functionality                │
│    Run benchmarks, check GPU access         │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 3. Shutdown VM, modify kernel source        │
│    Edit files in kernel/linux-6.8/          │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 4. Rebuild kernel                            │
│    ./build-kernel.sh ../kernel/linux-6.8    │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 5. Boot VM with new kernel                  │
│    ./start-vm.sh --gpu                      │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 6. Test changes                              │
│    Run tests, check behavior                │
└─────────────────┬───────────────────────────┘
                  │
                  └──────► Repeat from step 3
```

## Pro Tips

1. **Take snapshots before major changes:**
   ```bash
   qemu-img snapshot -c before-gpu-driver-mod ../images/kernel-dev.qcow2
   ```

2. **Use SSH for easier access:**
   ```bash
   # Add to ~/.ssh/config
   Host qemu-dev
       HostName localhost
       Port 2222
       User youruser

   # Then just: ssh qemu-dev
   ```

3. **Mount host directory in VM (optional):**
   Add to start-vm.sh:
   ```bash
   -virtfs local,path=/root/co-processor-demo,mount_tag=host0,security_model=mapped
   ```

4. **Faster rebuilds:**
   Only rebuild changed modules instead of full kernel when possible

5. **Serial console for debugging:**
   The VM's serial output appears in the terminal where you ran start-vm.sh

## System Resources

Your host has:
- **GPU:** NVIDIA H100 (8a:00.0)
- **Kernel:** 6.8.0-87-generic
- **IOMMU:** Passthrough mode enabled

VM Configuration:
- **Memory:** 16GB (adjustable in start-vm.sh)
- **CPUs:** 8 cores (adjustable in start-vm.sh)
- **Disk:** 50GB (expandable)
- **Network:** User-mode with SSH forwarding (2222→22)

## Getting Help

- Full documentation: `../README.md`
- QEMU docs: https://www.qemu.org/docs/master/
- GPU passthrough: https://wiki.archlinux.org/title/PCI_passthrough_via_OVMF
- Kernel development: https://www.kernel.org/doc/html/latest/

## Troubleshooting

**VM won't start:**
- Check QEMU is installed: `which qemu-system-x86_64`
- Check disk exists: `ls -lh ../images/kernel-dev.qcow2`

**GPU not visible in VM:**
- Verify on host: `lspci -k -s 8a:00.0` (should show vfio-pci)
- Re-run: `sudo ./setup-gpu-passthrough.sh`

**Kernel build fails:**
- Install dependencies: `sudo apt install build-essential flex bison libssl-dev libelf-dev`
- Check disk space: `df -h`

**Can't SSH to VM:**
- VM must be booted and SSH server running
- Check forwarding: `./connect-vm.sh` → option 1
