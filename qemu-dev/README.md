# QEMU Development Environment with GPU Passthrough

This directory contains scripts and tools for setting up a safe kernel development environment using QEMU with NVIDIA H100 GPU passthrough.

## Overview

This setup allows you to:
- Safely develop and test kernel modifications without risking your host system
- Access the H100 GPU from within the VM via PCI passthrough
- Build and boot custom kernels quickly
- Maintain isolated development environment

## Directory Structure

```
qemu-dev/
├── scripts/           # Helper scripts
│   ├── setup-vm.sh              # Initial VM setup
│   ├── setup-gpu-passthrough.sh # Configure GPU for passthrough
│   ├── start-vm.sh              # Start the VM (with/without GPU)
│   ├── install-os.sh            # Install OS in VM
│   ├── build-kernel.sh          # Build custom kernel
│   └── connect-vm.sh            # Connect to running VM
├── images/            # VM disk images
├── kernel/            # Custom kernel builds
└── README.md          # This file
```

## Quick Start

### 1. Initial Setup

```bash
cd scripts
chmod +x *.sh
./setup-vm.sh
```

### 2. Install Operating System

Download an Ubuntu Server or Debian ISO, then:

```bash
./install-os.sh /path/to/ubuntu-server.iso
```

Follow the on-screen installer. Recommended settings:
- Enable SSH server during installation
- Create a user account
- Install minimal system

### 3. Start VM (without GPU first)

```bash
./start-vm.sh
```

Configure the guest OS:
- Install necessary packages (build-essential, nvidia drivers, etc.)
- Set up SSH keys
- Configure networking

### 4. Setup GPU Passthrough

On the **host system**, configure the GPU for passthrough:

```bash
sudo ./setup-gpu-passthrough.sh
```

This will:
- Load VFIO kernel modules
- Unbind the GPU from the nvidia driver
- Bind it to vfio-pci for passthrough

### 5. Start VM with GPU

```bash
./start-vm.sh --gpu
```

Inside the VM, the H100 GPU will be available as a PCI device.

## Kernel Development Workflow

### Build a Custom Kernel

1. Download kernel source:
```bash
cd ../kernel
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.8.tar.xz
tar xf linux-6.8.tar.xz
```

2. Build the kernel:
```bash
cd ../scripts
./build-kernel.sh ../kernel/linux-6.8 --use-host-config
```

3. Make your modifications to the kernel source and rebuild

4. Start VM with custom kernel:
```bash
./start-vm.sh --gpu
```

The custom kernel will be automatically loaded.

## GPU Configuration Details

### Host System Requirements

Your system has:
- GPU: NVIDIA H100 (Bus: 8a:00.0)
- Current kernel: 6.8.0-87-generic
- IOMMU mode: passthrough (iommu=pt)

### IOMMU Configuration

Current kernel parameters show `iommu=pt` and `intel_iommu=off`. For full GPU passthrough, you may need to:

1. Enable IOMMU in BIOS/UEFI
2. Modify kernel parameters in `/etc/default/grub`:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on iommu=pt"
```
3. Update grub: `sudo update-grub`
4. Reboot

### Making GPU Passthrough Persistent

To automatically bind the GPU to vfio-pci on boot:

1. Create `/etc/modprobe.d/vfio.conf`:
```
options vfio-pci ids=10de:2336
```

2. Add to `/etc/modules`:
```
vfio
vfio_iommu_type1
vfio_pci
```

3. Update initramfs: `sudo update-initramfs -u`

## Troubleshooting

### VM won't start
- Check that KVM is available: `lsmod | grep kvm`
- Verify disk image exists: `ls -lh images/`

### GPU passthrough not working
- Verify GPU is bound to vfio-pci: `lspci -k -s 8a:00.0`
- Check IOMMU groups: `find /sys/kernel/iommu_groups/ -type l`
- Review dmesg for IOMMU errors: `dmesg | grep -i iommu`

### Custom kernel won't boot
- Check kernel build logs for errors
- Ensure required drivers are enabled (VIRTIO, VFIO)
- Try with host kernel config: `./build-kernel.sh <source> --use-host-config`

### Cannot access VM
- SSH: VM must be running and SSH server configured
  - Connect: `ssh -p 2222 user@localhost`
- Serial console: Check terminal where you ran start-vm.sh
- QEMU Monitor: `telnet localhost 5555`

## Advanced Usage

### Snapshot Management

Create a snapshot before making changes:
```bash
qemu-img snapshot -c before-changes images/kernel-dev.qcow2
```

Restore from snapshot:
```bash
qemu-img snapshot -a before-changes images/kernel-dev.qcow2
```

### Network Configuration

The VM uses user-mode networking with port forwarding:
- Host port 2222 → VM port 22 (SSH)

To add more port forwards, edit `start-vm.sh` and add to the netdev line:
```bash
-netdev user,id=net0,hostfwd=tcp::2222-:22,hostfwd=tcp::8080-:80
```

### Memory and CPU Tuning

Edit the variables in `start-vm.sh`:
```bash
VM_MEMORY="16G"  # Adjust based on your needs
VM_CPUS="8"      # Number of CPU cores
```

## Safety Notes

- The VM is isolated from your host system
- Kernel crashes in the VM won't affect the host
- GPU passthrough gives the VM direct hardware access
- Always test kernel changes in the VM before applying to host
- Keep snapshots before major changes

## Resources

- QEMU Documentation: https://www.qemu.org/docs/master/
- VFIO GPU Passthrough: https://wiki.archlinux.org/title/PCI_passthrough_via_OVMF
- Linux Kernel Development: https://www.kernel.org/doc/html/latest/
