# QEMU Development Environment - Setup Complete

## What Has Been Created

### Directory Structure
```
/root/co-processor-demo/qemu-dev/
├── scripts/
│   ├── setup-vm.sh                 # Initial VM setup
│   ├── setup-gpu-passthrough.sh    # GPU passthrough configuration
│   ├── start-vm.sh                 # Start VM without GPU
│   ├── start-with-gpu.sh           # Start VM WITH H100 GPU ⭐
│   ├── enable-iommu.sh             # Enable IOMMU (required for GPU) ⭐
│   ├── install-os.sh               # Manual OS installation
│   ├── build-kernel.sh             # Build custom kernels
│   └── connect-vm.sh               # Connect to running VM
├── images/
│   ├── kernel-dev.qcow2            # VM disk (Ubuntu 24.10, 50GB)
│   ├── cloud-init.iso              # Cloud-init configuration
│   └── ubuntu-24.10-cloudimg.img   # Original cloud image backup
├── kernel/                         # For custom kernel builds
├── README.md                       # Full documentation
├── QUICKSTART.md                   # Quick start guide
├── GPU-PASSTHROUGH-SETUP.md        # GPU setup instructions ⭐
└── SETUP-COMPLETE.md               # This file
```

### VM Configuration
- **OS:** Ubuntu 24.10 (Oracular)
- **Kernel:** 6.11.0-29-generic
- **Memory:** 16GB
- **CPUs:** 8 cores
- **Disk:** 50GB (expandable with `qemu-img resize`)
- **User:** ubuntu
- **Password:** ubuntu
- **SSH Port:** 2222 (forwards to VM port 22)

### GPU Information
- **Model:** NVIDIA H100
- **PCI Bus:** 8a:00.0
- **Vendor ID:** 10de
- **Device ID:** 2336

## Current Status

✅ VM image created and configured
✅ Cloud-init setup complete
✅ Scripts created for GPU passthrough
✅ Documentation complete

⚠️ **IOMMU is currently DISABLED** - GPU passthrough will NOT work until you enable it

## To Enable GPU Passthrough

### Step 1: Enable IOMMU (Requires Reboot)

```bash
cd /root/co-processor-demo/qemu-dev/scripts
sudo ./enable-iommu.sh
```

This script will:
1. Backup your GRUB configuration
2. Change `intel_iommu=off` to `intel_iommu=on`
3. Update GRUB
4. Prompt you to reboot

### Step 2: After Reboot

Verify IOMMU is enabled:
```bash
# Should show IOMMU initialization messages
dmesg | grep -i iommu | head -20

# Should list IOMMU groups
ls /sys/kernel/iommu_groups/
```

### Step 3: Start VM with GPU

```bash
cd /root/co-processor-demo/qemu-dev/scripts
./start-with-gpu.sh
```

### Step 4: Access VM and Install NVIDIA Drivers

```bash
# SSH into VM (password: ubuntu)
ssh -p 2222 ubuntu@localhost

# Inside VM - check if GPU is visible
lspci | grep -i nvidia

# Update package list
sudo apt update

# Install NVIDIA driver
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Install CUDA toolkit
sudo apt install -y nvidia-cuda-toolkit

# Reboot VM
sudo reboot
```

### Step 5: Test CUDA

After VM reboots, SSH back in:
```bash
ssh -p 2222 ubuntu@localhost

# Test GPU
nvidia-smi

# Check CUDA
nvcc --version

# Run a CUDA sample (if available)
cuda-install-samples-11.8.sh ~
cd ~/NVIDIA_CUDA-11.8_Samples/1_Utilities/deviceQuery
make
./deviceQuery
```

## Alternative: Test CUDA on Host (No Reboot Needed)

If you want to test CUDA immediately without enabling GPU passthrough:

```bash
# Test on host system directly
nvidia-smi
nvcc --version

# Your host already has CUDA installed
```

## Troubleshooting

### VM won't start
```bash
# Check if VM is already running
ps aux | grep qemu

# Kill existing VM
pkill -f qemu-system

# Check logs
tail -f /tmp/qemu-gpu.log
```

### Can't SSH into VM
```bash
# Wait longer (VM takes 60-90 seconds to boot)
sleep 90
ssh -p 2222 ubuntu@localhost

# Check if SSH service started
tail -f /tmp/qemu-gpu.log | grep ssh
```

### GPU not visible in VM
```bash
# On host: verify GPU is bound to vfio-pci
lspci -k -s 8a:00.0

# Should show: "Kernel driver in use: vfio-pci"
# If not, IOMMU may not be enabled
```

### IOMMU errors
```bash
# Verify IOMMU is enabled in kernel
cat /proc/cmdline | grep iommu

# Should show: intel_iommu=on iommu=pt
```

## Key Commands Reference

```bash
# Start VM with GPU
cd /root/co-processor-demo/qemu-dev/scripts
./start-with-gpu.sh

# Start VM without GPU
./start-vm.sh

# Stop VM
pkill -f qemu-system

# SSH to VM
ssh -p 2222 ubuntu@localhost

# Monitor VM console
tail -f /tmp/qemu-gpu.log

# Check GPU status on host
lspci -k -s 8a:00.0

# Check IOMMU groups
ls -la /sys/kernel/iommu_groups/

# Enable IOMMU (needs reboot)
sudo ./enable-iommu.sh
```

## Files and Logs

- **VM Disk:** `/root/co-processor-demo/qemu-dev/images/kernel-dev.qcow2`
- **Boot Log:** `/tmp/qemu-gpu.log`
- **GRUB Backup:** `/etc/default/grub.backup.*`
- **Scripts:** `/root/co-processor-demo/qemu-dev/scripts/`

## Next Steps

1. **For GPU Passthrough:** Run `sudo ./enable-iommu.sh` and reboot
2. **For Kernel Development:** Start modifying kernels and test in VM
3. **For CUDA Testing:** Install drivers in VM after enabling GPU passthrough

## Documentation

- **GPU-PASSTHROUGH-SETUP.md** - Detailed GPU passthrough guide
- **README.md** - Complete QEMU environment documentation
- **QUICKSTART.md** - Quick start walkthrough

## Summary

Everything is set up and ready! The only thing preventing GPU passthrough from working is that IOMMU is disabled in your kernel parameters.

**To make GPU work:**
1. Run `sudo ./enable-iommu.sh`
2. Reboot
3. Run `./start-with-gpu.sh`
4. SSH in and install NVIDIA drivers

Your QEMU development environment is fully configured and ready to use for safe kernel development with GPU access!
