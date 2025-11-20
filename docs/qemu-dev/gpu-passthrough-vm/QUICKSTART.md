# Quick Start Guide - GPU Passthrough VM

Fast guide to get Ubuntu 24.04 VM running with NVIDIA RTX 5090 passthrough.

## Current Status

**Host Configuration:**
- ✓ Intel GPU running host desktop (X.org)
- ✓ NVIDIA RTX 5090 currently used by host (nvidia driver)
- ✓ IOMMU enabled (25 groups)
- ✓ KVM/QEMU installed and ready

**Next Steps:** Configure GPU for passthrough, create VM, launch

---

## Step-by-Step Instructions

### Step 1: Configure GPU for VFIO Passthrough (5 minutes)

This binds the NVIDIA GPU to vfio-pci driver so it can be passed to VMs.

```bash
cd /home/yunwei37/workspace/playground/co-processor-demo/qemu-dev/gpu-passthrough-vm

# Configure NVIDIA GPU for passthrough
sudo ./setup-vfio.sh

# Reboot (required!)
sudo reboot
```

**What this does:**
- Creates VFIO configuration to bind GPU at boot
- Blacklists NVIDIA host drivers
- Updates initramfs

**After reboot:**
- NVIDIA GPU will be bound to vfio-pci (not nvidia)
- Host will continue using Intel GPU for display
- GPU will be ready for VM passthrough

---

### Step 2: Verify VFIO Configuration (1 minute)

After reboot, check that GPU is bound to vfio-pci:

```bash
cd /home/yunwei37/workspace/playground/co-processor-demo/qemu-dev/gpu-passthrough-vm

./check-status.sh
```

**Expected output:**
```
4. GPU Driver Binding:
   ✓ GPU is bound to vfio-pci (READY FOR PASSTHROUGH)
```

If you see "nvidia driver" instead, troubleshoot with `TROUBLESHOOTING.md`.

---

### Step 3: Create VM Disk and Download Ubuntu (30-60 minutes)

Creates a 50GB virtual disk and downloads Ubuntu 24.04 ISO (~6GB).

```bash
./create-vm-disk.sh
```

**What this does:**
- Creates ubuntu2404-gpu.qcow2 (50GB disk image)
- Downloads Ubuntu 24.04 desktop ISO
- Sets up OVMF UEFI firmware variables

**This will take time** because it downloads ~6GB Ubuntu ISO.

---

### Step 4: Launch VM with GPU Passthrough (2 minutes)

Start the VM. First boot will install Ubuntu from ISO.

```bash
sudo ./start-vm.sh
```

**What this does:**
- Starts QEMU with NVIDIA GPU passed through
- Boots from Ubuntu ISO (first time)
- Provides VNC access on localhost:5900

**VM Configuration:**
- 16GB RAM
- 8 CPU cores
- NVIDIA RTX 5090 fully passed through
- VirtIO disk and network for performance

---

### Step 5: Connect to VM and Install Ubuntu (20-30 minutes)

**Connect via VNC:**

```bash
# Install VNC viewer if needed
sudo apt install tigervnc-viewer

# Connect to VM
vncviewer localhost:5900
```

**VNC Password:** `gpu-vm-pass`

**Or use Remmina:**
```bash
remmina -c vnc://localhost:5900
```

**Install Ubuntu:**
1. Follow Ubuntu installer prompts
2. Choose "Normal installation"
3. Create user account
4. Wait for installation to complete (~15-20 minutes)
5. Reboot when prompted

---

### Step 6: Install NVIDIA Drivers in VM (10 minutes)

After Ubuntu installation completes and VM reboots:

```bash
# Inside the VM (via VNC):

# Update package lists
sudo apt update

# Install NVIDIA drivers
sudo ubuntu-drivers install

# Reboot VM
sudo reboot
```

**After reboot, verify GPU:**

```bash
# Inside VM:
nvidia-smi
```

You should see your RTX 5090 with full specs!

---

### Step 7: Enjoy Your VM!

Your VM now has:
- ✓ Native GPU performance (~98% of bare metal)
- ✓ Full CUDA support
- ✓ OpenGL/Vulkan acceleration
- ✓ All 32GB VRAM available

**Use cases:**
- Machine learning with CUDA
- GPU rendering (Blender, etc.)
- GPU-accelerated development
- Testing GPU workloads

---

## Managing the VM

### Stop VM

```bash
# Graceful shutdown
./stop-vm.sh

# Or from inside VM
sudo poweroff
```

### Start VM (subsequent boots)

```bash
sudo ./start-vm.sh
```

(No ISO needed after installation - boots from disk)

### Check VM Status

```bash
# See if VM is running
ps aux | grep qemu

# Check PID file
cat /tmp/qemu-ubuntu2404-gpu.pid
```

### View VM Logs

```bash
ls -lh logs/
tail -f logs/vm-*.log
```

---

## Reverting to Host GPU Use

If you want to use NVIDIA GPU on host again (undo passthrough):

```bash
sudo ./revert-gpu-to-host.sh
sudo reboot
```

After reboot:
- NVIDIA GPU returns to host
- `nvidia-smi` works on host
- Cannot run GPU passthrough VM

To enable passthrough again: `sudo ./setup-vfio.sh && sudo reboot`

---

## Troubleshooting

**VM won't start?**
- Check `./check-status.sh` - GPU must be on vfio-pci
- See `TROUBLESHOOTING.md` for detailed solutions

**Black screen in VNC?**
- Wait 30-60 seconds for UEFI to initialize
- GPU might take time to display

**No network in VM?**
- VM uses user networking (NAT)
- Should get IP automatically via DHCP

**Performance issues?**
- Check CPU/RAM allocation in `config.sh`
- Consider enabling huge pages (see TROUBLESHOOTING.md)

**Cannot connect VNC?**
- Check VM is running: `ps aux | grep qemu`
- Try: `vncviewer localhost:5900`

---

## Configuration Files

Edit `config.sh` to customize:

```bash
VM_MEMORY="16G"        # Change RAM allocation
VM_CPU_CORES="8"       # Change CPU cores
VM_DISK_SIZE="50G"     # Change disk size (before creating)
VNC_PASSWORD="..."     # Change VNC password
```

After editing, recreate VM or restart with new settings.

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Host System (Intel GPU - i915)        │
│  - Desktop running normally             │
│  - SSH, development tools, etc.        │
└─────────────────────────────────────────┘
                  │
                  │ VFIO Passthrough
                  ▼
┌─────────────────────────────────────────┐
│  Ubuntu 24.04 VM (NVIDIA RTX 5090)     │
│  - Full GPU access (32GB VRAM)         │
│  - CUDA, OpenGL, Vulkan                │
│  - ~98% native performance             │
│  - Isolated from host                  │
└─────────────────────────────────────────┘
```

---

## Summary of Commands

```bash
# Initial setup (once)
sudo ./setup-vfio.sh && sudo reboot

# Check status
./check-status.sh

# Create VM disk (once)
./create-vm-disk.sh

# Start VM
sudo ./start-vm.sh

# Connect to VM
vncviewer localhost:5900

# Stop VM
./stop-vm.sh

# Revert to host GPU
sudo ./revert-gpu-to-host.sh && sudo reboot
```

---

## Performance Tips

1. **Pin CPU cores** for lower latency (edit start-vm.sh)
2. **Enable huge pages** for better memory performance
3. **Use SSD** for VM disk storage
4. **Allocate sufficient RAM** (16GB recommended minimum)
5. **Use VirtIO drivers** (already configured)

See `TROUBLESHOOTING.md` for detailed performance tuning.

---

## Next Steps After Setup

Once VM is running with GPU:

1. **Install development tools:**
   ```bash
   sudo apt install build-essential cmake git
   ```

2. **Install CUDA toolkit:**
   ```bash
   sudo apt install nvidia-cuda-toolkit
   ```

3. **Test GPU performance:**
   ```bash
   nvidia-smi
   glxinfo | grep OpenGL
   vulkaninfo | grep deviceName
   ```

4. **Run your GPU workloads!**

---

## Need Help?

1. Check `TROUBLESHOOTING.md` for common issues
2. Check logs: `tail -f logs/vm-*.log`
3. Run `./check-status.sh` to diagnose configuration
4. Check QEMU command: `cat logs/last-qemu-command.txt`

---

**Estimated Total Time:** ~1-2 hours (mostly waiting for downloads and installation)

**Result:** Ubuntu 24.04 VM with native GPU performance, while host continues using Intel GPU for display.

Enjoy your GPU passthrough VM! 🚀
