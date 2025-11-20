# H100 GPU Passthrough Setup for QEMU

## Current Status

Your system has:
- **GPU:** NVIDIA H100 (PCI bus 8a:00.0)
- **Kernel:** 6.8.0-87-generic
- **Issue:** IOMMU is disabled (`intel_iommu=off` in kernel parameters)

## Problem

QEMU error: `vfio 0000:8a:00.0: no iommu_group found`

This happens because:
1. Current kernel cmdline has `intel_iommu=off`
2. VFIO requires IOMMU to be enabled for device isolation

## Solution

### Option 1: Enable IOMMU (Requires Reboot)

1. **Edit GRUB configuration:**
   ```bash
   sudo nano /etc/default/grub
   ```

2. **Modify GRUB_CMDLINE_LINUX_DEFAULT:**
   Change from:
   ```
   GRUB_CMDLINE_LINUX_DEFAULT="... nvidia_drm.modeset=1 iommu=pt intel_iommu=off ..."
   ```

   To:
   ```
   GRUB_CMDLINE_LINUX_DEFAULT="... nvidia_drm.modeset=1 intel_iommu=on iommu=pt ..."
   ```

3. **Update GRUB and reboot:**
   ```bash
   sudo update-grub
   sudo reboot
   ```

4. **After reboot, verify IOMMU:**
   ```bash
   dmesg | grep -i iommu
   ls /sys/kernel/iommu_groups/
   ```

5. **Run the GPU passthrough script:**
   ```bash
   cd /root/co-processor-demo/qemu-dev/scripts
   ./start-with-gpu.sh
   ```

### Option 2: Use GPU in Host for CUDA Testing (No Reboot)

If you want to test CUDA immediately without rebooting:

1. **Stop the VM:**
   ```bash
   pkill -f qemu-system
   ```

2. **Test CUDA on the host directly:**
   ```bash
   nvidia-smi
   nvcc --version
   ```

3. **Run a CUDA sample:**
   ```bash
   cd /usr/local/cuda/samples/1_Utilities/deviceQuery
   make
   ./deviceQuery
   ```

### Option 3: Use vGPU or SR-IOV (Advanced)

Some NVIDIA GPUs support SR-IOV for virtualization without full passthrough.
Check if H100 supports this for your use case.

## Quick Test Script

Created: `/root/co-processor-demo/qemu-dev/scripts/test-cuda-host.sh`

This script tests CUDA on the host system directly.

## Files Created

1. **start-with-gpu.sh** - Automated VM startup with GPU passthrough (requires IOMMU)
2. **setup-gpu-passthrough.sh** - Manual GPU passthrough configuration
3. **test-cuda-host.sh** - Test CUDA on host (no VM)

## VM Details

- **Image:** Ubuntu 24.10 (kernel 6.11.0)
- **Size:** 50GB (expandable)
- **RAM:** 16GB
- **CPUs:** 8 cores
- **SSH:** `ssh -p 2222 ubuntu@localhost` (password: ubuntu)
- **Location:** `/root/co-processor-demo/qemu-dev/images/`

## Verification Commands

```bash
# Check current IOMMU status
cat /proc/cmdline | grep iommu

# List IOMMU groups
ls -la /sys/kernel/iommu_groups/

# Check GPU driver
lspci -k -s 8a:00.0

# Check VFIO devices
ls -la /dev/vfio/
```

## Recommended Next Steps

**For GPU Passthrough (Safest for kernel development):**
1. Enable IOMMU as shown in Option 1
2. Reboot
3. Run `./start-with-gpu.sh`
4. SSH into VM and install NVIDIA drivers
5. Test CUDA in isolated environment

**For Quick CUDA Testing (No reboot):**
1. Use the host system directly
2. Test with existing CUDA installation
3. Use QEMU later for kernel modifications

## Support

If you need help with:
- **BIOS settings:** Ensure VT-d (Intel) or AMD-Vi is enabled
- **Kernel updates:** May require different IOMMU settings
- **GPU conflicts:** Ensure no other processes are using the GPU

Current kernel parameters:
```
BOOT_IMAGE=/boot/vmlinuz-6.8.0-87-generic root=UUID=23bccd26-1063-4609-846a-c7ec4b800f0b ro nvidia_drm.modeset=1 iommu=pt intel_iommu=off no5lvl
```
