# GPU Virtualization Support Analysis

Comprehensive analysis of GPU virtualization capabilities for this development machine.

---

## System Configuration

**Motherboard:** ASRock Z890M Riptide WiFi
**CPU:** Intel Core Ultra 9 285K
**Chipset:** Intel Z890

### GPU Hardware

1. **NVIDIA GeForce RTX 5090**
   - PCI Address: `02:00.0`
   - Device ID: `10de:2b85`
   - Memory: 32607 MiB
   - IOMMU Group: 14
   - Current Driver: `nvidia` (v575.57.08)
   - CUDA Version: 12.9
   - Audio Device: `02:00.1` (HDMI audio controller)

2. **Intel Arrow Lake-U Integrated Graphics**
   - PCI Address: `00:02.0`
   - Device ID: `8086:7d67`
   - IOMMU Group: 0
   - Current Driver: `i915`
   - Status: Currently driving display (X.org)

---

## Virtualization Infrastructure Status

### ✅ Fully Supported Features

| Component | Status | Details |
|-----------|--------|---------|
| **CPU Virtualization** | ✅ Enabled | Intel VT-x (vmx flag present) |
| **IOMMU/VT-d** | ✅ Enabled | 25 IOMMU groups detected |
| **Kernel Parameters** | ✅ Configured | `iommu=pt` enabled |
| **KVM Modules** | ✅ Loaded | `kvm_intel` active |
| **VFIO Support** | ✅ Available | `vfio-pci` module present |
| **QEMU** | ✅ Installed | Version 8.2.2 |
| **OVMF/UEFI** | ✅ Installed | For UEFI guest support |
| **User Permissions** | ✅ Configured | Member of `libvirt` group |

---

## GPU Usage Modes in QEMU

Your system supports **4 different methods** for using GPUs with QEMU:

### 1. ✅ GPU Passthrough (PCI Passthrough) - FULLY SUPPORTED

Pass the entire physical GPU to a VM with near-native performance.

**Best for:** Gaming VMs, GPU-intensive workloads, Windows VMs needing full GPU access

#### NVIDIA RTX 5090 Passthrough Configuration

**IOMMU Group Status:** ✅ **CLEAN** (only GPU + Audio in group 14)
```
IOMMU Group 14:
├── 02:00.0 - NVIDIA GeForce RTX 5090 (VGA)
└── 02:00.1 - NVIDIA HDMI Audio
```

**Prerequisites:**
- Bind NVIDIA GPU to `vfio-pci` driver instead of `nvidia`
- Unbind from host before VM starts
- Guest must be UEFI (OVMF)

**Performance:** ~95-98% of native GPU performance

**Limitations:**
- GPU cannot be used by host while passed to VM
- Requires VM restart to switch GPU back to host
- Some NVIDIA GPUs have "Error 43" with GeForce drivers (workarounds exist)

**Configuration Steps:**

1. **Identify GPU IDs:**
   ```bash
   lspci -nn | grep -E "02:00.[01]"
   # 02:00.0 VGA: NVIDIA [10de:2b85]
   # 02:00.1 Audio: NVIDIA [10de:22e8]
   ```

2. **Isolate GPU from host driver:**

   Edit `/etc/default/grub`:
   ```
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt intel_iommu=on vfio-pci.ids=10de:2b85,10de:22e8"
   ```

   Update grub:
   ```bash
   sudo update-grub
   sudo reboot
   ```

3. **Verify VFIO binding:**
   ```bash
   lspci -k -s 02:00.0
   # Should show: Kernel driver in use: vfio-pci
   ```

4. **QEMU command example:**
   ```bash
   qemu-system-x86_64 \
     -enable-kvm \
     -cpu host,kvm=off,hv_vendor_id=1234567890ab \
     -smp 8,cores=4,threads=2 \
     -m 16G \
     -machine q35 \
     -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE.fd \
     -drive if=pflash,format=raw,file=/path/to/OVMF_VARS.fd \
     -device vfio-pci,host=02:00.0,multifunction=on \
     -device vfio-pci,host=02:00.1 \
     -drive file=vm-disk.qcow2,if=virtio \
     -vga none \
     -nographic \
     -vnc :0
   ```

5. **For NVIDIA "Error 43" workaround (if needed):**
   ```bash
   # Add these to VM XML (if using virt-manager):
   <features>
     <hyperv>
       <vendor_id state='on' value='1234567890ab'/>
     </hyperv>
     <kvm>
       <hidden state='on'/>
     </kvm>
   </features>
   ```

---

### 2. ⚠️ Intel GVT-g (GPU Virtualization) - NOT SUPPORTED

**Status:** Arrow Lake integrated graphics **does not support GVT-g**

**Why not available:**
- GVT-g was deprecated in newer Intel generations
- Arrow Lake (your CPU) uses new Xe architecture without GVT-g
- Parameter exists in i915 driver but no mdev types exposed

**Alternative:** Use SR-IOV or virtio-gpu instead

---

### 3. ✅ Virtio-GPU (Paravirtualized 3D) - SUPPORTED

Software-rendered 3D acceleration using host CPU/GPU via Virgil 3D.

**Best for:** Development VMs, Linux guests, light graphics workloads

**Advantages:**
- No dedicated GPU needed
- Can run multiple VMs simultaneously
- Good for development/desktop Linux VMs

**Limitations:**
- Software rendering (slow for games/compute)
- Limited DirectX support
- OpenGL 4.3 maximum

**QEMU command example:**
```bash
qemu-system-x86_64 \
  -enable-kvm \
  -cpu host \
  -smp 4 \
  -m 8G \
  -device virtio-vga-gl \
  -display gtk,gl=on \
  -drive file=vm.qcow2,if=virtio
```

**With virglrenderer and OpenGL:**
```bash
qemu-system-x86_64 \
  -enable-kvm \
  -device virtio-gpu-gl-pci \
  -display sdl,gl=on \
  -m 4G \
  -smp 4 \
  -drive file=vm.qcow2,if=virtio
```

---

### 4. ✅ QXL/VGA Emulation - SUPPORTED

Basic framebuffer emulation for display output.

**Best for:** Servers, headless VMs, VNC access

**Advantages:**
- Works everywhere
- No special setup
- Good for SPICE/VNC remote desktop

**Limitations:**
- No 3D acceleration
- Slow graphics performance

**QEMU command example:**
```bash
qemu-system-x86_64 \
  -enable-kvm \
  -vga qxl \
  -spice port=5930,disable-ticketing=on \
  -device virtio-serial \
  -chardev spicevmc,id=vdagent,debug=0,name=vdagent \
  -device virtserialport,chardev=vdagent,name=com.redhat.spice.0
```

---

## Intel Integrated GPU Considerations

**Current Status:** Intel GPU is driving the display (X.org on DISPLAY :0)

**Options for Intel GPU:**

1. **Keep for host display** (Recommended)
   - NVIDIA GPU passed to VM
   - Intel GPU continues running host desktop
   - Best of both worlds

2. **Pass Intel GPU to VM**
   - IOMMU Group 0 contains **only** `00:02.0` (clean passthrough possible)
   - Would require host to use NVIDIA for display
   - Less common setup

3. **Use for GVT-g (SR-IOV)** ❌
   - Not supported on Arrow Lake

**Recommended:** Keep Intel GPU for host, pass NVIDIA to VM

---

## Recommended Configurations

### Configuration A: High-Performance Gaming/Compute VM

**Use Case:** Windows gaming VM, CUDA development, GPU rendering

**Setup:**
- Pass NVIDIA RTX 5090 to VM
- Use Intel GPU for host display
- Near-native GPU performance in VM

**Steps:**
1. Bind NVIDIA GPU to vfio-pci (see passthrough section)
2. Keep using Intel GPU on host
3. Launch VM with NVIDIA passthrough
4. Install NVIDIA drivers in guest

---

### Configuration B: Multiple Development VMs

**Use Case:** Running several Linux development environments

**Setup:**
- Use virtio-gpu-gl for each VM
- No GPU passthrough needed
- All VMs can run simultaneously

**Steps:**
1. Launch VMs with virtio-vga-gl
2. Enable GL acceleration
3. Each VM gets paravirtualized 3D

---

### Configuration C: Dual GPU Passthrough

**Use Case:** Dedicated VM workstation

**Setup:**
- Pass NVIDIA to Windows VM
- Pass Intel to Linux VM
- Host runs headless

**Limitations:**
- Requires external GPU or second system for host management
- Complex setup
- Not recommended unless necessary

---

## IOMMU Group Analysis

**Your System IOMMU Groups:** 25 groups (0-24)

### Key Groups:

**Group 0 (Intel GPU):**
```
00:02.0 - Intel Arrow Lake-U Graphics (CLEAN - passthrough ready)
```

**Group 14 (NVIDIA GPU):**
```
02:00.0 - NVIDIA RTX 5090 VGA (CLEAN - passthrough ready)
02:00.1 - NVIDIA HDMI Audio (must pass with GPU)
```

**Status:** ✅ Both GPUs are in **clean IOMMU groups** - ideal for passthrough

---

## Performance Comparison

| Method | 3D Performance | Compute (CUDA) | Multi-VM | Complexity |
|--------|---------------|----------------|----------|------------|
| **GPU Passthrough** | ~98% native | ~98% native | ❌ No | High |
| **Virtio-GPU** | ~5-10% native | ❌ None | ✅ Yes | Low |
| **QXL/VGA** | ~1% native | ❌ None | ✅ Yes | Low |
| **GVT-g/SR-IOV** | ~60-80% native | Limited | ✅ Yes | N/A (not supported) |

---

## Quick Start: NVIDIA GPU Passthrough

**Fastest way to test GPU passthrough:**

1. **Install virt-manager (if not already):**
   ```bash
   sudo apt install virt-manager
   ```

2. **Enable VFIO for NVIDIA GPU:**
   ```bash
   sudo nano /etc/modprobe.d/vfio.conf
   ```
   Add:
   ```
   options vfio-pci ids=10de:2b85,10de:22e8
   ```

3. **Blacklist NVIDIA driver from loading:**
   ```bash
   sudo nano /etc/modprobe.d/blacklist-nvidia.conf
   ```
   Add:
   ```
   blacklist nvidia
   blacklist nvidia_drm
   blacklist nvidia_modeset
   blacklist nouveau
   ```

4. **Update initramfs:**
   ```bash
   sudo update-initramfs -u
   sudo reboot
   ```

5. **After reboot, verify VFIO binding:**
   ```bash
   lspci -k -s 02:00.0 | grep "Kernel driver"
   # Should show: vfio-pci
   ```

6. **Create Windows VM in virt-manager:**
   - Chipset: Q35
   - Firmware: UEFI (OVMF)
   - Add Hardware → PCI Host Device → Select `0000:02:00.0` and `0000:02:00.1`
   - Remove display and video devices (or keep VNC for troubleshooting)

7. **Install Windows and NVIDIA drivers in guest**

---

## Reverting GPU to Host Use

If you want to use NVIDIA GPU on host again:

1. **Disable VFIO binding:**
   ```bash
   sudo rm /etc/modprobe.d/vfio.conf
   sudo rm /etc/modprobe.d/blacklist-nvidia.conf
   ```

2. **Update initramfs:**
   ```bash
   sudo update-initramfs -u
   sudo reboot
   ```

3. **Verify NVIDIA driver loads:**
   ```bash
   nvidia-smi
   ```

---

## Troubleshooting

### Issue: "vfio-pci not binding to GPU"

**Solution:**
```bash
# Check if IOMMU is enabled
dmesg | grep -i iommu

# Manually bind to vfio-pci
echo "10de 2b85" | sudo tee /sys/bus/pci/drivers/vfio-pci/new_id
echo "0000:02:00.0" | sudo tee /sys/bus/pci/devices/0000:02:00.0/driver/unbind
echo "0000:02:00.0" | sudo tee /sys/bus/pci/drivers/vfio-pci/bind
```

### Issue: "NVIDIA Error 43 in Windows VM"

**Solution:**
Add to VM XML:
```xml
<features>
  <hyperv>
    <vendor_id state='on' value='1234567890ab'/>
  </hyperv>
  <kvm>
    <hidden state='on'/>
  </kvm>
</features>
```

### Issue: "VM won't boot with GPU passthrough"

**Checklist:**
- ✅ VM firmware is UEFI (OVMF)
- ✅ Both GPU and Audio device passed (02:00.0 and 02:00.1)
- ✅ ROM file provided (sometimes needed): `-device vfio-pci,host=02:00.0,romfile=/path/to/vbios.rom`
- ✅ Sufficient RAM allocated (16GB+ recommended)
- ✅ CPU cores properly allocated

### Issue: "Black screen after passing GPU"

**Solutions:**
1. Use a secondary display device (QXL/VNC) initially
2. Check BIOS: Enable "Above 4G Decoding" if available
3. Check BIOS: Enable "Resizable BAR" if supported
4. Dump and patch GPU vBIOS if necessary

---

## Performance Tuning

### CPU Pinning (for better performance)

```bash
# In VM XML:
<vcpu placement='static'>8</vcpu>
<cputune>
  <vcpupin vcpu='0' cpuset='2'/>
  <vcpupin vcpu='1' cpuset='3'/>
  <vcpupin vcpu='2' cpuset='4'/>
  <vcpupin vcpu='3' cpuset='5'/>
  <vcpupin vcpu='4' cpuset='6'/>
  <vcpupin vcpu='5' cpuset='7'/>
  <vcpupin vcpu='6' cpuset='8'/>
  <vcpupin vcpu='7' cpuset='9'/>
</cputune>
```

### Huge Pages (reduce memory latency)

```bash
# Add to /etc/sysctl.conf:
vm.nr_hugepages = 4096

# Apply:
sudo sysctl -p

# In VM XML:
<memoryBacking>
  <hugepages/>
</memoryBacking>
```

---

## Security Considerations

**GPU Passthrough Risks:**
- Guest can directly access GPU firmware
- Potential GPU memory side-channels
- GPU malware could persist in VRAM

**Mitigations:**
- Only pass GPU to trusted VMs
- Reset GPU between VM runs
- Use isolated IOMMU groups
- Keep GPU firmware updated

---

## Summary & Recommendations

### ✅ What Your System CAN Do:

1. **Full NVIDIA RTX 5090 passthrough** - Native gaming/compute performance
2. **Intel GPU for host display** - Seamless desktop experience while VM runs
3. **Multiple VMs with virtio-gpu** - Development environments with light graphics
4. **Clean IOMMU groups** - No ACS override needed

### ❌ What Your System CANNOT Do:

1. **Intel GVT-g virtualization** - Arrow Lake doesn't support it
2. **Split single GPU to multiple VMs** - No SR-IOV on consumer GPUs
3. **Use NVIDIA on host and VM simultaneously** - Need to choose one

### 🎯 Recommended Setup:

**For most users:**
- Keep Intel GPU for host (current setup)
- Pass NVIDIA RTX 5090 to Windows gaming VM when needed
- Use virtio-gpu for Linux development VMs
- Switch between host/VM GPU usage as needed

**Performance:** This gives you the best of both worlds - stable host desktop and near-native GPU performance in VMs.

---

## Additional Resources

- [VFIO Discussion Forum](https://reddit.com/r/VFIO)
- [Arch Linux PCI Passthrough Guide](https://wiki.archlinux.org/title/PCI_passthrough_via_OVMF)
- [Level1Techs GPU Passthrough](https://forum.level1techs.com/c/software/vfio/42)
- [QEMU Documentation](https://www.qemu.org/docs/master/system/devices/vfio-pci.html)

---

**Last Updated:** 2025-11-10
**System:** ASRock Z890M Riptide WiFi | Intel Core Ultra 9 285K | NVIDIA RTX 5090
