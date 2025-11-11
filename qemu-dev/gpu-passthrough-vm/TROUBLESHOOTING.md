# Troubleshooting GPU Passthrough

Common issues and solutions for QEMU GPU passthrough.

## Issue: GPU not binding to vfio-pci after reboot

**Symptoms:**
- `./check-status.sh` shows GPU still using nvidia driver
- `lspci -k -s 02:00.0` shows "Kernel driver in use: nvidia"

**Solutions:**

1. Check if VFIO modules loaded:
   ```bash
   lsmod | grep vfio
   ```

2. Manually bind GPU to vfio-pci:
   ```bash
   sudo modprobe vfio-pci
   echo "10de 2b85" | sudo tee /sys/bus/pci/drivers/vfio-pci/new_id
   echo "0000:02:00.0" | sudo tee /sys/bus/pci/devices/0000:02:00.0/driver/unbind
   echo "0000:02:00.0" | sudo tee /sys/bus/pci/drivers/vfio-pci/bind
   ```

3. Check kernel parameters:
   ```bash
   cat /proc/cmdline | grep iommu
   ```
   Should include: `iommu=pt` or `intel_iommu=on`

4. Verify VFIO config:
   ```bash
   cat /etc/modprobe.d/vfio.conf
   ```

## Issue: VM fails to start with "Could not initialize KVM"

**Solution:**
```bash
# Check if KVM modules are loaded
lsmod | grep kvm

# Load KVM module if needed
sudo modprobe kvm_intel

# Check KVM device permissions
ls -la /dev/kvm
sudo chmod 666 /dev/kvm
```

## Issue: Black screen in VNC after VM starts

**Possible causes:**

1. **GPU is still initializing**
   - Wait 30-60 seconds for UEFI/BIOS to load
   - GPU may take time to initialize

2. **OVMF firmware issue**
   - Ensure OVMF_VARS.fd is writable
   - Try recreating: `cp /usr/share/OVMF/OVMF_VARS.fd ./OVMF_VARS.fd`

3. **GPU ROM issue**
   - Some GPUs need ROM file extracted
   - Download vBIOS from TechPowerUp
   - Add to QEMU: `-device vfio-pci,host=02:00.0,romfile=path/to/vbios.rom`

## Issue: NVIDIA Error 43 in Windows VM

**Solution:**
Edit start-vm.sh and ensure these CPU flags are present:
```bash
-cpu host,kvm=off,hv_vendor_id=1234567890ab,hv_relaxed,hv_spinlocks=0x1fff
```

For stubborn cases, hide hypervisor completely:
```bash
-cpu host,kvm=off,hv_vendor_id=random12char,-hypervisor
```

## Issue: VM performance is slow

**Solutions:**

1. **Enable CPU pinning:**
   Edit start-vm.sh and add CPU affinity

2. **Enable Huge Pages:**
   ```bash
   # Add to /etc/sysctl.conf
   vm.nr_hugepages = 4096

   sudo sysctl -p

   # Add to start-vm.sh:
   -mem-path /dev/hugepages
   ```

3. **Use virtio drivers:**
   Already included in start-vm.sh for disk and network

## Issue: No network in VM

**Solutions:**

1. **Check if user networking is working:**
   VM should get IP via DHCP (usually 10.0.2.15)

2. **Switch to bridged networking:**
   Edit start-vm.sh and replace user networking with:
   ```bash
   -netdev bridge,id=net0,br=br0 \
   -device virtio-net-pci,netdev=net0,mac=$VM_MAC_ADDRESS
   ```

3. **Create bridge (if using bridged mode):**
   ```bash
   sudo apt install bridge-utils
   sudo brctl addbr br0
   sudo ip link set br0 up
   sudo brctl addif br0 enp129s0
   ```

## Issue: Cannot connect to VNC

**Solutions:**

1. **Check if VM is running:**
   ```bash
   ps aux | grep qemu
   ```

2. **Check VNC port:**
   ```bash
   sudo netstat -tlnp | grep 5900
   ```

3. **Install VNC client:**
   ```bash
   sudo apt install tigervnc-viewer
   vncviewer localhost:5900
   ```

4. **Try different VNC client:**
   ```bash
   sudo apt install remmina
   remmina -c vnc://localhost:5900
   ```

## Issue: GPU not releasing after VM shutdown

**Solution:**
```bash
# Stop VM
./stop-vm.sh

# Reset GPU
echo 1 | sudo tee /sys/bus/pci/devices/0000:02:00.0/remove
echo 1 | sudo tee /sys/bus/pci/rescan

# Check status
./check-status.sh
```

## Issue: "VFIO device in use"

**Cause:** Another process or VM is using the GPU

**Solution:**
```bash
# Find what's using vfio devices
sudo lsof | grep vfio

# Kill any old QEMU processes
pkill -9 qemu-system-x86

# Remove stale lock files
sudo rm -f /tmp/qemu-*.pid
sudo rm -f /tmp/qemu-monitor-*.sock
```

## Issue: Host crashes when starting VM

**Possible causes:**

1. **Insufficient RAM:**
   - Reduce VM_MEMORY in config.sh
   - Close other applications

2. **BIOS settings:**
   - Enable "Above 4G Decoding" in BIOS
   - Enable "Resizable BAR" if available

3. **Bad IOMMU group:**
   - Run `./check-status.sh` to verify clean IOMMU group
   - Group 14 should only contain GPU and GPU audio

## Issue: Cannot hear sound from VM

**Solution:**

1. **GPU HDMI audio should work** (02:00.1 is passed through)

2. **Add software audio:**
   Add to start-vm.sh:
   ```bash
   -device ich9-intel-hda \
   -device hda-duplex
   ```

## Issue: Mouse/keyboard not working in VM

**Solution:**

1. **Already using virtio input** in start-vm.sh

2. **Alternative - USB tablet:**
   Replace virtio-keyboard/mouse with:
   ```bash
   -device usb-tablet
   ```

3. **USB passthrough:**
   Add your USB devices:
   ```bash
   -device usb-host,vendorid=0x1234,productid=0x5678
   ```

## Getting Help

If none of these solutions work:

1. **Check logs:**
   ```bash
   tail -f logs/vm-*.log
   cat logs/last-qemu-command.txt
   ```

2. **Verbose QEMU output:**
   Remove `-daemonize` from start-vm.sh and run in foreground

3. **Check system logs:**
   ```bash
   sudo dmesg | grep -i vfio
   sudo dmesg | grep -i iommu
   ```

4. **Community resources:**
   - /r/VFIO on Reddit
   - Level1Techs forums
   - QEMU mailing list
