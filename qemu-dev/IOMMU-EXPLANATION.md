# What Happens When IOMMU is Disabled

## TL;DR

**With IOMMU OFF (`intel_iommu=off`):**
- ❌ GPU passthrough to VM: **WILL NOT WORK**
- ❌ VFIO cannot isolate devices
- ❌ No IOMMU groups created
- ✅ Host can use GPU normally
- ✅ VM can run (just without GPU)

**With IOMMU ON (`intel_iommu=on`):**
- ✅ GPU passthrough to VM: **WORKS**
- ✅ VFIO can safely isolate devices
- ✅ IOMMU groups exist
- ⚠️ Host may lose GPU access (when passed to VM)
- ✅ VM gets full GPU access

---

## What is IOMMU?

**IOMMU = Input-Output Memory Management Unit**

Think of it as a "bouncer" for hardware devices:

```
Without IOMMU:
┌─────────────┐
│     GPU     │──► Direct access to ALL system memory
└─────────────┘    (Can read/write anywhere - DANGEROUS in VM!)

With IOMMU:
┌─────────────┐
│     GPU     │──► IOMMU ──► Only assigned memory regions
└─────────────┘              (VM is isolated and safe)
```

## Current Status on Your System

```bash
# Your current kernel parameters
intel_iommu=off  # ← IOMMU is DISABLED
iommu=pt         # ← Passthrough mode (but disabled above)
```

### What This Means:

1. **No IOMMU Groups:**
   ```bash
   $ ls /sys/kernel/iommu_groups/
   # (empty or minimal - no device isolation)
   ```

2. **VFIO Cannot Work:**
   ```bash
   $ ./start-with-gpu.sh
   # Error: "vfio 0000:8a:00.0: no iommu_group found"
   ```

3. **Why It Fails:**
   - VFIO requires IOMMU to safely assign devices to VMs
   - Without IOMMU groups, there's no isolation
   - Security risk: VM could potentially access host memory

## Real-World Impact

### Scenario 1: Try to Start VM with GPU (IOMMU OFF)

```bash
$ ./start-with-gpu.sh

Result:
❌ VM starts BUT without GPU
❌ Error in log: "no iommu_group found"
❌ Inside VM: lspci | grep -i nvidia  → (nothing)
❌ Inside VM: nvidia-smi  → "No devices found"
```

### Scenario 2: Start VM with GPU (IOMMU ON)

```bash
$ sudo ./enable-iommu.sh
$ sudo reboot
$ ./start-with-gpu.sh

Result:
✅ VM starts with GPU attached
✅ Inside VM: lspci | grep -i nvidia  → H100 detected!
✅ Inside VM: nvidia-smi  → GPU visible
✅ Can install drivers and run CUDA
```

## Why Was IOMMU Disabled?

Possible reasons on your system:

1. **NVIDIA Driver Compatibility:**
   - Some NVIDIA driver versions had issues with IOMMU
   - May have been disabled for stability

2. **Performance:**
   - Very slight overhead with IOMMU (usually negligible)
   - Some users disable for max performance on host

3. **Default Configuration:**
   - May have been set during system setup

## What Happens When You Enable IOMMU

### Changes to Your System:

```bash
# Before (current)
intel_iommu=off   → IOMMU disabled globally
iommu=pt          → Passthrough mode (but overridden by above)

# After enable-iommu.sh
intel_iommu=on    → IOMMU enabled globally
iommu=pt          → Passthrough mode for performance
```

### After Reboot:

1. **IOMMU Groups Created:**
   ```bash
   $ ls /sys/kernel/iommu_groups/
   0  1  2  3  ... 150  151  ...
   # Each group = isolated set of devices
   ```

2. **GPU Gets Its Own Group:**
   ```bash
   $ find /sys/kernel/iommu_groups/ -name "*8a:00.0*"
   /sys/kernel/iommu_groups/42/devices/0000:8a:00.0
   # GPU is in IOMMU group 42 (example)
   ```

3. **VFIO Can Bind:**
   ```bash
   $ echo "10de 2336" > /sys/bus/pci/drivers/vfio-pci/new_id
   $ lspci -k -s 8a:00.0
   Kernel driver in use: vfio-pci  # ← Success!
   ```

4. **QEMU Can Attach GPU:**
   ```bash
   qemu-system-x86_64 ... -device vfio-pci,host=8a:00.0
   # ✅ Works now!
   ```

## Safety & Risks

### Is It Safe to Enable IOMMU?

**YES - Very Safe:**
- ✅ Standard Linux feature (used by all cloud providers)
- ✅ Enable-iommu.sh creates backup of GRUB config
- ✅ Can revert easily if needed
- ✅ Only affects device isolation
- ✅ No data loss risk

### Potential Issues (Rare):

1. **Some hardware incompatibility**
   - Fix: Boot with old kernel from GRUB menu
   - Restore: Edit `/etc/default/grub` back to `intel_iommu=off`

2. **Slight performance impact**
   - Usually < 1% overhead
   - Only matters for extreme benchmarks

3. **GPU not in isolated group**
   - Some systems group GPU with other devices
   - May need to pass multiple devices

## Testing IOMMU Status

### Before Enabling:

```bash
# Check current status
$ cat /proc/cmdline | grep iommu
intel_iommu=off iommu=pt

# Check for IOMMU groups (will be empty/minimal)
$ ls /sys/kernel/iommu_groups/
(empty or very few)

# Try GPU passthrough (will fail)
$ ./start-with-gpu.sh
Error: no iommu_group found
```

### After Enabling + Reboot:

```bash
# Check current status
$ cat /proc/cmdline | grep iommu
intel_iommu=on iommu=pt

# Check for IOMMU groups (will have many)
$ ls /sys/kernel/iommu_groups/
0  1  2  3  4  5  ... 150  151  152

# Check GPU's group
$ find /sys/kernel/iommu_groups/ -name "*8a:00.0*"
/sys/kernel/iommu_groups/XX/devices/0000:8a:00.0

# Try GPU passthrough (will work!)
$ ./start-with-gpu.sh
✅ VM started with PID: XXXXX
✅ GPU successfully bound to vfio-pci
```

## Alternative: Use GPU on Host Instead

If you don't want to enable IOMMU or reboot:

### Option 1: Just Use Host GPU
```bash
# On your host system (no VM)
$ nvidia-smi
$ nvcc --version
# CUDA already works here!
```

### Option 2: Docker with GPU
```bash
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
# Shares GPU with host, no passthrough needed
```

### Option 3: VM Without GPU
```bash
# For kernel development only (no GPU)
$ ./start-vm.sh  # (not start-with-gpu.sh)
# VM works fine, just no GPU inside
```

## Summary Table

| Feature | IOMMU OFF (current) | IOMMU ON (after enable) |
|---------|-------------------|----------------------|
| Host GPU usage | ✅ Works | ✅ Works |
| VM GPU passthrough | ❌ Fails | ✅ Works |
| VFIO support | ❌ No | ✅ Yes |
| Security isolation | ⚠️ Limited | ✅ Full |
| IOMMU groups | ❌ None | ✅ Many |
| Reboot required | - | ✅ Yes |
| Risk | ✅ None | ✅ Very low |

## Recommendation

**For your use case (kernel development + GPU testing in VM):**

✅ **Enable IOMMU** - It's the correct setup for safe GPU passthrough

**Steps:**
```bash
cd /root/co-processor-demo/qemu-dev/scripts
sudo ./enable-iommu.sh
sudo reboot
./start-with-gpu.sh
```

This gives you the best of both worlds:
- Safe, isolated VM for kernel experiments
- Full GPU access inside VM
- No risk to host system if VM crashes
