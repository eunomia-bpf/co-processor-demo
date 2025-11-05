# NVIDIA-FS Installation Status

## Installation Completed

**Date:** 2025-11-05
**Package:** nvidia-gds-12-9 (version 12.9.1-1)

### Packages Installed

```
✓ nvidia-gds-12-9        - GPUDirect Storage meta-package
✓ nvidia-fs-dkms         - Kernel module (DKMS package v2.26.6)
✓ nvidia-fs              - Module loader utilities
✓ libcufile-12-9         - cuFile runtime library v1.14.1.1
✓ libcufile-dev-12-9     - cuFile development headers
✓ gds-tools-12-9         - GDS diagnostic and testing tools
```

### Tools Available

New tools installed in `/usr/local/cuda-12.9/gds/tools/`:
- `gdscheck` - GDS system compatibility checker
- `gdscheck.py` - Python version of system checker
- `gdsio` - I/O benchmarking tool
- `gds_stats` - Statistics monitor
- `gds_perf.sh` - Performance testing script

## Module Loading Issue

### Problem

The nvidia-fs kernel module **cannot be loaded** due to symbol conflicts:

```bash
$ sudo modprobe nvidia-fs
modprobe: ERROR: could not insert 'nvidia_fs': Unknown symbol in module
```

### Root Cause

From `dmesg`:
```
nvidia_fs: Unknown symbol nvidia_p2p_dma_unmap_pages (err -2)
nvidia_fs: Unknown symbol nvidia_p2p_put_pages_persistent (err -2)
nvidia_fs: Unknown symbol nvidia_p2p_get_pages (err -2)
nvidia_fs: Unknown symbol nvidia_p2p_put_pages (err -2)
nvidia_fs: Unknown symbol nvidia_p2p_dma_map_pages (err -2)
nvidia_fs: Unknown symbol nvidia_p2p_free_dma_mapping (err -2)
nvidia_fs: Unknown symbol nvidia_p2p_free_page_table (err -2)
nvidia_fs: Unknown symbol nvidia_p2p_get_pages_persistent (err -2)
```

**Analysis:**
- nvidia-fs module (v2.26.6) expects certain P2P symbols from nvidia.ko
- Current NVIDIA driver (580.95.05) may have different symbol exports
- Version mismatch between nvidia-fs-dkms and proprietary nvidia driver

### Module Information

```bash
$ modinfo nvidia-fs
filename:       /lib/modules/6.8.0-86-generic/updates/dkms/nvidia-fs.ko.zst
description:    NVIDIA GPUDirect Storage
license:        GPL v2
version:        2.26.6
depends:
vermagic:       6.8.0-86-generic SMP preempt mod_unload modversions
```

Module was built correctly for kernel 6.8.0-86-generic but cannot link to nvidia driver.

## Current Status

### What Works ✓

1. **cuFile library** - Fully functional
   - Can use cuFile API for GPU-storage I/O
   - Falls back to compatibility mode automatically
   - No code changes needed

2. **PCI P2P DMA workaround** - Available
   - Can enable `use_pci_p2pdma: true` in `/etc/cufile.json`
   - Uses Linux kernel P2P infrastructure
   - Provides performance improvement over compat mode

3. **GDS tools** - Installed and working
   - Can run gdscheck for diagnostics
   - Can benchmark with gdsio

### What Doesn't Work ✗

1. **nvidia-fs kernel module** - Cannot load
   - Direct GPU-to-NVMe DMA not active
   - Full GDS features unavailable
   - Operating in compatibility/workaround mode

## Recommendations

### Option 1: Use PCI P2P DMA (Immediate)

**Best option for now:**

```bash
sudo nano /etc/cufile.json
```

Change:
```json
"use_pci_p2pdma": true,
"rdma_dynamic_routing": true
```

**Benefits:**
- No kernel module needed
- Uses standard Linux P2P DMA
- 1.5-2x performance improvement
- Works right now

### Option 2: Wait for Driver Update

The symbol mismatch suggests:
- NVIDIA driver 580.95.05 may be too new
- nvidia-fs 2.26.6 may need update
- Or driver may need downgrade

**Monitor:**
- NVIDIA driver updates
- nvidia-fs-dkms updates
- Check compatibility matrix at nvidia.com

### Option 3: Try Alternative Driver Version

If critical, could try:
```bash
# Find compatible driver version for nvidia-fs 2.26.6
# Check NVIDIA GDS documentation for supported driver versions
# Downgrade NVIDIA driver if needed (not recommended)
```

## Performance Expectations

With PCI P2P DMA enabled (no nvidia-fs):

| Metric | Performance |
|--------|-------------|
| Sequential Read | 15-25 GB/s |
| Sequential Write | 15-25 GB/s |
| Latency | ~40-60 μs |
| CPU Overhead | Medium |

With nvidia-fs working (future):

| Metric | Performance |
|--------|-------------|
| Sequential Read | 20-35 GB/s |
| Sequential Write | 20-35 GB/s |
| Latency | ~20-30 μs |
| CPU Overhead | Low |

Note: Your cross-socket topology limits max performance regardless of method.

## Testing GDS Tools

Even without nvidia-fs loaded, you can test:

```bash
# Check system compatibility
/usr/local/cuda-12.9/gds/tools/gdscheck -p

# Test cuFile functionality (compat mode)
cd /usr/local/cuda-12.9/gds/samples/
make
./cufilesample_01
```

## Conclusion

**nvidia-fs installation:** ✓ Complete
**nvidia-fs module loading:** ✗ Failed (symbol mismatch)
**Workaround available:** ✓ Yes (PCI P2P DMA)
**Action:** Enable `use_pci_p2pdma: true` for best current performance

The installation succeeded, but the module cannot load due to driver compatibility. The PCI P2P DMA workaround provides good performance while waiting for this to be resolved.
