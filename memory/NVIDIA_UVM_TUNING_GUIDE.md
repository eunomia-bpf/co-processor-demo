# NVIDIA UVM (Unified Virtual Memory) Tuning Guide

Complete guide to tuning NVIDIA UVM parameters for optimal performance on H100 GPUs.

---

## System Information

**GPU:** NVIDIA H100 (97871 MiB)
**Driver Version:** 580.105.08
**CUDA Version:** 12.9
**Kernel:** 6.8.0-87-generic

---

## Table of Contents

1. [What is UVM?](#what-is-uvm)
2. [Current Configuration](#current-configuration)
3. [Tunable Parameters](#tunable-parameters)
4. [How to Tune](#how-to-tune)
5. [Common Tuning Scenarios](#common-tuning-scenarios)
6. [Monitoring & Debugging](#monitoring--debugging)
7. [Performance Best Practices](#performance-best-practices)

---

## What is UVM?

**Unified Virtual Memory (UVM)** is NVIDIA's technology that provides a single unified address space accessible from any processor in the system (CPU or GPU). It enables:

- **Automatic page migration** between CPU and GPU memory
- **Oversubscription** - use more memory than physically available on GPU
- **Simplified programming** - no explicit memory copies needed
- **Page fault handling** - data migrates on-demand

### UVM vs Traditional CUDA Memory Management

| Feature | Traditional CUDA | UVM |
|---------|-----------------|-----|
| Memory allocation | `cudaMalloc()` | `cudaMallocManaged()` |
| Data transfers | Explicit `cudaMemcpy()` | Automatic migration |
| Oversubscription | Limited | Supported |
| Programming complexity | Higher | Lower |
| Performance overhead | None | Page fault overhead |

---

## Current Configuration

### Performance - Prefetching

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| `uvm_perf_prefetch_enable` | **1** (enabled) | Enable predictive page prefetching |
| `uvm_perf_prefetch_threshold` | **51** | Fault percentage threshold to trigger prefetch (0-100) |
| `uvm_perf_prefetch_min_faults` | **1** | Minimum faults before prefetch activates |
| `uvm_perf_reenable_prefetch_faults_lapse_msec` | **1000** | Time before re-enabling prefetch after disable |

**Impact:** Controls automatic prefetching to reduce future page faults.
- Lower threshold = more aggressive prefetching
- Higher threshold = more conservative (waits for more faults)

---

### Performance - Access Counters

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| `uvm_perf_access_counter_migration_enable` | **-1** (default) | Enable migration based on access counters |
| `uvm_perf_access_counter_threshold` | **256** | Remote access count to trigger migration |
| `uvm_perf_access_counter_batch_count` | **256** | Number of access counter notifications to batch |

**Impact:** Controls when frequently-accessed remote pages migrate.
- Lower threshold = migrate sooner (more aggressive)
- Higher threshold = migrate later (more conservative)

---

### Performance - Thrashing Protection

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| `uvm_perf_thrashing_enable` | **1** (enabled) | Enable thrashing detection/mitigation |
| `uvm_perf_thrashing_threshold` | **3** | Thrashing detection threshold |
| `uvm_perf_thrashing_pin_threshold` | **10** | Pin threshold for thrashing pages |
| `uvm_perf_thrashing_pin` | **300** | Duration to pin thrashing pages (ms) |
| `uvm_perf_thrashing_lapse_usec` | **500** | Time window for thrash detection (μs) |
| `uvm_perf_thrashing_epoch` | **2000** | Epoch duration for thrash tracking (ms) |
| `uvm_perf_thrashing_nap` | **1** | Enable "napping" for thrashing pages |
| `uvm_perf_thrashing_max_resets` | **4** | Max resets before permanent mitigation |

**Impact:** Prevents excessive page migrations when memory is accessed from both CPU and GPU.
- Lower threshold = detect thrashing earlier
- Higher pin duration = keep pages stable longer

---

### Performance - Fault Handling

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| `uvm_perf_fault_batch_count` | **256** | Number of faults to batch together |
| `uvm_perf_fault_coalesce` | **1** (enabled) | Coalesce adjacent faults |
| `uvm_perf_fault_max_batches_per_service` | **20** | Max fault batches to service at once |
| `uvm_perf_fault_max_throttle_per_service` | **5** | Max throttle count per service |
| `uvm_perf_fault_replay_policy` | **2** | Fault replay policy |
| `uvm_perf_fault_replay_update_put_ratio` | **50** | Replay update/put ratio |

**Impact:** Controls how page faults are batched and serviced.
- Larger batches = fewer servicing operations (more efficient)
- Smaller batches = lower latency per fault

---

### Memory Management

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| `uvm_global_oversubscription` | **1** (enabled) | Allow memory oversubscription |
| `uvm_page_table_location` | **(null)** auto | Page table location: "vid" or "sys" |
| `uvm_fault_force_sysmem` | **0** (disabled) | Force faulted pages to system memory |
| `uvm_peer_copy` | **phys** | Peer copy mode: "phys" or "virt" |
| `uvm_perf_map_remote_on_eviction` | **1** (enabled) | Map remotely on eviction |
| `uvm_perf_map_remote_on_native_atomics_fault` | **0** (disabled) | Map remote on atomic faults |
| `uvm_perf_migrate_cpu_preunmap_enable` | **1** (enabled) | Preunmap before CPU migration |
| `uvm_perf_migrate_cpu_preunmap_block_order` | **2** | Block order for preunmap |
| `uvm_cpu_chunk_allocation_sizes` | **(varies)** | CPU chunk allocation sizes |

**Impact:** Core memory management behavior.
- Oversubscription allows using more memory than GPU has
- Page table location affects TLB performance

---

### Advanced Features

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| `uvm_ats_mode` | **1** (enabled) | Address Translation Services |
| `uvm_disable_hmm` | **Y** (disabled) | Disable Heterogeneous Memory Management |
| `uvm_enable_va_space_mm` | **1** (enabled) | Enable VA space mm_notifiers |
| `uvm_block_cpu_to_cpu_copy_with_ce` | **(varies)** | Use GPU CE for CPU-to-CPU copies |

**Impact:** Advanced integration features.
- **HMM** (currently disabled) provides better integration with Linux memory management
- **ATS** enables PCIe address translation

---

### Experimental/Debug Parameters

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| `uvm_exp_gpu_cache_peermem` | **(default)** | Force caching for peer memory (experimental) |
| `uvm_exp_gpu_cache_sysmem` | **(default)** | Force caching for system memory (experimental) |
| `uvm_debug_prints` | **0** (disabled) | Enable debug prints |
| `uvm_enable_debug_procfs` | **0** (disabled) | Enable debug procfs entries |
| `uvm_leak_checker` | **0** (disabled) | Memory leak checking (0=off, 1=basic, 2=detailed) |

**Impact:** Debugging and experimental features. Use with caution.

---

## Tunable Parameters

### Read-Only Parameters (Boot-time only)

These can only be set via kernel module parameters:

- `uvm_ats_mode`
- `uvm_disable_hmm`
- `uvm_enable_va_space_mm`
- `uvm_channel_*` parameters
- `uvm_peer_copy`
- Most `uvm_perf_*` parameters

### Writable Runtime Parameters

These can be changed at runtime via `/sys/module/nvidia_uvm/parameters/`:

- `uvm_block_cpu_to_cpu_copy_with_ce`
- `uvm_cpu_chunk_allocation_sizes`
- `uvm_debug_enable_push_desc`
- `uvm_debug_enable_push_acquire_info`
- `uvm_debug_prints`
- `uvm_downgrade_force_membar_sys`
- `uvm_fault_force_sysmem`
- `uvm_release_asserts*`

---

## How to Tune

### Method 1: View Current Values

```bash
# View single parameter
cat /sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable

# View all parameters
ls -la /sys/module/nvidia_uvm/parameters/

# View specific category (e.g., prefetch)
for p in /sys/module/nvidia_uvm/parameters/*prefetch*; do
    echo "$(basename $p): $(cat $p)"
done
```

### Method 2: Runtime Changes (Temporary)

**Note:** Most performance parameters are READ-ONLY at runtime. Changes require module reload.

For writable parameters:
```bash
# Example: Enable debug prints
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_debug_prints
```

### Method 3: Module Parameters (Persistent)

Create configuration file:
```bash
sudo nano /etc/modprobe.d/nvidia-uvm.conf
```

Add parameters:
```
# Performance tuning
options nvidia-uvm uvm_perf_prefetch_enable=1
options nvidia-uvm uvm_perf_prefetch_threshold=25
options nvidia-uvm uvm_perf_access_counter_threshold=128
options nvidia-uvm uvm_perf_thrashing_threshold=5

# Enable HMM (if kernel supports it)
options nvidia-uvm uvm_disable_hmm=0

# Page table location
options nvidia-uvm uvm_page_table_location=vid
```

Apply changes:
```bash
# Unload nvidia-uvm module (requires no processes using it)
sudo rmmod nvidia_uvm

# Reload with new parameters
sudo modprobe nvidia-uvm

# Or reboot for guaranteed clean state
sudo reboot
```

### Method 4: Boot-time Kernel Parameters

Add to `/etc/default/grub`:
```bash
GRUB_CMDLINE_LINUX_DEFAULT="... nvidia-uvm.uvm_perf_prefetch_threshold=25"
```

Update GRUB and reboot:
```bash
sudo update-grub
sudo reboot
```

---

## Common Tuning Scenarios

### Scenario 1: Reduce Page Faults (Aggressive Prefetching)

**Goal:** Minimize page fault overhead for predictable access patterns.

**Tuning:**
```bash
# /etc/modprobe.d/nvidia-uvm.conf
options nvidia-uvm uvm_perf_prefetch_enable=1
options nvidia-uvm uvm_perf_prefetch_threshold=25        # More aggressive (was 51)
options nvidia-uvm uvm_perf_prefetch_min_faults=1        # Prefetch after 1 fault
options nvidia-uvm uvm_perf_fault_batch_count=512        # Larger batches (was 256)
```

**Best for:**
- Streaming workloads
- Sequential memory access
- Large dataset processing

**Trade-off:** May prefetch unnecessary pages, wasting bandwidth.

---

### Scenario 2: Reduce Thrashing (Memory-Intensive Workloads)

**Goal:** Prevent excessive page migrations when data accessed from both CPU and GPU.

**Tuning:**
```bash
# /etc/modprobe.d/nvidia-uvm.conf
options nvidia-uvm uvm_perf_thrashing_threshold=5        # Detect earlier (was 3)
options nvidia-uvm uvm_perf_thrashing_pin=500            # Pin longer (was 300ms)
options nvidia-uvm uvm_perf_thrashing_lapse_usec=1000    # Wider window (was 500μs)
options nvidia-uvm uvm_perf_thrashing_pin_threshold=5    # Lower pin threshold (was 10)
```

**Best for:**
- CPU-GPU collaborative algorithms
- Iterative computations
- Graph processing

**Trade-off:** Pages stay pinned longer, potentially reducing migration opportunities.

---

### Scenario 3: Optimize for Large Memory Oversubscription

**Goal:** Handle workloads larger than GPU memory efficiently.

**Tuning:**
```bash
# /etc/modprobe.d/nvidia-uvm.conf
options nvidia-uvm uvm_global_oversubscription=1         # Ensure enabled
options nvidia-uvm uvm_page_table_location=vid           # Page tables in VRAM
options nvidia-uvm uvm_perf_access_counter_threshold=512 # Higher threshold
options nvidia-uvm uvm_perf_fault_max_batches_per_service=40  # More batches
```

**Best for:**
- Machine learning training with large models
- Large-scale simulations
- Data analytics on massive datasets

**Trade-off:** More page fault overhead, slower than fitting in GPU memory.

---

### Scenario 4: Enable HMM (Heterogeneous Memory Management)

**Goal:** Better integration with Linux kernel memory management.

**Prerequisites:**
- Kernel 5.13+ with HMM support
- ATS-capable system

**Tuning:**
```bash
# /etc/modprobe.d/nvidia-uvm.conf
options nvidia-uvm uvm_disable_hmm=0                     # Enable HMM
options nvidia-uvm uvm_ats_mode=1                        # Enable ATS
options nvidia-uvm uvm_enable_va_space_mm=1              # Enable VA space
```

**Best for:**
- Modern kernels with HMM support
- Complex memory management scenarios
- Improved CPU/GPU memory coherence

**Trade-off:** Requires kernel support, may have compatibility issues.

---

### Scenario 5: Low-Latency Workloads

**Goal:** Minimize latency for real-time or interactive applications.

**Tuning:**
```bash
# /etc/modprobe.d/nvidia-uvm.conf
options nvidia-uvm uvm_perf_fault_batch_count=64         # Smaller batches
options nvidia-uvm uvm_perf_fault_max_batches_per_service=5
options nvidia-uvm uvm_perf_access_counter_migration_enable=1  # Enable proactive
options nvidia-uvm uvm_perf_access_counter_threshold=64  # Lower threshold
```

**Best for:**
- Interactive rendering
- Real-time processing
- Low-latency inference

**Trade-off:** More frequent servicing, potentially lower throughput.

---

### Scenario 6: High-Throughput Batch Processing

**Goal:** Maximize throughput for batch workloads.

**Tuning:**
```bash
# /etc/modprobe.d/nvidia-uvm.conf
options nvidia-uvm uvm_perf_fault_batch_count=1024       # Large batches
options nvidia-uvm uvm_perf_fault_max_batches_per_service=40
options nvidia-uvm uvm_perf_fault_coalesce=1             # Ensure coalescing
options nvidia-uvm uvm_perf_prefetch_enable=1            # Enable prefetch
```

**Best for:**
- Batch inference
- Large-scale data processing
- Offline analytics

**Trade-off:** Higher latency per operation, optimized for throughput.

---

## Monitoring & Debugging

### UVM Statistics

```bash
# View UVM statistics
cat /proc/driver/nvidia-uvm/stats

# Monitor in real-time
watch -n 1 cat /proc/driver/nvidia-uvm/stats
```

### GPU Memory Usage

```bash
# nvidia-smi
nvidia-smi

# Detailed memory info
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# Monitor in real-time
watch -n 1 nvidia-smi
```

### Page Fault Analysis

Enable debug output:
```bash
# Enable debug procfs (if writable)
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_enable_debug_procfs

# View debug info
cat /proc/driver/nvidia-uvm/debug
```

### Kernel Messages

```bash
# View UVM-related kernel messages
dmesg | grep -i uvm

# Monitor in real-time
dmesg -w | grep -i uvm
```

### Application Profiling

Use NVIDIA profiling tools:

```bash
# Nsight Systems
nsys profile --trace=cuda,nvtx ./your_app

# Nsight Compute
ncu --set full ./your_app

# nvprof (legacy)
nvprof --print-gpu-trace ./your_app
```

### Memory Leak Detection

Enable leak checker:
```bash
# /etc/modprobe.d/nvidia-uvm.conf
options nvidia-uvm uvm_leak_checker=2  # Detailed tracking
```

---

## Performance Best Practices

### General Guidelines

1. **Start with defaults** - Measure baseline performance
2. **Change one parameter at a time** - Isolate impact
3. **Profile your workload** - Use nvprof/nsys before tuning
4. **Monitor page faults** - High fault rate indicates tuning opportunity
5. **Test with realistic data** - Synthetic benchmarks may mislead

### Parameter Selection Strategy

```
┌─────────────────────────────────────────────┐
│    Analyze Workload Characteristics         │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
    Sequential          Random
    Access             Access
        │                   │
        ▼                   ▼
  Enable Prefetch    Disable Prefetch
  (threshold=25)     (threshold=75)
        │                   │
        └─────────┬─────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
    CPU+GPU              GPU-only
    Sharing             Access
        │                   │
        ▼                   ▼
  Thrashing          Access Counter
  Mitigation         Migration
  (pin=500ms)       (threshold=128)
```

### Memory Access Patterns

| Pattern | Prefetch | Access Counters | Thrashing | Batch Size |
|---------|----------|----------------|-----------|------------|
| **Sequential read** | High | Low | Low | Large |
| **Random read** | Low | Medium | Low | Medium |
| **CPU-GPU shared** | Medium | Medium | High | Medium |
| **Streaming** | High | Low | Low | Large |
| **Iterative** | Medium | High | High | Small |

### Workload-Specific Recommendations

#### Deep Learning Training
```bash
options nvidia-uvm uvm_perf_prefetch_enable=1
options nvidia-uvm uvm_perf_prefetch_threshold=40
options nvidia-uvm uvm_global_oversubscription=1
options nvidia-uvm uvm_page_table_location=vid
```

#### Graph Analytics
```bash
options nvidia-uvm uvm_perf_thrashing_threshold=5
options nvidia-uvm uvm_perf_thrashing_pin=400
options nvidia-uvm uvm_perf_access_counter_threshold=256
options nvidia-uvm uvm_perf_fault_batch_count=512
```

#### Molecular Dynamics
```bash
options nvidia-uvm uvm_perf_prefetch_enable=1
options nvidia-uvm uvm_perf_prefetch_threshold=30
options nvidia-uvm uvm_perf_thrashing_enable=1
options nvidia-uvm uvm_perf_fault_batch_count=256
```

#### Ray Tracing / Rendering
```bash
options nvidia-uvm uvm_perf_fault_batch_count=128
options nvidia-uvm uvm_perf_access_counter_migration_enable=1
options nvidia-uvm uvm_perf_access_counter_threshold=64
```

---

## Troubleshooting

### Issue: High Page Fault Overhead

**Symptoms:**
- Low GPU utilization
- High kernel time in profiler
- Frequent stalls

**Solutions:**
1. Enable prefetching: `uvm_perf_prefetch_enable=1`
2. Lower prefetch threshold: `uvm_perf_prefetch_threshold=25`
3. Increase batch size: `uvm_perf_fault_batch_count=512`
4. Consider using explicit `cudaMemPrefetchAsync()`

---

### Issue: Memory Thrashing

**Symptoms:**
- High CPU and GPU utilization
- Poor performance on shared data
- Excessive migrations in profiler

**Solutions:**
1. Increase thrashing threshold: `uvm_perf_thrashing_threshold=5`
2. Increase pin duration: `uvm_perf_thrashing_pin=500`
3. Widen detection window: `uvm_perf_thrashing_lapse_usec=1000`
4. Use `cudaMemAdvise()` hints in application

---

### Issue: Out of Memory (OOM)

**Symptoms:**
- CUDA OOM errors
- System hangs
- Kernel OOM killer

**Solutions:**
1. Ensure oversubscription enabled: `uvm_global_oversubscription=1`
2. Check system memory: `free -h`
3. Reduce batch sizes if applicable
4. Use memory-mapped files for very large datasets

---

### Issue: Poor Prefetch Performance

**Symptoms:**
- Prefetching doesn't help
- Increased memory bandwidth usage
- No reduction in page faults

**Solutions:**
1. Profile access patterns - may be random
2. Increase threshold: `uvm_perf_prefetch_threshold=75`
3. Use explicit prefetch: `cudaMemPrefetchAsync()`
4. Disable if random access: `uvm_perf_prefetch_enable=0`

---

### Issue: Cannot Change Parameters

**Symptoms:**
- `echo` to sysfs returns error
- Changes don't persist

**Solutions:**
1. Check if parameter is read-only: `ls -la /sys/module/nvidia_uvm/parameters/`
2. Use modprobe.conf for read-only parameters
3. Ensure no UVM processes running before `rmmod`
4. Reboot for guaranteed clean state

---

## Advanced Topics

### HMM (Heterogeneous Memory Management)

**Current Status:** Disabled (`uvm_disable_hmm=Y`)

**To Enable:**
```bash
# Check kernel support
cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm

# Enable via modprobe
echo "options nvidia-uvm uvm_disable_hmm=0" | sudo tee /etc/modprobe.d/nvidia-uvm.conf

# Reboot
sudo reboot
```

**Benefits:**
- Better Linux kernel integration
- Improved memory coherence
- More efficient memory management

**Requirements:**
- Kernel 5.13+ with HMM support
- Compatible GPU (H100 supported)
- ATS-capable platform

---

### ATS (Address Translation Services)

**Current Status:** Enabled (`uvm_ats_mode=1`)

**Impact:**
- Allows GPU to use CPU page tables
- Reduces TLB misses
- Improves performance for sparse access patterns

**Disable if issues:**
```bash
options nvidia-uvm uvm_ats_mode=0
```

---

### Page Table Location

**Current:** Auto (`uvm_page_table_location=(null)`)

**Options:**
- `vid` - Store in GPU video memory (VRAM)
- `sys` - Store in system memory

**Recommendation:**
- `vid` for workloads with frequent GPU memory access
- `sys` for large oversubscription scenarios

```bash
options nvidia-uvm uvm_page_table_location=vid
```

---

### Peer Copy Mode

**Current:** Physical addressing (`uvm_peer_copy=phys`)

**Options:**
- `phys` - Physical addressing for peer copies
- `virt` - Virtual addressing (Ampere+ GPUs)

**Impact:** Affects multi-GPU peer-to-peer transfers.

```bash
options nvidia-uvm uvm_peer_copy=virt
```

---

## Quick Reference

### View All Current Settings

```bash
#!/bin/bash
echo "=== NVIDIA UVM Configuration ==="
echo ""
for param in /sys/module/nvidia_uvm/parameters/*; do
    name=$(basename $param)
    value=$(cat $param 2>/dev/null || echo "N/A")
    printf "%-50s %s\n" "$name:" "$value"
done | sort
```

### Reset to Defaults

```bash
# Remove custom configuration
sudo rm /etc/modprobe.d/nvidia-uvm.conf

# Reload module
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

# Or reboot
sudo reboot
```

---

## References

- [NVIDIA UVM Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [UVM Driver Source](https://github.com/NVIDIA/open-gpu-kernel-modules)
- [Linux Kernel HMM](https://www.kernel.org/doc/html/latest/mm/hmm.html)

---

**Last Updated:** 2025-11-11
**System:** NVIDIA H100 | Driver 580.105.08 | CUDA 12.9
**Author:** Generated for co-processor-demo project
