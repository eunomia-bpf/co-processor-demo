# Qwen3 CUDA Inference with CLC Policy Framework

This directory contains a modified version of the Qwen3 CUDA inference engine that integrates the CLC (Cooperative Launch Control) policy framework for the `matmul_kernel`.

## Overview

The CLC framework allows runtime policy-based scheduling of CUDA kernels using nvJitLink. The matmul kernel can now be controlled by different scheduling policies without recompilation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Runtime Linking (nvJitLink)              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │ User Kernel PTX  │  │ Wrapper Kernel   │  │ Policy    │ │
│  │ (matmul_kernel)  │  │ (CLC Scheduler)  │  │ Logic     │ │
│  │                  │  │                  │  │           │ │
│  │ Auto-extracted   │  │ Work-stealing    │  │ Greedy/   │ │
│  │ from binary      │  │ + Policy calls   │  │ MaxSteals │ │
│  └──────────────────┘  └──────────────────┘  └───────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Files

### Core Components

- **runcu.cu** - Modified Qwen3 inference engine with CLC support
  - `#ifdef USE_POLICY_FRAMEWORK` enables policy mode
  - `matmul_kernel_impl` is now a `__device__` function
  - Uses `PolicyFramework::launch_with_shared()` for execution

- **wrapper_kernel.cu** - CLC scheduler wrapper for matmul
  - Implements elect-and-broadcast pattern
  - Thread 0 evaluates policy, broadcasts to all threads
  - Maintains uniform control flow (CLC requirement)

- **policy_framework.h** - nvJitLink integration framework
  - Auto-extracts PTX from binary using cuobjdump
  - Links user kernel + wrapper + policy at runtime
  - Supports shared memory in kernel launches

### Policies

- **policy_greedy.cu** - Always execute (baseline)
  - Stateless policy
  - Mimics default CUDA behavior
  - Best for uniform workloads

- **policy_maxsteals.cu** - Limited execution (max 8 times)
  - Stateful policy with execution counter
  - Prevents excessive work stealing
  - Useful for load balancing

## Build

```bash
# Build all components (original + policy versions)
make policy-all

# Build individual targets
make runcu              # Original version (no policy)
make runcu_policy       # Policy framework version
make wrapper_kernel.ptx # CLC scheduler wrapper
make policy_greedy.ptx  # Greedy policy
make policy_maxsteals.ptx # MaxSteals policy

# Clean
make clean     # Remove binaries
make clean-all # Remove binaries + PTX files
```

## Usage

### Run with different policies

```bash
# GreedyPolicy (always execute)
make run-greedy

# MaxStealsPolicy (limit to 8 executions)
make run-maxsteals

# Original version (no policy framework)
make run-original

# Compare all versions
make run-all
```

### Manual execution

```bash
# GreedyPolicy
WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx \
POLICY_PTX_PATH=./policy_greedy.ptx \
./runcu_policy Qwen3-0.6B-FP32.gguf -q "What is CUDA?" -r 0

# MaxStealsPolicy
WRAPPER_KERNEL_PATH=./wrapper_kernel.ptx \
POLICY_PTX_PATH=./policy_maxsteals.ptx \
./runcu_policy Qwen3-0.6B-FP32.gguf -q "What is CUDA?" -r 0
```

## Performance Results

Based on test runs with the query "What is CUDA?":

| Version         | TTFT (s) | Tokens/sec | Notes                          |
|-----------------|----------|------------|--------------------------------|
| Original        | 0.294    | 54.65      | Direct kernel launch           |
| GreedyPolicy    | 0.348    | 49.31      | +18% overhead, always execute  |
| MaxStealsPolicy | 0.347    | 49.08      | +18% overhead, limited to 8x   |

The ~18% overhead comes from:
1. nvJitLink runtime linking (one-time cost)
2. Policy evaluation in wrapper kernel
3. Elect-and-broadcast pattern overhead

## How It Works

### 1. Compilation Phase

```bash
nvcc -DUSE_POLICY_FRAMEWORK --keep-device-functions runcu.cu -o runcu_policy
```

- Compiles `matmul_kernel_impl` as `__device__` function
- Preserves device functions in binary for PTX extraction

### 2. Runtime Linking

When `runcu_policy` starts:

1. **PTX Extraction**: cuobjdump extracts user kernel PTX from binary
2. **Load Components**: Loads wrapper kernel and policy from environment variables
3. **nvJitLink**: Links all three PTX modules together
4. **Module Load**: Loads linked CUBIN into CUDA context

### 3. Kernel Execution

For each matmul operation:

1. **Thread 0 evaluates policy**: `Policy_should_try(state, blockIdx.x)`
2. **Broadcast decision**: Store result in shared memory
3. **All threads sync**: `__syncthreads()`
4. **Uniform execution**: All threads follow the same branch
5. **Execute kernel**: If policy allows, call `matmul_kernel_impl()`

## Key Features

✓ **Zero-overhead when disabled** - Original version unchanged
✓ **Runtime policy switching** - No recompilation needed
✓ **Automatic PTX extraction** - No manual PTX generation required
✓ **Shared memory support** - Full compatibility with original kernel
✓ **CLC-compliant** - Follows elect-and-broadcast pattern

## Implementation Notes

### Shared Memory Handling

The original `matmul_kernel` uses dynamic shared memory. The policy framework was extended with `launch_with_shared()` to support this:

```cpp
fw->launch_with_shared("matmul_kernel", gridDim, blockDim,
                       shared_mem, stream, xout, x, w, n, d);
```

### Policy State

Policies maintain state in shared memory (64 bytes allocated). This allows:
- GreedyPolicy: No state needed (stateless)
- MaxStealsPolicy: Execution counter per block

### Uniform Control Flow

CLC requires all threads in a block to take the same execution path. The elect-and-broadcast pattern ensures this:

```cuda
// Thread 0 evaluates
if (threadIdx.x == 0) {
    go = Policy_should_try(policy_state, blockIdx.x) ? 1 : 0;
}
__syncthreads();

// All threads follow
if (go) {
    matmul_kernel_impl(...);
}
```

## Limitations

1. **Simplified CLC**: Current implementation doesn't do full work-stealing (would require updating block indices dynamically)
2. **Policy Overhead**: ~18% performance overhead from policy evaluation
3. **One-time Setup Cost**: nvJitLink initialization adds ~300ms to startup

## Future Enhancements

- [ ] Implement full CLC work-stealing for matmul
- [ ] Add more sophisticated policies (e.g., adaptive, priority-based)
- [ ] Optimize policy evaluation overhead
- [ ] Add performance profiling hooks
- [ ] Support multiple kernel policies in single binary

## References

- Based on CLC framework: `gpu_examples/nvjitlink_policy_clc/`
- NVIDIA nvJitLink documentation
- CUDA Cooperative Groups documentation

## Help

```bash
make help-policy  # Show detailed usage information
```
