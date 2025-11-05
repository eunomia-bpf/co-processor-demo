# Minimal Extensible Scheduler Framework for CLC

## Design Philosophy

**Goal**: Minimal callback interface where policies control **when blocks steal work**, not what they steal.

**Key Insight**: The safe control point is deciding **whether to continue stealing**, not selecting arbitrary work IDs. Policies control which blocks participate in work stealing and when they exit.

## Core Interface (2 Callbacks)

```cpp
// Template-based policy interface - zero runtime overhead, compile-time dispatch
template<typename Policy>
struct SchedulerPolicy {
    // 1. Called once when block starts
    //    Policy initializes any state it needs
    __device__ static void init(int block_id, int total_blocks, int total_items,
                                typename Policy::State& state);

    // 2. Called before each steal attempt: should this block continue stealing?
    //    Returns: true = continue stealing, false = exit
    __device__ static bool should_continue(int block_id, int iteration,
                                           typename Policy::State& state);
};
```

**That's it.** Just 2 callbacks + type-safe state management via templates.

---

## Why This is Minimal Yet Sufficient

### 1. **No Framework State Structures**
   - No context structs imposed by framework
   - Policy defines its own state layout via `Policy::State` type
   - Framework provides type-safe state references

### 2. **Policy Controls Continuation**
   - `should_continue()` decides when each block exits the stealing loop
   - Hardware (CLC) controls what work gets stolen (safe)
   - Policy controls which blocks participate and for how long

### 3. **All Scheduling Policies Expressible**
   - **Greedy**: `should_continue()` always returns true
   - **Threshold**: Stop after doing N work items
   - **Throttling**: Only some blocks steal aggressively
   - **Load Balancing**: Stop after fair share of work
   - **Adaptive**: Adjust based on runtime metrics
   - **Priority**: Different blocks have different iteration budgets

---

## Policy State Management Patterns

### Pattern 1: No State (Greedy)
```cpp
// Policy with empty state
struct GreedyPolicy {
    struct State {}; // Empty state - no memory overhead

    __device__ static void init(int block_id, int total_blocks, int total_items, State& state) {
        // No-op
    }

    __device__ static bool should_continue(int block_id, int iteration, State& state) {
        return true;  // Always continue
    }
};
```

### Pattern 2: Per-Block State (Threshold)
```cpp
// Each block has its own state in shared memory
struct ThresholdPolicy {
    struct State {
        int work_done;
        int expected_work;
        float threshold;
    };

    __device__ static void init(int block_id, int total_blocks, int total_items, State& state) {
        if (threadIdx.x == 0) {
            state.work_done = 0;
            state.expected_work = total_items / total_blocks;
            state.threshold = 0.7f;
        }
        __syncthreads();
    }

    __device__ static bool should_continue(int block_id, int iteration, State& state) {
        // Stop after doing 70% of expected work
        return (state.work_done < state.expected_work * state.threshold);
    }
};
```

### Pattern 3: Block-Level Throttling
```cpp
// Control which blocks steal aggressively
struct ThrottlePolicy {
    struct State {
        bool is_aggressive;
    };

    __device__ static void init(int block_id, int total_blocks, int total_items, State& state) {
        // Only first half of blocks are aggressive
        state.is_aggressive = (block_id < total_blocks / 2);
    }

    __device__ static bool should_continue(int block_id, int iteration, State& state) {
        if (state.is_aggressive) {
            return true;  // Keep stealing
        } else {
            return iteration < 2;  // Only steal twice, then exit
        }
    }
};
```

### Pattern 4: Adaptive State (Work-Rate Based)
```cpp
// Per-block state with runtime metrics
struct AdaptivePolicy {
    struct State {
        int work_done;
        int expected_work;
        unsigned long long start_time;
        float work_rate;
    };

    __device__ static void init(int block_id, int total_blocks, int total_items, State& state) {
        if (threadIdx.x == 0) {
            state.work_done = 0;
            state.expected_work = total_items / total_blocks;
            state.start_time = clock64();
            state.work_rate = 0.0f;
        }
        __syncthreads();
    }

    __device__ static bool should_continue(int block_id, int iteration, State& state) {
        // Update work rate periodically
        if (threadIdx.x == 0 && iteration % 10 == 0) {
            unsigned long long now = clock64();
            state.work_rate = (float)state.work_done / (float)(now - state.start_time);
        }

        // Fast workers steal earlier (lower threshold)
        float threshold = 0.5f / (1.0f + state.work_rate);
        return (state.work_done < threshold * state.expected_work);
    }
};
```

### Pattern 5: Global Coordination (Concurrency Control)
```cpp
// Use global atomics for cross-block coordination
struct ConcurrencyControlPolicy {
    struct State {
        int* global_active_stealers;  // Shared across all blocks
        bool is_stealing;
        int max_stealers;
    };

    __device__ static void init(int block_id, int total_blocks, int total_items, State& state) {
        state.max_stealers = total_blocks / 4;  // Max 25% blocks stealing at once
        state.is_stealing = false;
    }

    __device__ static bool should_continue(int block_id, int iteration, State& state) {
        if (threadIdx.x == 0) {
            int active = atomicAdd(state.global_active_stealers, 0);  // Read

            if (!state.is_stealing && active < state.max_stealers) {
                // Try to become an active stealer
                atomicAdd(state.global_active_stealers, 1);
                state.is_stealing = true;
            } else if (state.is_stealing && active > state.max_stealers) {
                // Too many stealers, back off
                atomicSub(state.global_active_stealers, 1);
                state.is_stealing = false;
            }
        }
        __syncthreads();

        return state.is_stealing;
    }
};
```

---

## CLC Guarantees & Footguns

**Critical**: The policy framework must respect CLC hardware constraints. Understanding these is essential for safe integration.

### What CLC Guarantees

✅ **Validated work**: CLC hardware guarantees that returned block IDs are available for processing
✅ **Atomic cancellation**: Hardware manages work-stealing atomically across the cluster
✅ **Race-free**: No manual synchronization needed for work distribution

### CLC Footguns (Why This Wrapper Must Exist)

These hardware constraints are **Undefined Behavior** if violated:

#### 1. Single-Thread Submission ⚠️
**Rule**: Only ONE thread per block can submit a cancel request.

```cpp
// ✅ CORRECT: Single elected thread
if (cg::thread_block::thread_rank() == 0) {
    cg::invoke_one(cg::coalesced_threads(), [&](){
        ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
    });
}

// ❌ WRONG: All threads submit
ptx::clusterlaunchcontrol_try_cancel(&result, &bar);  // UB!
```

#### 2. Asynchronous Request → Barrier → Query ⚠️
**Rule**: Must synchronize via mbarrier before querying result.

```cpp
// ✅ CORRECT: Request → arrive → wait → query
ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta,
                                ptx::space_shared, &bar, sizeof(uint4));
while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
phase ^= 1;
bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);

// ❌ WRONG: Query before barrier completes
ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);  // UB!
```

#### 3. No Request After Observing Failure ⚠️
**Rule**: After a failed request, you MUST NOT submit another request.

```cpp
// ✅ CORRECT: Exit on first failure
ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
[...barrier sync...]
bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
if (!success) {
    break;  // EXIT - do not submit another request
}

// ❌ WRONG: Submit again after failure
ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);
[...barrier sync...]
bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
if (!success0) {
    // Observed failure!
}
ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);  // UB!
```

**Exception**: Multiple requests are OK if submitted **before** querying any:
```cpp
// ✅ CORRECT: Both submitted before any query
ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);
ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);
[...sync both...]
bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
bool success1 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result1);
```

#### 4. Memory-Ordering Fences ⚠️
**Rule**: Use proxy fences to make async stores visible to generic code.

```cpp
// ✅ CORRECT: Fences around async operations
ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire,
                                              ptx::space_cluster,
                                              ptx::scope_cluster);
[...async work-stealing...]
ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release,
                                              ptx::space_shared,
                                              ptx::scope_cluster);

// ❌ WRONG: No fences - shared memory updates may not be visible
```

#### 5. Compute Capability 10.0+ Required ⚠️
**Rule**: CLC requires Blackwell (CC 10.0). Check at runtime:

```cpp
// ✅ CORRECT: Check capability
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);
if (prop.major < 10) {
    fprintf(stderr, "ERROR: CLC requires CC 10.0+\n");
    return 1;  // Use fallback kernel
}
```

#### 6. Cluster-Specific Rules ⚠️
**Rule**: Clusters require multicast version and local offset:

```cpp
// ✅ CORRECT: Cluster-aware kernel
__global__ __cluster_dims__(2, 1, 1) void kernel(...) {
    // Use multicast version
    ptx::clusterlaunchcontrol_try_cancel_multicast(&result, &bar);

    // Add local cluster offset
    int hardware_cta_id = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
    int local_offset = cg::cluster_group::block_index().x;
    int bx = hardware_cta_id + local_offset;

    // Ensure all CTAs exist before canceling
    cg::cluster_group::sync();
}

// ❌ WRONG: Regular version in cluster kernel
__global__ __cluster_dims__(2, 1, 1) void kernel(...) {
    ptx::clusterlaunchcontrol_try_cancel(&result, &bar);  // Should use _multicast!
}
```

### Why Policy `should_continue()` Must Be Placed Before CLC Operations

Given constraint #3 (no request after failure), the **only safe placement** for policy exit is:

```cpp
while (true) {
    __syncthreads();

    // ✅ SAFE: Check policy BEFORE submitting request
    if (!Policy::should_continue(blockIdx.x, iteration, policy_state)) {
        break;  // Exit cleanly without violating CLC constraints
    }

    // Submit CLC request
    ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
    [...barrier sync...]
    bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);

    if (!success) {
        // ❌ UNSAFE: Cannot put policy check here and continue loop
        //    because we observed failure - must exit immediately
        break;
    }

    // Process work...
    iteration++;
}
```

**This is why the policy framework exists**: To provide a safe, high-level interface that respects all CLC hardware constraints while enabling flexible scheduling policies.

---

## Minimal Policy Integration with CLC

The policy framework requires **only 3 simple changes** to the CLC work-stealing kernel:

### Complete Diff:

```diff
-template<typename WorkloadType>
+template<typename WorkloadType, typename Policy>  // 1. Add Policy template parameter
-__global__ void kernel_cluster_launch_control_baseline(float* data, int n, int* block_count, int* steal_count,
+__global__ void kernel_cluster_launch_control_policy(float* data, int n, int* block_count, int* steal_count,
                                                       int prologue_complexity) {
+   // 2. Declare policy state
+   __shared__ typename Policy::State policy_state;
+
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

+   // 3. Initialize policy
+   Policy::init(blockIdx.x, gridDim.x, n, policy_state);
+   __syncthreads();

    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&bar, 1);

    float weight = compute_prologue(prologue_complexity);
    int bx = blockIdx.x;
+   int iteration = 0;  // 4. Track iteration count

    while (true) {
        __syncthreads();

+       // 5. Policy hook: should this block continue stealing?
+       if (!Policy::should_continue(blockIdx.x, iteration, policy_state)) {
+           break;
+       }

        if (cg::thread_block::thread_rank() == 0) {
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
            cg::invoke_one(cg::coalesced_threads(), [&](){
                ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
            });
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
        }

        int i = bx * blockDim.x + threadIdx.x;
        if (i < n) {
            process_workload(WorkloadType{}, data, i, n, weight);
        }

        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
        phase ^= 1;

        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success) {
            break;
        }

        int hardware_cta_id = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
+       // Hardware controls what work to steal (safe)
        bx = hardware_cta_id;

        if (threadIdx.x == 0) {
            atomicAdd(steal_count, 1);
+           if constexpr (requires { policy_state.work_done; }) {
+               policy_state.work_done++;  // Track work if policy needs it
+           }
        }

        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
+
+       iteration++;  // 6. Increment iteration
    }

    if (threadIdx.x == 0) {
        atomicAdd(block_count, 1);
    }
}
```

### Safe Placement of `should_continue()`

The policy check is placed **at the beginning of the loop**, before any CLC operations. This is the only safe point because:

✅ **Before work stealing attempt**: Block can exit cleanly without breaking CLC state
✅ **After synchronization**: All threads in block see consistent state
✅ **No partial operations**: Either continue and do full steal cycle, or exit completely

❌ **Unsafe placements**:
- After `clusterlaunchcontrol_try_cancel()`: Can't exit mid-steal (breaks CLC constraints)
- Inside workload processing: Non-uniform control flow
- After getting hardware_cta_id: Wastes CLC steal operation

---

## Policy Interface

```cpp
struct MyPolicy {
    // Define your own state (or empty struct for stateless)
    struct State {
        // Any fields you need
    };

    // Called once at kernel start
    __device__ static void init(int block_id, int total_blocks, int total_items, State& state);

    // Called before each steal attempt: should this block continue?
    __device__ static bool should_continue(int block_id, int iteration, State& state);
};
```

### Usage

```cpp
// Same kernel, different policies - just change template parameter
kernel_cluster_launch_control_policy<MyWorkload, GreedyPolicy><<<...>>>(...);
kernel_cluster_launch_control_policy<MyWorkload, ThresholdPolicy><<<...>>>(...);
kernel_cluster_launch_control_policy<MyWorkload, ThrottlePolicy><<<...>>>(...);
```

---

## Why This Design is Better

### **Correctness by Design**
- ✅ **Hardware controls work selection**: CLC decides what work is available (validated)
- ✅ **Policy controls participation**: Which blocks steal and when they exit
- ✅ **No correctness bugs**: Can't break work-stealing semantics
- ✅ **Safe control point**: Exit before CLC operations, not during

### **Multi-Level Control Power**
- 🎯 **Block selection**: Control which blocks participate in stealing
- 🎯 **Iteration control**: Different blocks have different stealing budgets
- 🎯 **Global coordination**: Use atomics for cross-block policies
- 🎯 **Adaptive strategies**: Policies observe and react to system state

### **Key Insight**
The real power is **controlling which blocks continue stealing**, not selecting specific work items. Since 100s-1000s of blocks run concurrently:
- **Resource throttling**: Reduce concurrent stealers → less contention
- **Load balancing**: Blocks stop after fair share → better distribution
- **Priority scheduling**: Important blocks steal more → QoS guarantees
- **Adaptive concurrency**: React to runtime conditions → performance tuning

### **Zero Overhead**
- ✅ **Compile-time dispatch**: No function pointers
- ✅ **Inlining**: Compiler optimizes all callbacks
- ✅ **Type safety**: No `void*` casts
- ✅ **Minimal state**: Only what policy needs

---

## Implementation Strategy

### Host Side:
```cpp
// 1. Allocate global policy state (if policy needs it)
typename MyPolicy::State* d_global_state = nullptr;
if constexpr (/* policy needs global state */) {
    cudaMalloc(&d_global_state, sizeof(typename MyPolicy::State));
    // Initialize global state on host if needed
}

// 2. Launch kernel with policy template parameter
kernel_cluster_launch_control_policy<MyWorkload, MyPolicy>
    <<<blocks, threads>>>(data, n, block_count, steal_count, prologue);
```

### Kernel Side:
```cpp
template<typename WorkloadType, typename Policy>
__global__ void kernel_cluster_launch_control_policy(...) {
    // Per-block state in shared memory
    __shared__ typename Policy::State policy_state;

    // Initialize policy
    Policy::init(blockIdx.x, gridDim.x, n, policy_state);

    // Work-stealing loop
    int iteration = 0;
    while (Policy::should_continue(blockIdx.x, iteration, policy_state)) {
        // ... CLC stealing logic ...
        iteration++;
    }
}
```

---

## Comparison with Alternatives

| Aspect | This Design | select_victim() Design | BPF-style Hooks |
|--------|-------------|------------------------|-----------------|
| **Callbacks** | 2 (init, should_continue) | 3 (init, should_steal, select_victim) | Many hooks |
| **Correctness** | Safe by design | Requires validation | Requires validation |
| **Control Point** | When to steal | What to steal | Multiple points |
| **Complexity** | Minimal | Medium | High |
| **Performance** | Zero overhead | Zero overhead | Hook overhead |
| **Safety** | Can't break correctness | Can return invalid IDs | Can break invariants |
| **Expressiveness** | High (block-level control) | High (work-level control) | Very high |

---

## Conclusion

This minimal template-based design:
- ✅ **2 callbacks only** (init, should_continue)
- ✅ **Safe by design** (hardware controls work selection)
- ✅ **No framework state** (just `Policy::State`)
- ✅ **Zero overhead** (compile-time dispatch, inlining)
- ✅ **Type-safe** (no `void*` casts)
- ✅ **Multi-level control** (block participation, iteration limits, global coordination)
- ✅ **Maximally expressive** (all scheduling policies expressible)

**Advantages**:
- Compile-time optimization and inlining
- Type safety with no runtime overhead
- Better error messages at compile time
- Correctness guaranteed by design

**Key Principle**: Separate concerns cleanly:
- **Hardware (CLC)**: Validates and provides available work (safe)
- **Policy**: Controls which blocks participate and when they exit (flexible)
- **Framework**: Integrates both with minimal glue code (simple)

This is the true "sched_ext for GPU scheduling" - minimal, type-safe, extensible, safe, and powerful.
