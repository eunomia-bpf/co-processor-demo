# Minimal Extensible Scheduler Framework for CLC

## Design Philosophy

**Goal**: Minimal callback interface where policies control **when blocks steal work**, not what they steal.

**Key Insight**: The safe control point is deciding **whether to continue stealing**, not selecting arbitrary work IDs. Policies control which blocks participate in work stealing and when they exit.

## Core Interface (3 Callbacks)

```cpp
// Template-based policy interface - zero runtime overhead, compile-time dispatch
template<typename Policy>
struct SchedulerPolicy {
    // 1. Called once at kernel start - initialize policy state
    __device__ static void init();

    // 2. Called BEFORE submitting a steal request: should we try to steal?
    //    Returns: false = skip this steal attempt (safe), true = submit request
    __device__ static bool should_try_steal();

    // 3. Called AFTER successful steal: should we continue to next iteration?
    //    Returns: false = exit loop, true = continue stealing
    //    Parameters: stolen_bx = the block ID we just stole
    __device__ static bool keep_going_after_success(int stolen_bx);
};
```

**That's it.** Just 3 callbacks. Policies manage their own state (in shared memory, registers, or global memory) if needed.

---

## Why This is Minimal Yet Sufficient

### 1. **Safe by Construction**
   - Policy cannot violate CLC constraints (no access to raw CLC APIs)
   - Two safe control points: before submitting request, after successful steal
   - Cannot submit after failure (framework handles exit immediately)
   - Policies manage their own state (framework doesn't impose structure)

### 2. **Policy Controls Participation**
   - `should_try_steal()` decides whether to submit a steal request (can skip safely)
   - `keep_going_after_success()` decides whether to continue after successful steal
   - Hardware (CLC) controls what work gets stolen (validated by hardware)
   - Policy controls which blocks participate and for how long

### 3. **All Scheduling Policies Expressible**
   - **Greedy**: Both callbacks always return true
   - **Max Steals**: Stop after N successful steals
   - **Time Budget**: Stop after T nanoseconds (using `clock64()`)
   - **Throttling**: Skip steal attempts on some iterations
   - **Load Balancing**: Stop after fair share of work
   - **Composable**: AND/OR multiple policies together

---

## Policy Examples

### Pattern 1: Greedy (Always Steal)
```cpp
struct GreedyPolicy {
    __device__ static void init() {
        // No state to initialize
    }

    __device__ static bool should_try_steal() {
        return true;  // Always submit steal request
    }

    __device__ static bool keep_going_after_success(int stolen_bx) {
        return true;  // Always continue
    }
};
```

### Pattern 2: Max Steals (Stop After N Successes)
```cpp
struct MaxStealsPolicy {
    __shared__ static int steals_done;  // Track successes in shared memory
    static constexpr int max_steals = 8;

    __device__ static void init() {
        if (threadIdx.x == 0) {
            steals_done = 0;
        }
        __syncthreads();
    }

    __device__ static bool should_try_steal() {
        return steals_done < max_steals;
    }

    __device__ static bool keep_going_after_success(int stolen_bx) {
        if (threadIdx.x == 0) {
            atomicAdd(&steals_done, 1);
        }
        __syncthreads();
        return steals_done < max_steals;
    }
};
```

### Pattern 3: Iteration-Based Throttling
```cpp
struct EveryNthIterationPolicy {
    __shared__ static int iteration;
    static constexpr int N = 2;  // Only steal every 2nd iteration

    __device__ static void init() {
        if (threadIdx.x == 0) {
            iteration = 0;
        }
        __syncthreads();
    }

    __device__ static bool should_try_steal() {
        bool result = (iteration % N) == 0;
        if (threadIdx.x == 0) iteration++;
        __syncthreads();
        return result;
    }

    __device__ static bool keep_going_after_success(int stolen_bx) {
        return true;  // Continue (throttling happens in should_try_steal)
    }
};
```

### Pattern 4: Block-Based Policy (Different Behavior Per Block)
```cpp
struct FirstHalfOnly {
    __device__ static void init() {
        // No state to initialize
    }

    __device__ static bool should_try_steal() {
        // Only first half of blocks steal aggressively
        return blockIdx.x < (gridDim.x / 2);
    }

    __device__ static bool keep_going_after_success(int stolen_bx) {
        return blockIdx.x < (gridDim.x / 2);
    }
};
```

### Pattern 5: Composable Policies (AND Combination)
```cpp
// Combine multiple policies with AND logic
template<typename P1, typename P2>
struct AndPolicy {
    __device__ static void init() {
        P1::init();
        P2::init();
    }

    __device__ static bool should_try_steal() {
        return P1::should_try_steal() && P2::should_try_steal();
    }

    __device__ static bool keep_going_after_success(int stolen_bx) {
        return P1::keep_going_after_success(stolen_bx) &&
               P2::keep_going_after_success(stolen_bx);
    }
};

// Example: Only first half of blocks + max 8 steals each
using SelectiveThrottled = AndPolicy<FirstHalfOnly, MaxStealsPolicy>;
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
- ✅ **3 callbacks only** (init, should_try_steal, keep_going_after_success)
- ✅ **Safe by construction** (policies cannot violate CLC constraints)
- ✅ **No framework state** (policies manage their own state)
- ✅ **Zero overhead** (compile-time dispatch, inlining)
- ✅ **Type-safe** (no `void*` casts, no raw CLC API exposure)
- ✅ **Two safe control points** (before submit, after success)
- ✅ **Composable** (AND/OR multiple policies together)
- ✅ **Maximally expressive** (all scheduling policies expressible)

**Safety Guarantees**:
- Policy cannot submit after observing failure (framework exits immediately)
- Policy cannot decode block IDs on failure (framework only calls callback on success)
- Policy cannot use multiple submitters (framework uses `invoke_one`)
- Policy cannot skip fences/barriers (framework manages all CLC mechanics)
- Skipping steal attempts is always safe (just do local work)

**Advantages**:
- Compile-time optimization and inlining
- Type safety with no runtime overhead
- Better error messages at compile time
- Correctness guaranteed by construction

**Key Principle**: Separate concerns cleanly:
- **Hardware (CLC)**: Validates and provides available work (safe, atomic)
- **Policy**: Controls when to steal and when to stop (flexible, expressive)
- **Framework**: Enforces all CLC constraints and safety invariants (safe wrapper)

This is the true "sched_ext for GPU scheduling" - minimal, type-safe, extensible, **safe by construction**, and powerful.
