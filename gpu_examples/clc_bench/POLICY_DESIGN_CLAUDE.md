# Minimal Extensible Scheduler Framework for CLC

## Design Philosophy

**Goal**: Minimal callback interface where policies control **when blocks steal work**, not what they steal.

**Key Insight**: The safe control point is deciding **whether to continue stealing**, not selecting arbitrary work IDs. Policies control which blocks participate in work stealing and when they exit.

## Core Interface (3 Callbacks)

```cpp
// Template-based policy interface - zero runtime overhead, compile-time dispatch
template<typename Policy>
struct SchedulerPolicy {
    // Policy-defined state (held by framework in __shared__)
    using State = typename Policy::State;

    // 1. Called once at kernel start - initialize policy state
    __device__ static void init(State& s);

    // 2. Called BEFORE submitting a steal request: should we try to steal?
    //    Returns: false = skip this steal attempt (safe), true = submit request
    //    CRITICAL: Decision must be uniform across CTA (evaluated by one thread, broadcast via __shared__)
    __device__ static bool should_try_steal(State& s);

    // 3. Called AFTER successful steal: should we continue to next iteration?
    //    Returns: false = exit loop, true = continue stealing
    //    Parameters: stolen_bx = the block ID we just stole
    //    CRITICAL: Decision must be uniform across CTA
    __device__ static bool keep_going_after_success(int stolen_bx, State& s);
};
```

**That's it.** Just 3 callbacks.

**Key Design Points:**
- State is defined by the policy (`Policy::State`) but **held by the framework** in `__shared__` memory
- All callbacks receive state by reference (no `__shared__ static` members in policy types)
- Policy decisions MUST be **uniform across the CTA** to avoid divergence/deadlock around barriers

---

## Why This is Minimal Yet Sufficient

### 1. **Safe by Construction**
   - Policy cannot violate CLC constraints (no access to raw CLC APIs)
   - Two safe control points: before submitting request, after successful steal
   - Cannot submit after failure (framework handles exit immediately)
   - Framework holds policy state in `__shared__` (avoids `__shared__ static` UB)
   - Framework ensures uniform control flow (evaluates policy in thread 0, broadcasts decision)

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
    struct State {};  // Empty state

    __device__ static void init(State& s) {
        // No state to initialize
    }

    __device__ static bool should_try_steal(State& s) {
        return true;  // Always submit steal request
    }

    __device__ static bool keep_going_after_success(int stolen_bx, State& s) {
        return true;  // Always continue
    }
};
```

### Pattern 2: Max Steals (Stop After N Successes)
```cpp
struct MaxStealsPolicy {
    struct State {
        int steals_done = 0;
        int max_steals = 8;  // Configurable limit
    };

    __device__ static void init(State& s) {
        // s.max_steals can be set by host/constructor if needed
        s.steals_done = 0;
    }

    __device__ static bool should_try_steal(State& s) {
        return s.steals_done < s.max_steals;
    }

    __device__ static bool keep_going_after_success(int stolen_bx, State& s) {
        s.steals_done++;  // Thread 0 updates; framework ensures uniform control
        return s.steals_done < s.max_steals;
    }
};
```

### Pattern 3: Iteration-Based Throttling
```cpp
struct EveryNthIterationPolicy {
    struct State {
        int iteration = 0;
        int N = 2;  // Only steal every Nth iteration
    };

    __device__ static void init(State& s) {
        s.iteration = 0;
    }

    __device__ static bool should_try_steal(State& s) {
        bool result = (s.iteration % s.N) == 0;
        s.iteration++;
        return result;
    }

    __device__ static bool keep_going_after_success(int stolen_bx, State& s) {
        return true;  // Continue (throttling happens in should_try_steal)
    }
};
```

### Pattern 4: Block-Based Policy (Different Behavior Per Block)
```cpp
struct FirstHalfOnly {
    struct State {
        int block_id;
        int half_blocks;
    };

    __device__ static void init(State& s) {
        s.block_id = blockIdx.x;
        s.half_blocks = gridDim.x / 2;
    }

    __device__ static bool should_try_steal(State& s) {
        // Only first half of blocks steal aggressively
        return s.block_id < s.half_blocks;
    }

    __device__ static bool keep_going_after_success(int stolen_bx, State& s) {
        return s.block_id < s.half_blocks;
    }
};
```

### Pattern 5: Composable Policies (AND Combination)
```cpp
// Combine multiple policies with AND logic
template<typename P1, typename P2>
struct AndPolicy {
    struct State {
        typename P1::State s1;
        typename P2::State s2;
    };

    __device__ static void init(State& s) {
        P1::init(s.s1);
        P2::init(s.s2);
    }

    __device__ static bool should_try_steal(State& s) {
        return P1::should_try_steal(s.s1) && P2::should_try_steal(s.s2);
    }

    __device__ static bool keep_going_after_success(int stolen_bx, State& s) {
        return P1::keep_going_after_success(stolen_bx, s.s1) &&
               P2::keep_going_after_success(stolen_bx, s.s2);
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
// ✅ CORRECT: Acquire fence BEFORE request, release fence AFTER decoding
ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire,
                                              ptx::space_shared,
                                              ptx::scope_cta);
ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
[...barrier wait...]
bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release,
                                              ptx::space_shared,
                                              ptx::scope_cta);

// ❌ WRONG: No fences - shared memory updates may not be visible
// ❌ WRONG: Fences in wrong order (release before request, acquire after)
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
**Rule**: Clusters require multicast version, cluster scopes, and local offset:

```cpp
// ✅ CORRECT: Cluster-aware kernel
__global__ __cluster_dims__(2, 1, 1) void kernel(...) {
    auto cluster = cg::this_cluster();

    // Sync cluster before first request
    cluster.sync();

    // Use multicast version with ONE cluster thread
    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire,
                                                      ptx::space_cluster,
                                                      ptx::scope_cluster);
        ptx::clusterlaunchcontrol_try_cancel_multicast(&result, &bar);
    }

    // Use scope_cluster for barrier ops
    ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cluster,
                                    ptx::space_shared, &bar, sizeof(uint4));
    while (!ptx::mbarrier_try_wait_parity(&bar, phase)) {}

    // Add local cluster offset to decoded result
    int hardware_cta_id = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
    int local_offset = cluster.block_index().x;
    int bx = hardware_cta_id + local_offset;

    ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release,
                                                  ptx::space_cluster,
                                                  ptx::scope_cluster);
}

// ❌ WRONG: Regular version in cluster kernel
__global__ __cluster_dims__(2, 1, 1) void kernel(...) {
    ptx::clusterlaunchcontrol_try_cancel(&result, &bar);  // Should use _multicast!
    // Missing: cluster sync, scope_cluster, local offset
}
```

### Why Policy Decisions Must Be Uniform and Placed Before CLC Operations

Given constraint #3 (no request after failure) and barrier synchronization requirements, the **only safe placement** for policy decisions is:

```cpp
while (true) {
    __syncthreads();

    // ✅ SAFE: Uniform decision BEFORE submitting request
    // Thread 0 evaluates, all threads read the same decision
    if (threadIdx.x == 0) {
        go = Policy::should_try_steal(pol_state) ? 1 : 0;
    }
    __syncthreads();

    if (!go) {
        break;  // All threads exit uniformly
    }

    // Submit CLC request (all threads participate in barrier)
    if (threadIdx.x == 0) {
        ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
        ptx::mbarrier_arrive_expect_tx(...);
    }

    // ALL threads must wait (uniform control flow)
    while (!ptx::mbarrier_try_wait_parity(&bar, phase)) {}

    bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
    if (!success) {
        // ❌ UNSAFE: Cannot put policy check here and continue loop
        //    because we observed failure - must exit immediately
        break;
    }

    // Uniform decision after success
    if (threadIdx.x == 0) {
        int bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x(result);
        go = Policy::keep_going_after_success(bx, pol_state) ? 1 : 0;
    }
    __syncthreads();

    if (!go) break;
}
```

**Critical requirements:**
1. **Uniform control flow**: All threads must take same path around barriers/waits
2. **Thread 0 decides**: Policy callbacks called by thread 0, result broadcast via `__shared__`
3. **Placement**: Decisions only at safe points (before request, after success)

**This is why the policy framework exists**: To provide a safe, high-level interface that respects all CLC hardware constraints (uniform control flow, proper barrier synchronization) while enabling flexible scheduling policies.

---

## Critical: Uniform Control Flow Requirement

**The #1 footgun**: Thread divergence around barriers causes **deadlock**.

### The Problem
CLC uses `mbarrier_try_wait_parity()` which requires **ALL threads** in the CTA to participate. If threads disagree on whether to enter the wait loop, you deadlock:

```cpp
// ❌ DEADLOCK: Threads diverge based on thread ID
if (threadIdx.x < 16) {
    // These 16 threads call should_try_steal() and might get true
    if (Policy::should_try_steal(state)) {
        // They enter the barrier wait...
        while (!ptx::mbarrier_try_wait_parity(&bar, phase)) {}
    }
} else {
    // These threads skip the wait → DEADLOCK
}

// ❌ DEADLOCK: Non-uniform policy state
__shared__ int per_thread_state[32];
if (Policy::should_try_steal_per_thread(per_thread_state[threadIdx.x])) {
    // Some threads enter, others don't → DEADLOCK
    while (!ptx::mbarrier_try_wait_parity(&bar, phase)) {}
}
```

### The Solution: Elect + Broadcast Pattern

```cpp
// ✅ CORRECT: Single decision, uniform control flow
__shared__ int go;                      // Shared decision flag
__shared__ typename Policy::State pol;  // Policy state in shared memory

while (true) {
    __syncthreads();

    // Step 1: Thread 0 makes the decision
    if (threadIdx.x == 0) {
        go = Policy::should_try_steal(pol) ? 1 : 0;
    }

    // Step 2: Broadcast to all threads
    __syncthreads();

    // Step 3: All threads take the same branch
    if (!go) break;  // Uniform exit

    // Step 4: All threads participate in CLC operations
    if (threadIdx.x == 0) {
        ptx::clusterlaunchcontrol_try_cancel(&result, &bar);
        ptx::mbarrier_arrive_expect_tx(...);
    }

    // ALL threads wait (uniform control flow)
    while (!ptx::mbarrier_try_wait_parity(&bar, phase)) {}

    // ... rest of steal logic ...
}
```

**Key principle**: Policy logic runs in **one thread**; all threads read the **same result** and execute the **same path**.

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

## Best Practices Summary

### DO ✅
1. **Define `Policy::State`** and let the framework hold it in `__shared__`
2. **Pass state by reference** to all callbacks
3. **Evaluate policies in thread 0**, broadcast via `__shared__` flag
4. **Use `__syncthreads()`** before and after policy decisions
5. **Place decisions at safe points**: before request, after success
6. **Use cluster scopes/fences** when `__cluster_dims__()` is set
7. **Check `__CUDA_ARCH__ >= 1000`** at compile time
8. **Test with different grid sizes** to catch uniformity bugs

### DON'T ❌
1. **Never use `__shared__ static`** inside policy types (UB)
2. **Never let threads diverge** around policy decisions
3. **Never call policy functions from multiple threads** without synchronization
4. **Never submit requests after observing failure** (framework handles this)
5. **Never use per-thread state** that could cause divergence
6. **Never skip `__syncthreads()`** around policy decision points
7. **Never access `result` before barrier completes**
8. **Never use regular CLC calls** in cluster kernels (use `_multicast`)

---

## Conclusion

This minimal template-based design:
- ✅ **3 callbacks only** (init, should_try_steal, keep_going_after_success)
- ✅ **Safe by construction** (policies cannot violate CLC constraints)
- ✅ **Framework-held state** (policies define `State`, framework holds it in `__shared__`)
- ✅ **Uniform control flow** (thread 0 evaluates, all threads execute same path)
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
- Policy cannot cause thread divergence (framework enforces uniform control flow)
- Skipping steal attempts is always safe (just do local work)

**Advantages**:
- Compile-time optimization and inlining
- Type safety with no runtime overhead
- Better error messages at compile time
- Correctness guaranteed by construction
- No `__shared__ static` UB or uniformity pitfalls

**Key Principle**: Separate concerns cleanly:
- **Hardware (CLC)**: Validates and provides available work (safe, atomic)
- **Policy**: Controls when to steal and when to stop (flexible, expressive)
- **Framework**: Enforces all CLC constraints and safety invariants (safe wrapper, uniform control)

This is the true "sched_ext for GPU scheduling" - minimal, type-safe, extensible, **safe by construction**, and powerful.
