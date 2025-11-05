# Minimal Extensible Scheduler Framework for CLC

## Design Philosophy

**Goal**: Minimal callback interface where policies are self-contained and manage their own state.

No framework-imposed context structures. Policies define what state they need and how to synchronize it.

## Core Interface (3 Callbacks)

```cpp
// Template-based policy interface - zero runtime overhead, compile-time dispatch
template<typename Policy>
struct SchedOps {
    // 1. Called once when block starts
    //    Policy initializes any state it needs (shared mem, global mem, etc.)
    __device__ static void init(int block_id, int total_blocks, int total_items,
                                typename Policy::State& state) {
        Policy::init(block_id, total_blocks, total_items, state);
    }

    // 2. Called in work-stealing loop to decide: should I try to steal?
    //    Returns: true = attempt steal, false = exit
    __device__ static bool should_steal(int block_id, int iteration,
                                        typename Policy::State& state) {
        return Policy::should_steal(block_id, iteration, state);
    }

    // 3. Called when should_steal() returns true
    //    Returns: target block ID to steal from, or -1 for hardware default
    __device__ static int select_victim(int block_id, int iteration,
                                        typename Policy::State& state) {
        return Policy::select_victim(block_id, iteration, state);
    }
};
```

**That's it.** Just 3 callbacks + type-safe state management via templates.

---

## Why This is Minimal Yet Sufficient

### 1. **No Framework State Structures**
   - No `SchedBlockCtx`, no `SchedGlobalCtx`
   - Policy defines its own state layout via `Policy::State` type
   - Framework provides type-safe state references (no `void*`)

### 2. **Policy Controls Synchronization**
   - If policy needs atomics → it uses them
   - If policy needs shared memory → it declares it
   - If policy needs global memory → it allocates it
   - Framework doesn't impose synchronization model

### 3. **All Policies Expressible**
   - **Greedy**: `should_steal()` always returns true, no state needed
   - **Threshold**: State = work counter, check threshold in `should_steal()`
   - **Locality**: State = group ID, implement logic in `select_victim()`
   - **Adaptive**: State = work rate, update in `should_steal()`
   - **Priority**: State = priority queue, manage in all callbacks

---

## Policy State Management Patterns

### Pattern 1: No State (Greedy)
```cpp
// Policy with empty state
struct GreedyPolicy {
    struct State {}; // Empty state - no memory overhead

    __device__ static void init(int bid, int total_blocks, int total_items, State& state) {
        // Nothing to do
    }

    __device__ static bool should_steal(int bid, int iter, State& state) {
        return true;
    }

    __device__ static int select_victim(int bid, int iter, State& state) {
        return -1; // Hardware chooses
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

    __device__ static void init(int bid, int total_blocks, int total_items, State& state) {
        if (threadIdx.x == 0) {
            state.work_done = 0;
            state.expected_work = total_items / total_blocks;
            state.threshold = 0.7f;
        }
        __syncthreads();
    }

    __device__ static bool should_steal(int bid, int iter, State& state) {
        return (state.work_done >= state.expected_work * state.threshold);
    }

    __device__ static int select_victim(int bid, int iter, State& state) {
        return -1; // Let hardware choose
    }
};
```

### Pattern 3: Global Shared State (Locality)
```cpp
// All blocks share global state for coordination
struct LocalityPolicy {
    struct State {
        int* block_groups;     // Device memory: group ID per block
        int group_size;
        int my_group;          // Cached local copy
    };

    __device__ static void init(int bid, int total_blocks, int total_items, State& state) {
        if (threadIdx.x == 0) {
            state.block_groups[bid] = bid / state.group_size;
            state.my_group = state.block_groups[bid];
        }
        __syncthreads();
    }

    __device__ static bool should_steal(int bid, int iter, State& state) {
        return true;
    }

    __device__ static int select_victim(int bid, int iter, State& state) {
        // Try to find victim in same group
        for (int i = 0; i < state.group_size; i++) {
            int candidate = state.my_group * state.group_size + i;
            if (candidate != bid) return candidate;
        }
        return -1;
    }
};
```

### Pattern 4: Adaptive State (Work-Rate)
```cpp
// Per-block state with runtime metrics
struct AdaptivePolicy {
    struct State {
        int work_done;
        int expected_work;
        unsigned long long start_time;
        float work_rate;
    };

    __device__ static void init(int bid, int total_blocks, int total_items, State& state) {
        if (threadIdx.x == 0) {
            state.work_done = 0;
            state.expected_work = total_items / total_blocks;
            state.start_time = clock64();
            state.work_rate = 0.0f;
        }
        __syncthreads();
    }

    __device__ static bool should_steal(int bid, int iter, State& state) {
        // Update work rate periodically
        if (threadIdx.x == 0 && iter % 10 == 0) {
            unsigned long long now = clock64();
            state.work_rate = (float)state.work_done / (float)(now - state.start_time);
        }

        // Fast workers steal earlier (lower threshold)
        float threshold = 0.5f / (1.0f + state.work_rate);
        return (state.work_done >= threshold * state.expected_work);
    }

    __device__ static int select_victim(int bid, int iter, State& state) {
        return -1; // Hardware chooses
    }
};
```

---

## Example: Complete Threshold Policy

```cpp
// 1. Define policy with state and callbacks
struct ThresholdPolicy {
    struct State {
        int work_done;
        int expected_work;
        float threshold;
    };

    __device__ static void init(int bid, int total_blocks, int total_items, State& state) {
        if (threadIdx.x == 0) {
            state.work_done = 0;
            state.expected_work = total_items / total_blocks;
            state.threshold = 0.7f;
        }
        __syncthreads();
    }

    __device__ static bool should_steal(int bid, int iter, State& state) {
        return (state.work_done >= state.expected_work * state.threshold);
    }

    __device__ static int select_victim(int bid, int iter, State& state) {
        return -1; // Let hardware choose
    }
};

// 2. Use in kernel (template instantiation)
template<typename Policy>
__global__ void work_stealing_kernel(float* data, int n) {
    __shared__ typename Policy::State policy_state;

    // Initialize policy
    Policy::init(blockIdx.x, gridDim.x, n, policy_state);

    // Work-stealing loop
    int iteration = 0;
    while (Policy::should_steal(blockIdx.x, iteration, policy_state)) {
        int victim = Policy::select_victim(blockIdx.x, iteration, policy_state);
        // ... CLC stealing logic ...
        iteration++;
    }
}

// 3. Launch with specific policy
work_stealing_kernel<ThresholdPolicy><<<blocks, threads>>>(data, n);
```

---

## Advantages of This Design

### 1. **True Minimalism**
   - Only 3 callbacks, no framework state
   - Policies are self-contained
   - No forced abstractions

### 2. **Maximum Flexibility**
   - Policy controls memory layout
   - Policy controls synchronization
   - Policy controls performance tradeoffs

### 3. **Easy to Extend**
   - Add new policy = implement 3 functions
   - No framework changes needed
   - Policies can share implementation

### 4. **Zero Overhead**
   - No unused state fields
   - No framework bookkeeping
   - Compile-time dispatch (no function pointers)
   - Compiler can inline all callbacks

---

## Comparison with Context-Based Design

| Aspect | Context-Based | Template-Based (This) |
|--------|--------------|----------------|
| Callbacks | 3 | 3 |
| Framework State | `SchedBlockCtx`, `SchedGlobalCtx` | None |
| Policy State | Fixed fields + custom[4] | `Policy::State` (fully custom) |
| Memory Layout | Framework decides | Policy decides |
| Synchronization | Framework provides | Policy implements |
| Dispatch | Function pointers (runtime) | Templates (compile-time) |
| Type Safety | `void*` casts required | Fully type-safe |
| Overhead | Some unused fields + indirection | Zero |
| Flexibility | Medium | Maximum |

---

## Policy Ideas That Benefit From This

### 1. **Priority-Based Work Stealing**
   - State: Priority queue in global memory
   - `select_victim()` reads queue, picks highest priority
   - Requires custom data structure → easy with `Policy::State`

### 2. **Load-Imbalance Detector**
   - State: Atomics to track block progress
   - `should_steal()` checks global imbalance metric
   - Requires cross-block coordination → policy implements it

### 3. **Cache-Aware Stealing**
   - State: Per-SM data structure tracking what's cached
   - `select_victim()` consults cache state
   - Requires complex state → policy manages it

### 4. **Preemptive Scheduling**
   - State: Signal flags for high-priority work
   - `should_steal()` checks for preemption request
   - Requires async communication → policy handles it

### 5. **Work Estimation Based**
   - State: Per-block work complexity estimates
   - `select_victim()` steals from blocks with most remaining work
   - Requires heuristics → policy defines them

---

## Implementation Strategy

### Host Side:
```cpp
// 1. Allocate global policy state (if policy needs it)
typename MyPolicy::State* d_global_state;
cudaMalloc(&d_global_state, sizeof(typename MyPolicy::State));

// 2. Initialize global policy parameters (if needed)
init_policy_on_host(d_global_state, params);

// 3. Launch kernel with policy template parameter
work_stealing_kernel<MyPolicy><<<blocks, threads>>>(data, n, d_global_state);
```

### Kernel Side:
```cpp
template<typename Policy>
__global__ void work_stealing_kernel(float* data, int n,
                                     typename Policy::State* global_state = nullptr) {
    // Per-block state in shared memory
    __shared__ typename Policy::State policy_state;

    // Copy global state to shared memory if needed
    if (global_state != nullptr && threadIdx.x == 0) {
        policy_state = *global_state;
    }
    __syncthreads();

    // Initialize policy
    Policy::init(blockIdx.x, gridDim.x, n, policy_state);

    // Work-stealing loop
    int iteration = 0;
    while (Policy::should_steal(blockIdx.x, iteration, policy_state)) {
        int victim = Policy::select_victim(blockIdx.x, iteration, policy_state);
        // ... CLC stealing logic ...
        iteration++;
    }
}
```

---

## Conclusion

This minimal template-based design:
- ✅ **3 callbacks only** (init, should_steal, select_victim)
- ✅ **No framework state** (just `Policy::State`)
- ✅ **Policy controls everything** (state, sync, memory)
- ✅ **Zero overhead** (compile-time dispatch, inlining)
- ✅ **Type-safe** (no `void*` casts)
- ✅ **Maximally flexible** (any policy expressible)

**Advantages over function pointers**:
- Compile-time optimization and inlining
- Type safety with no runtime overhead
- Better error messages at compile time
- No function pointer indirection cost

**Trade-off**: Policies need to manage their own state carefully, but they have complete control.

This is the true "sched_ext for GPU scheduling" - minimal, type-safe, and extensible.
