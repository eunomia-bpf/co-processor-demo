# Minimal Extensible Scheduler Framework for CLC

## Design Philosophy

**Goal**: Minimal callback interface where policies are self-contained and manage their own state.

No framework-imposed context structures. Policies define what state they need and how to synchronize it.

## Core Interface (3 Callbacks)

```c
struct SchedOps {
    // 1. Called once when block starts
    //    Policy initializes any state it needs (shared mem, global mem, etc.)
    __device__ void (*init)(int block_id, int total_blocks, int total_items, void* policy_state);

    // 2. Called in work-stealing loop to decide: should I try to steal?
    //    Returns: true = attempt steal, false = exit
    __device__ bool (*should_steal)(int block_id, int iteration, void* policy_state);

    // 3. Called when should_steal() returns true
    //    Returns: target block ID to steal from, or -1 for hardware default
    __device__ int (*select_victim)(int block_id, int iteration, void* policy_state);
};
```

**That's it.** Just 3 callbacks + opaque state pointer.

---

## Why This is Minimal Yet Sufficient

### 1. **No Framework State Structures**
   - No `SchedBlockCtx`, no `SchedGlobalCtx`
   - Policy defines its own state layout
   - Framework just provides `void* policy_state` pointer

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
```c
// No state struct needed
void* policy_state = NULL;

__device__ void greedy_init(int bid, int total_blocks, int total_items, void* state) {
    // Nothing to do
}

__device__ bool greedy_should_steal(int bid, int iter, void* state) {
    return true;
}

__device__ int greedy_select_victim(int bid, int iter, void* state) {
    return -1; // Hardware chooses
}
```

### Pattern 2: Per-Block State (Threshold)
```c
// Each block has its own state in shared memory
struct ThresholdState {
    int work_done;
    float threshold;
};

__shared__ ThresholdState my_state;

__device__ void threshold_init(int bid, int total_blocks, int total_items, void* state) {
    ThresholdState* s = (ThresholdState*)state;
    s->work_done = 0;
    s->threshold = 0.7f;
}

__device__ bool threshold_should_steal(int bid, int iter, void* state) {
    ThresholdState* s = (ThresholdState*)state;
    int expected = total_items / total_blocks;
    return (s->work_done >= expected * s->threshold);
}
```

### Pattern 3: Global Shared State (Locality)
```c
// All blocks share global state for coordination
struct LocalityState {
    int* block_groups;     // Device memory: group ID per block
    int group_size;
};

__device__ void locality_init(int bid, int total_blocks, int total_items, void* state) {
    LocalityState* s = (LocalityState*)state;
    s->block_groups[bid] = bid / s->group_size;
}

__device__ int locality_select_victim(int bid, int iter, void* state) {
    LocalityState* s = (LocalityState*)state;
    int my_group = s->block_groups[bid];

    // Try to find victim in same group
    for (int i = 0; i < s->group_size; i++) {
        int candidate = my_group * s->group_size + i;
        if (candidate != bid) return candidate;
    }
    return -1;
}
```

### Pattern 4: Adaptive State (Work-Rate)
```c
// Per-block state with runtime metrics
struct AdaptiveState {
    int work_done;
    unsigned long long start_time;
    float work_rate;
};

__shared__ AdaptiveState my_state;

__device__ bool adaptive_should_steal(int bid, int iter, void* state) {
    AdaptiveState* s = (AdaptiveState*)state;

    // Update work rate periodically
    if (iter % 10 == 0) {
        unsigned long long now = clock64();
        s->work_rate = (float)s->work_done / (float)(now - s->start_time);
    }

    // Fast workers steal earlier (lower threshold)
    float threshold = 0.5f / (1.0f + s->work_rate);
    return (s->work_done >= threshold * expected_work);
}
```

---

## Example: Complete Threshold Policy

```c
// 1. Define policy state
struct ThresholdState {
    int work_done;
    int expected_work;
    float threshold;
};

// 2. Implement callbacks
__device__ void threshold_init(int bid, int total_blocks, int total_items, void* state) {
    if (threadIdx.x == 0) {
        ThresholdState* s = (ThresholdState*)state;
        s->work_done = 0;
        s->expected_work = total_items / total_blocks;
        s->threshold = 0.7f;
    }
    __syncthreads();
}

__device__ bool threshold_should_steal(int bid, int iter, void* state) {
    ThresholdState* s = (ThresholdState*)state;
    return (s->work_done >= s->expected_work * s->threshold);
}

__device__ int threshold_select_victim(int bid, int iter, void* state) {
    return -1; // Let hardware choose
}

// 3. Register policy
__device__ SchedOps threshold_ops = {
    threshold_init,
    threshold_should_steal,
    threshold_select_victim
};

// 4. Use in kernel
__shared__ ThresholdState my_policy_state;
SchedOps* ops = &threshold_ops;
ops->init(blockIdx.x, gridDim.x, n, &my_policy_state);
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
   - Direct function pointers

---

## Comparison with Context-Based Design

| Aspect | Context-Based | Minimal (This) |
|--------|--------------|----------------|
| Callbacks | 3 | 3 |
| Framework State | `SchedBlockCtx`, `SchedGlobalCtx` | None |
| Policy State | Fixed fields + custom[4] | Fully custom |
| Memory Layout | Framework decides | Policy decides |
| Synchronization | Framework provides | Policy implements |
| Overhead | Some unused fields | Zero |
| Flexibility | Medium | Maximum |

---

## Policy Ideas That Benefit From This

### 1. **Priority-Based Work Stealing**
   - State: Priority queue in global memory
   - `select_victim()` reads queue, picks highest priority
   - Requires custom data structure → easy with `void*`

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
```c
// 1. Allocate policy state (if needed)
void* d_policy_state;
cudaMalloc(&d_policy_state, policy_state_size);

// 2. Initialize policy parameters
init_policy_on_host(d_policy_state, params);

// 3. Launch kernel with policy
kernel<<<blocks, threads>>>(data, n, &policy_ops, d_policy_state);
```

### Kernel Side:
```c
__global__ void kernel(float* data, int n, SchedOps* ops, void* policy_state) {
    // Per-block state (if policy uses shared memory)
    extern __shared__ char shared_state[];

    // Initialize policy
    ops->init(blockIdx.x, gridDim.x, n, shared_state);

    // Work loop
    int iteration = 0;
    while (ops->should_steal(blockIdx.x, iteration, shared_state)) {
        // Try to steal work...
        int victim = ops->select_victim(blockIdx.x, iteration, shared_state);
        // ... CLC stealing logic ...
        iteration++;
    }
}
```

---

## Conclusion

This minimal design:
- ✅ **3 callbacks only** (init, should_steal, select_victim)
- ✅ **No framework state** (just `void*`)
- ✅ **Policy controls everything** (state, sync, memory)
- ✅ **Zero overhead** (no unused fields)
- ✅ **Maximally flexible** (any policy expressible)

**Trade-off**: Policies need to manage their own state carefully, but they have complete control.

This is the true "sched_ext for GPU scheduling" - minimal, general, and extensible.
