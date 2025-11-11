//
// Cluster Launch Control (CLC) - Safe Scheduler Policy Interface
//
// This header defines a minimal, safe-by-construction interface for creating
// custom scheduling policies on top of the CLC work-stealing framework.
//
// Key safety principle: Policies control WHEN to steal (participation),
// hardware controls WHAT to steal (work selection). This separation ensures
// policies cannot violate CLC constraints or introduce correctness bugs.
//

#pragma once

#include <cuda/ptx>

namespace clc_policy {

// ============================================================================
// 1. Safe Scheduler Policy Interface
// ============================================================================

/**
 * @brief Template-based CLC Scheduler Policy Interface
 *
 * Policies provide 3 static callbacks that control block participation in
 * work stealing. The CLC framework manages all hardware interactions and
 * enforces safety constraints.
 *
 * Safety guarantees:
 * - Policy cannot submit requests after observing failure
 * - Policy cannot decode block IDs on failure
 * - Policy cannot violate CLC memory ordering or synchronization
 * - Skipping steal attempts is always safe
 *
 * @tparam Policy The user-defined policy implementation
 */
template<typename Policy>
struct ClcSchedulerPolicy {
    /**
     * @brief Initialize policy state.
     * Called once per thread block at kernel start, before any work stealing.
     */
    __device__ static void init() {
        Policy::init();
    }

    /**
     * @brief Decide whether to submit a steal request.
     * Called BEFORE each steal attempt. Returning false safely skips the attempt.
     *
     * @return true = submit steal request, false = skip (safe)
     */
    __device__ static bool should_try_steal() {
        return Policy::should_try_steal();
    }

    /**
     * @brief Decide whether to continue after successful steal.
     * Called AFTER a successful steal, with the stolen block ID.
     *
     * @param stolen_bx The block ID that was stolen (validated by hardware)
     * @return true = continue stealing, false = exit loop
     */
    __device__ static bool keep_going_after_success(int stolen_bx) {
        return Policy::keep_going_after_success(stolen_bx);
    }
};


// ============================================================================
// 2. Example Policy Implementations
// ============================================================================

// ----------------------------------------------------------------------------
// Policy 1: Greedy (Baseline)
//
// Always steal until hardware says no more work. Mimics default CLC behavior.
// ----------------------------------------------------------------------------

struct GreedyPolicy {
    __device__ static void init() {
        // No state to initialize
    }

    __device__ static bool should_try_steal() {
        return true;  // Always try to steal
    }

    __device__ static bool keep_going_after_success(int stolen_bx) {
        return true;  // Always continue
    }
};


// ----------------------------------------------------------------------------
// Policy 2: MaxSteals
//
// Stop after N successful steals. Useful for load balancing.
// ----------------------------------------------------------------------------

struct MaxStealsPolicy {
    static constexpr int max_steals = 8;

    // Helper to get per-block shared state
    __device__ static int& get_steals_done() {
        __shared__ int steals_done;
        return steals_done;
    }

    __device__ static void init() {
        if (threadIdx.x == 0) {
            get_steals_done() = 0;
        }
        __syncthreads();
    }

    __device__ static bool should_try_steal() {
        return get_steals_done() < max_steals;
    }

    __device__ static bool keep_going_after_success(int stolen_bx) {
        if (threadIdx.x == 0) {
            atomicAdd(&get_steals_done(), 1);
        }
        __syncthreads();
        return get_steals_done() < max_steals;
    }
};


// ----------------------------------------------------------------------------
// Policy 3: ThrottledStealing
//
// Only steal every Nth iteration. Reduces contention.
// ----------------------------------------------------------------------------

struct ThrottledPolicy {
    static constexpr int N = 2;  // Steal every 2nd iteration

    // Helper to get per-block shared state
    __device__ static int& get_iteration() {
        __shared__ int iteration;
        return iteration;
    }

    __device__ static bool& get_result() {
        __shared__ bool result;
        return result;
    }

    __device__ static void init() {
        if (threadIdx.x == 0) {
            get_iteration() = 0;
        }
        __syncthreads();
    }

    __device__ static bool should_try_steal() {
        if (threadIdx.x == 0) {
            get_result() = (get_iteration() % N) == 0;
            get_iteration()++;
        }
        __syncthreads();
        return get_result();
    }

    __device__ static bool keep_going_after_success(int stolen_bx) {
        return true;
    }
};


// ----------------------------------------------------------------------------
// Policy 4: SelectiveBlocks
//
// Only certain blocks steal aggressively (e.g., first half).
// ----------------------------------------------------------------------------

struct SelectiveBlocksPolicy {
    __device__ static void init() {
        // No state to initialize
    }

    __device__ static bool should_try_steal() {
        // Only first half of blocks steal
        return blockIdx.x < (gridDim.x / 2);
    }

    __device__ static bool keep_going_after_success(int stolen_bx) {
        return blockIdx.x < (gridDim.x / 2);
    }
};


// ----------------------------------------------------------------------------
// Policy Composition: AND combinator
//
// Combine multiple policies - both must agree to steal.
// ----------------------------------------------------------------------------

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

// Example: First half of blocks + max 8 steals each
using SelectiveThrottled = AndPolicy<SelectiveBlocksPolicy, MaxStealsPolicy>;


// ----------------------------------------------------------------------------
// Policy 5: Voting Policy
//
// Warps vote on whether to attempt a steal. A steal is only tried if a
// threshold of warps agree. This demonstrates intra-block consensus.
// ----------------------------------------------------------------------------

struct VotingPolicy {
    // Threshold: at least half the warps must vote to steal.
    // Note: In a real scenario, this could be tuned.
    __device__ static int get_threshold() {
        return (blockDim.x + 31) / 32 / 2;
    }

    // Shared memory for votes and the final decision.
    __device__ static int& get_vote_count() {
        __shared__ int vote_count;
        return vote_count;
    }
    __device__ static bool& get_decision() {
        __shared__ bool decision;
        return decision;
    }

    __device__ static void init() {
        if (threadIdx.x == 0) {
            get_vote_count() = 0;
        }
        __syncthreads();
    }

    __device__ static bool should_try_steal() {
        // 1. Each warp leader casts a vote.
        //    For this example, we'll use a simple probabilistic vote.
        //    A real policy could base this on workload progress, etc.
        if ((threadIdx.x % 32) == 0) {
            // Simple probabilistic vote: 75% chance to vote 'yes'
            if ((clock() & 0xFF) > 64) {
                atomicAdd(&get_vote_count(), 1);
            }
        }
        __syncthreads();

        // 2. Thread 0 checks if the threshold is met and makes a decision.
        if (threadIdx.x == 0) {
            get_decision() = (get_vote_count() >= get_threshold());
            get_vote_count() = 0; // Reset for next iteration
        }
        __syncthreads();

        // 3. All threads return the collective decision.
        return get_decision();
    }

    __device__ static bool keep_going_after_success(int stolen_bx) {
        // For simplicity, we always continue after a successful steal.
        // A more complex policy could change its voting behavior based on
        // which block was stolen.
        return true;
    }
};

} // namespace clc_policy
