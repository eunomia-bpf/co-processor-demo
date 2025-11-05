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
 * Key design principle: Policy defines State type, but the FRAMEWORK holds
 * it in __shared__ memory and passes it by reference to callbacks. This
 * enforces uniform control flow and prevents __shared__ static UB.
 *
 * Safety guarantees:
 * - Policy cannot submit requests after observing failure
 * - Policy cannot decode block IDs on failure
 * - Policy cannot violate CLC memory ordering or synchronization
 * - Skipping steal attempts is always safe
 * - Framework enforces uniform control flow (elect-and-broadcast pattern)
 *
 * @tparam Policy The user-defined policy implementation
 */
template<typename Policy>
struct ClcSchedulerPolicy {
    // Policy must define this type (can be empty struct for stateless policies)
    using State = typename Policy::State;

    /**
     * @brief Initialize policy state.
     * Called once per thread block at kernel start, before any work stealing.
     *
     * @param s Policy state (held by framework in __shared__ memory)
     */
    __device__ static void init(State& s) {
        Policy::init(s);
    }

    /**
     * @brief Decide whether to submit a steal request.
     * Called BEFORE each steal attempt. Returning false safely skips the attempt.
     *
     * CRITICAL: This is called by thread 0 only. Framework broadcasts result.
     *
     * @param s Policy state (held by framework in __shared__ memory)
     * @return true = submit steal request, false = skip (safe)
     */
    __device__ static bool should_try_steal(State& s) {
        return Policy::should_try_steal(s);
    }

    /**
     * @brief Decide whether to continue after successful steal.
     * Called AFTER a successful steal, with the stolen block ID.
     *
     * CRITICAL: This is called by thread 0 only. Framework broadcasts result.
     *
     * @param stolen_bx The block ID that was stolen (validated by hardware)
     * @param s Policy state (held by framework in __shared__ memory)
     * @return true = continue stealing, false = exit loop
     */
    __device__ static bool keep_going_after_success(int stolen_bx, State& s) {
        return Policy::keep_going_after_success(stolen_bx, s);
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
    struct State {};  // Empty state - stateless policy

    __device__ static void init(State& s) {
        // No state to initialize
    }

    __device__ static bool should_try_steal(State& s) {
        return true;  // Always try to steal
    }

    __device__ static bool keep_going_after_success(int stolen_bx, State& s) {
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

    struct State {
        int steals_done;
    };

    __device__ static void init(State& s) {
        s.steals_done = 0;
    }

    __device__ static bool should_try_steal(State& s) {
        return s.steals_done < max_steals;
    }

    __device__ static bool keep_going_after_success(int stolen_bx, State& s) {
        s.steals_done++;  // Thread 0 updates; framework ensures uniform control
        return s.steals_done < max_steals;
    }
};


// ----------------------------------------------------------------------------
// Policy 3: ThrottledStealing
//
// Only steal every Nth iteration. Reduces contention.
// ----------------------------------------------------------------------------

struct ThrottledPolicy {
    static constexpr int N = 2;  // Steal every 2nd iteration

    struct State {
        int iteration;
    };

    __device__ static void init(State& s) {
        s.iteration = 0;
    }

    __device__ static bool should_try_steal(State& s) {
        bool result = (s.iteration % N) == 0;
        s.iteration++;
        return result;
    }

    __device__ static bool keep_going_after_success(int stolen_bx, State& s) {
        return true;  // Continue (throttling happens in should_try_steal)
    }
};


// ----------------------------------------------------------------------------
// Policy 4: SelectiveBlocks
//
// Only certain blocks steal aggressively (e.g., first half).
// ----------------------------------------------------------------------------

struct SelectiveBlocksPolicy {
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


// ----------------------------------------------------------------------------
// Policy Composition: AND combinator
//
// Combine multiple policies - both must agree to steal.
// ----------------------------------------------------------------------------

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

// Example: First half of blocks + max 8 steals each
using SelectiveThrottled = AndPolicy<SelectiveBlocksPolicy, MaxStealsPolicy>;


// ----------------------------------------------------------------------------
// Policy 5: Voting Policy
//
// Demonstrates a complex stateful policy. In the framework-held state pattern,
// all voting logic must be called by thread 0 only. This simplified version
// uses a probabilistic decision instead of warp voting.
// ----------------------------------------------------------------------------

struct VotingPolicy {
    struct State {
        int iteration;
    };

    __device__ static void init(State& s) {
        s.iteration = 0;
    }

    __device__ static bool should_try_steal(State& s) {
        // Simple probabilistic decision (75% chance to steal)
        // Thread 0 makes this decision, framework broadcasts it
        // In a real implementation, this could use more sophisticated logic
        bool decision = ((clock64() + s.iteration) & 0xFF) > 64;
        s.iteration++;
        return decision;
    }

    __device__ static bool keep_going_after_success(int stolen_bx, State& s) {
        return true;  // Always continue after successful steal
    }
};

} // namespace clc_policy
