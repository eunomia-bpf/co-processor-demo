//
// Cluster Launch Control (CLC) - Programmable Scheduler Policy Interface
//
// This header defines a minimal, general-purpose interface for creating
// custom scheduling policies on top of the CLC work-stealing framework.
// It is inspired by extensible scheduler concepts like sched_ext.
//

#pragma once

#include <cuda/ptx>

namespace clc_policy {

// ============================================================================
// 1. Scheduler Policy Interface Definition
// ============================================================================

/**
 * @brief Base template for a CLC Scheduler Policy.
 *
 * A policy is a struct that provides a set of static callbacks (or "operations")
 * that the CLC work-stealing loop can invoke at key decision points.
 *
 * This allows developers to implement custom logic for work selection,
 * preemption, and granularity without altering the core CLC framework.
 *
 * @tparam Policy The user-defined policy implementation.
 * @tparam SharedMem The type for the policy's shared memory struct.
 */
template<typename Policy, typename SharedMem>
struct ClcSchedulerPolicy {
    /**
     * @brief Callback to initialize the policy's shared memory.
     * Invoked once per thread block at the start of the kernel.
     */
    __device__ static void init(SharedMem& smem) {
        Policy::init(smem);
    }

    /**
     * @brief Callback to select the next work item for a thread block.
     *
     * This is the core of the policy. It's called when a block is ready for new
     * work. The policy decides which work item ID to assign.
     *
     * @param smem The policy's shared memory.
     * @param hardware_cta_id The block ID returned by the hardware work-stealing.
     *                        This can be used or ignored by the policy.
     * @return The ID of the next work item to be processed.
     */
    __device__ static int select_work(SharedMem& smem, int hardware_cta_id) {
        return Policy::select_work(smem, hardware_cta_id);
    }

    /**
     * @brief Callback to determine if a block should stop and yield.
     *
     * Allows for implementing preemption. If this returns true, the work-stealing
     * loop can be terminated, allowing the block to exit.
     *
     * @param smem The policy's shared memory.
     * @return True if the block should preempt its work, false otherwise.
     */
    __device__ static bool should_preempt(SharedMem& smem) {
        return Policy::should_preempt(smem);
    }
};


// ============================================================================
// 2. Example Policy Implementations
// ============================================================================

// ----------------------------------------------------------------------------
// Policy 1: DefaultGreedy (Baseline)
//
// This policy mimics the default hardware behavior: a block that successfully
// steals work immediately processes the work item corresponding to the stolen
// block ID.
// ----------------------------------------------------------------------------

struct DefaultGreedyPolicy {
    // This policy requires no special shared memory.
    struct SharedMemory {};

    __device__ static void init(SharedMemory& smem) {
        // No-op
    }

    __device__ static int select_work(SharedMemory& smem, int hardware_cta_id) {
        // Directly use the block ID provided by the hardware.
        return hardware_cta_id;
    }

    __device__ static bool should_preempt(SharedMemory& smem) {
        // Never preempt.
        return false;
    }
};


// ----------------------------------------------------------------------------
// Policy 2: PriorityBased
//
// This policy implements priority-based scheduling. It uses a software queue
// stored in shared memory to manage work items. Blocks always pick the
// highest-priority available work item, ignoring the ID from the hardware.
// ----------------------------------------------------------------------------

// A simple fixed-size priority queue for demonstrating the concept.
// In a real implementation, this could be backed by global memory.
#define PRIORITY_QUEUE_SIZE 256

struct PriorityBasedPolicy {
    struct WorkItem {
        int id;
        int priority; // Lower value = higher priority
    };

    struct SharedMemory {
        WorkItem queue[PRIORITY_QUEUE_SIZE];
        int queue_size;
    };

    __device__ static void init(SharedMemory& smem) {
        // In a real scenario, this would be populated from global memory.
        // For this example, we create a dummy queue.
        if (threadIdx.x == 0) {
            smem.queue_size = PRIORITY_QUEUE_SIZE;
            for (int i = 0; i < PRIORITY_QUEUE_SIZE; ++i) {
                smem.queue[i].id = blockIdx.x * PRIORITY_QUEUE_SIZE + i;
                // Assign higher priority to items at the end of the list
                smem.queue[i].priority = PRIORITY_QUEUE_SIZE - i;
            }
        }
        __syncthreads();
    }

    __device__ static int select_work(SharedMemory& smem, int hardware_cta_id) {
        // This policy ignores the hardware_cta_id and uses its own logic.
        // It finds and claims the highest-priority item from the shared queue.
        int best_priority = -1;
        int best_idx = -1;
        int work_id = -1;

        // Find highest priority item (simplified for demo)
        if (threadIdx.x == 0) {
            for (int i = 0; i < smem.queue_size; ++i) {
                if (smem.queue[i].priority > best_priority) {
                    best_priority = smem.queue[i].priority;
                    best_idx = i;
                }
            }

            if (best_idx != -1) {
                work_id = smem.queue[best_idx].id;
                // Mark as taken by setting priority to -1
                smem.queue[best_idx].priority = -1;
            }
        }

        // Broadcast the chosen work ID to all threads in the block.
        work_id = __shfl_sync(0xffffffff, work_id, 0);
        return work_id;
    }

    __device__ static bool should_preempt(SharedMemory& smem) {
        // Could check a global flag for high-priority preemption signal.
        return false;
    }
};

// ----------------------------------------------------------------------------
// Policy 3: LocalityAware
//
// This policy attempts to schedule work that is "close" to the block's
// physical location to improve cache reuse.
// ----------------------------------------------------------------------------

struct LocalityAwarePolicy {
    struct SharedMemory {
        int locality_group_id;
    };

    __device__ static void init(SharedMemory& smem) {
        // Determine a locality group based on the physical location of the block.
        // This is a simplified example.
        if (threadIdx.x == 0) {
            smem.locality_group_id = blockIdx.x % 16; // Example: 16 groups
        }
        __syncthreads();
    }

    __device__ static int select_work(SharedMemory& smem, int hardware_cta_id) {
        // A real implementation would check if `hardware_cta_id` belongs to the
        // same locality group. If not, it might try to steal again or consult
        // a work queue for a better match.
        // For this example, we just return the hardware ID.
        return hardware_cta_id;
    }

    __device__ static bool should_preempt(SharedMemory& smem) {
        return false;
    }
};

} // namespace clc_policy
