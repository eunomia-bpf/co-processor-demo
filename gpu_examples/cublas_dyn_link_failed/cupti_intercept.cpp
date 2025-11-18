// CUPTI-based kernel interception for cuBLAS
// This uses CUDA Profiling Tools Interface to intercept ALL kernel launches

#include <cupti.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool policy_enabled = false;
static CUmodule policy_module = nullptr;
static CUfunction policy_function = nullptr;
static bool policy_initialized = false;
static bool in_policy_launch = false;  // Prevent recursive interception

// Store matrix parameters for policy application
static float* last_matrix_C = nullptr;
static int last_M = 0;
static int last_N = 0;

#define CUPTI_CALL(call)                                                \
do {                                                                     \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
        const char *errstr;                                             \
        cuptiGetResultString(_status, &errstr);                         \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                __FILE__, __LINE__, #call, errstr);                     \
        exit(-1);                                                       \
    }                                                                   \
} while (0)

#define CU_CALL(call)                                                   \
do {                                                                     \
    CUresult _status = call;                                            \
    if (_status != CUDA_SUCCESS) {                                      \
        const char *errstr;                                             \
        cuGetErrorString(_status, &errstr);                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                __FILE__, __LINE__, #call, errstr);                     \
    }                                                                   \
} while (0)

// Initialize policy kernel
static void init_policy() {
    if (policy_initialized) return;

    printf("[CUPTI Policy] Loading policy kernel...\n");

    // Load standalone policy kernel cubin
    CU_CALL(cuModuleLoad(&policy_module, "./policy_launcher.cubin"));

    if (policy_module) {
        // Get the policy kernel function
        CU_CALL(cuModuleGetFunction(&policy_function, policy_module,
                                    "apply_policy_upper_triangle_zero"));

        if (policy_function) {
            printf("[CUPTI Policy] Policy kernel loaded successfully\n");
            policy_initialized = true;
        } else {
            fprintf(stderr, "[CUPTI Policy] Failed to get policy function\n");
        }
    } else {
        fprintf(stderr, "[CUPTI Policy] Failed to load policy module\n");
    }
}

// Set matrix parameters for next policy application
extern "C" void cupti_set_matrix_params(float* C, int M, int N) {
    last_matrix_C = C;
    last_M = M;
    last_N = N;
}

// Callback for kernel launches
void CUPTIAPI callback_handler(void *userdata, CUpti_CallbackDomain domain,
                               CUpti_CallbackId cbid, const void *cbdata)
{
    const CUpti_CallbackData *cbd = (const CUpti_CallbackData *)cbdata;

    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        // Runtime API callbacks
        if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
            cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {

            if (cbd->callbackSite == CUPTI_API_ENTER) {
                printf("[CUPTI] Kernel launch detected (Runtime API)\n");
                printf("  Callback ID: %u\n", cbid);
                printf("  Function: %s\n", cbd->functionName);

                if (policy_enabled) {
                    printf("[CUPTI] *** Policy enforcement point ***\n");
                    // Here we could:
                    // 1. Extract kernel parameters
                    // 2. Modify the launch to use our wrapper
                    // 3. Inject policy code
                }
            }
        }
    } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        // Driver API callbacks
        if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
            cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {

            if (cbd->callbackSite == CUPTI_API_ENTER) {
                // Skip if we're launching our own policy kernel (prevent recursion)
                if (in_policy_launch) return;

                printf("[CUPTI] Kernel launch detected (Driver API)\n");
                printf("  Callback ID: %u\n", cbid);
                printf("  Function: %s\n", cbd->functionName);

                if (policy_enabled) {
                    printf("[CUPTI] *** Policy enforcement point ***\n");

                    // Extract launch parameters
                    const cuLaunchKernel_params *params =
                        (const cuLaunchKernel_params *)cbd->functionParams;

                    if (params) {
                        printf("[CUPTI] Grid: (%u, %u, %u), Block: (%u, %u, %u)\n",
                               params->gridDimX, params->gridDimY, params->gridDimZ,
                               params->blockDimX, params->blockDimY, params->blockDimZ);
                        printf("[CUPTI] Stream: %p, SharedMem: %u bytes\n",
                               params->hStream, params->sharedMemBytes);

                        // NOTE: We can't easily modify the cuBLAS kernel launch here
                        // because the parameters are const and already queued
                        // Instead, we'll launch a follow-up policy kernel
                        printf("[CUPTI] Will launch policy kernel after cuBLAS completes\n");
                    }
                }
            } else if (cbd->callbackSite == CUPTI_API_EXIT) {
                // Skip if we're in policy launch
                if (in_policy_launch) return;

                // Kernel has been launched, now we can inject policy
                if (policy_enabled && last_matrix_C != nullptr) {
                    init_policy();

                    if (policy_function) {
                        printf("[CUPTI] Launching policy kernel to enforce constraints\n");

                        // Extract parameters from the original launch
                        const cuLaunchKernel_params *params =
                            (const cuLaunchKernel_params *)cbd->functionParams;

                        // Calculate grid/block for policy kernel
                        // Use 16x16 blocks for good occupancy
                        unsigned int blockX = 16;
                        unsigned int blockY = 16;
                        unsigned int gridX = (last_N + blockX - 1) / blockX;
                        unsigned int gridY = (last_M + blockY - 1) / blockY;

                        printf("[CUPTI Policy] Launching with grid(%u,%u) block(%u,%u)\n",
                               gridX, gridY, blockX, blockY);
                        printf("[CUPTI Policy] Matrix C=%p, M=%d, N=%d\n",
                               last_matrix_C, last_M, last_N);

                        // Set up kernel parameters
                        void* policy_args[] = {
                            &last_matrix_C,
                            &last_M,
                            &last_N
                        };

                        // Set flag to prevent recursive interception
                        in_policy_launch = true;

                        // Launch policy kernel on the same stream as cuBLAS
                        CUresult res = cuLaunchKernel(
                            policy_function,
                            gridX, gridY, 1,      // grid dim
                            blockX, blockY, 1,    // block dim
                            0,                     // shared mem
                            params ? params->hStream : nullptr,  // same stream as cuBLAS
                            policy_args,
                            nullptr
                        );

                        // Clear flag
                        in_policy_launch = false;

                        if (res == CUDA_SUCCESS) {
                            printf("[CUPTI] ✓ Policy kernel launched successfully!\n");
                        } else {
                            const char* errstr;
                            cuGetErrorString(res, &errstr);
                            fprintf(stderr, "[CUPTI] Failed to launch policy kernel: %s\n", errstr);
                        }
                    }
                }
            }
        }
    }
}

// Initialize CUPTI callbacks
extern "C" void cupti_init() {
    printf("[CUPTI] Initializing CUPTI callbacks...\n");

    CUpti_SubscriberHandle subscriber;
    CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callback_handler, NULL));

    // Enable runtime API callbacks
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

    // Enable driver API callbacks
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

    printf("[CUPTI] Callbacks registered successfully\n");
}

extern "C" void cupti_enable_policy() {
    policy_enabled = true;
    printf("[CUPTI] Policy enforcement ENABLED\n");
}

extern "C" void cupti_disable_policy() {
    policy_enabled = false;
    printf("[CUPTI] Policy enforcement DISABLED\n");
}

// Auto-initialize on library load
__attribute__((constructor))
static void init() {
    printf("[CUPTI] Library loaded\n");
    cupti_init();
}
