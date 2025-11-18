/**
 * Demo 2: cuGetProcAddress / cudaGetDriverEntryPoint
 *
 * This demo shows how to use CUDA 11.3+ driver function pointer APIs:
 * - cuGetProcAddress: Get driver API function pointers
 * - cudaGetDriverEntryPoint: Get driver entry points from runtime
 *
 * This is the foundation for building proxy/hook libraries:
 * - Build a shim libcuda.so that intercepts all driver calls
 * - Use cuGetProcAddress to get the real driver functions
 * - Implement your policy/CLC wrapper logic in the interception layer
 *
 * Benefits:
 * 1. Official way to hook driver APIs (no LD_PRELOAD hacks)
 * 2. Handles versioning and symbol resolution automatically
 * 3. Works with static/dynamic linking
 * 4. Foundation for your policy injection framework
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <dlfcn.h>

#define CHECK_CU(call) { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        std::cerr << "CUDA Driver Error: " << errStr \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Function pointer types for driver APIs
typedef CUresult (*PFN_cuInit)(unsigned int Flags);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice *device, int ordinal);
typedef CUresult (*PFN_cuDeviceGetName)(char *name, int len, CUdevice dev);
typedef CUresult (*PFN_cuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev);
typedef CUresult (*PFN_cuLaunchKernel)(CUfunction f,
                                       unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                       unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                       unsigned int sharedMemBytes,
                                       CUstream hStream, void **kernelParams, void **extra);

void printDriverVersion(int driverVersion) {
    std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "."
              << (driverVersion % 100) / 10 << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "CUDA 11.3+ cuGetProcAddress Demo" << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize CUDA driver
    CHECK_CU(cuInit(0));

    int driverVersion;
    CHECK_CU(cuDriverGetVersion(&driverVersion));
    printDriverVersion(driverVersion);

    if (driverVersion < 11030) {
        std::cerr << "This demo requires CUDA 11.3 or higher" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n=== Part 1: Basic cuGetProcAddress Usage ===" << std::endl;

    // Get function pointers using cuGetProcAddress
    void* funcPtr;
    CUdriverProcAddressQueryResult queryResult;

    // Method 1: Get cuDeviceGet
    CHECK_CU(cuGetProcAddress("cuDeviceGet", &funcPtr, driverVersion,
                              CU_GET_PROC_ADDRESS_DEFAULT, &queryResult));

    if (queryResult == CU_GET_PROC_ADDRESS_SUCCESS) {
        std::cout << "✓ Got cuDeviceGet function pointer: " << funcPtr << std::endl;

        // Use the function pointer
        PFN_cuDeviceGet pfn_cuDeviceGet = (PFN_cuDeviceGet)funcPtr;
        CUdevice device;
        CHECK_CU(pfn_cuDeviceGet(&device, 0));
        std::cout << "  Device 0: " << device << std::endl;
    }

    // Method 2: Get cuDeviceGetName
    CHECK_CU(cuGetProcAddress("cuDeviceGetName", &funcPtr, driverVersion,
                              CU_GET_PROC_ADDRESS_DEFAULT, &queryResult));

    if (queryResult == CU_GET_PROC_ADDRESS_SUCCESS) {
        std::cout << "✓ Got cuDeviceGetName function pointer: " << funcPtr << std::endl;

        PFN_cuDeviceGetName pfn_cuDeviceGetName = (PFN_cuDeviceGetName)funcPtr;
        CUdevice device;
        CHECK_CU(cuDeviceGet(&device, 0));

        char deviceName[256];
        CHECK_CU(pfn_cuDeviceGetName(deviceName, sizeof(deviceName), device));
        std::cout << "  Device Name: " << deviceName << std::endl;
    }

    std::cout << "\n=== Part 2: Demonstrating Hook/Interception Pattern ===" << std::endl;
    std::cout << "This pattern is the foundation for building a proxy libcuda" << std::endl;

    // Simulate what a hook library would do
    struct DriverAPIHook {
        PFN_cuDeviceGetAttribute real_cuDeviceGetAttribute = nullptr;

        void init(int driverVersion) {
            void* funcPtr;
            CUdriverProcAddressQueryResult result;
            CUresult err = cuGetProcAddress("cuDeviceGetAttribute", &funcPtr,
                                           driverVersion, CU_GET_PROC_ADDRESS_DEFAULT, &result);
            if (err == CUDA_SUCCESS && result == CU_GET_PROC_ADDRESS_SUCCESS) {
                real_cuDeviceGetAttribute = (PFN_cuDeviceGetAttribute)funcPtr;
                std::cout << "✓ Hook initialized: cuDeviceGetAttribute -> " << funcPtr << std::endl;
            }
        }

        // This is what your hook function would look like
        CUresult hooked_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
            std::cout << "  [HOOK] Intercepted cuDeviceGetAttribute call" << std::endl;
            std::cout << "  [HOOK]   Attribute: " << attrib << ", Device: " << dev << std::endl;

            // Here you could inject your policy/CLC logic:
            // - Check if this attribute query should be modified
            // - Log the query for profiling
            // - Apply resource limits based on policy
            // - etc.

            // Call the real driver function
            CUresult result = real_cuDeviceGetAttribute(pi, attrib, dev);

            std::cout << "  [HOOK]   Result: " << *pi << std::endl;
            return result;
        }
    };

    DriverAPIHook hook;
    hook.init(driverVersion);

    // Use the hooked version
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    int computeCapabilityMajor, computeCapabilityMinor;
    hook.hooked_cuDeviceGetAttribute(&computeCapabilityMajor,
                                     CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    hook.hooked_cuDeviceGetAttribute(&computeCapabilityMinor,
                                     CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

    std::cout << "  Compute Capability: " << computeCapabilityMajor << "."
              << computeCapabilityMinor << std::endl;

    std::cout << "\n=== Part 3: cudaGetDriverEntryPoint (Runtime API) ===" << std::endl;

    // CUDA 11.3+ also provides cudaGetDriverEntryPoint in runtime API
    void* driverFuncPtr;
    cudaError_t rtErr = cudaGetDriverEntryPoint("cuDeviceGetAttribute", &driverFuncPtr,
                                                 cudaEnableDefault);

    if (rtErr == cudaSuccess) {
        std::cout << "✓ Got driver function via runtime API: " << driverFuncPtr << std::endl;
        std::cout << "  This allows runtime to access driver functions directly" << std::endl;

        // Use it
        PFN_cuDeviceGetAttribute pfn = (PFN_cuDeviceGetAttribute)driverFuncPtr;
        int maxThreadsPerBlock;
        CHECK_CU(pfn(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
        std::cout << "  Max threads per block: " << maxThreadsPerBlock << std::endl;
    } else {
        std::cout << "⚠ cudaGetDriverEntryPoint not available: "
                  << cudaGetErrorString(rtErr) << std::endl;
    }

    std::cout << "\n=== Part 4: Building a Minimal Proxy Pattern ===" << std::endl;
    std::cout << "Example of how to build a complete proxy libcuda.so:\n" << std::endl;

    std::cout << R"(
// proxy_libcuda.cpp
// Compile: g++ -shared -fPIC proxy_libcuda.cpp -o libcuda.so -ldl

#include <cuda.h>
#include <dlfcn.h>

// Storage for real driver functions
static PFN_cuLaunchKernel real_cuLaunchKernel = nullptr;
static void* real_libcuda_handle = nullptr;

// Initialize proxy
static void init_proxy() {
    static bool initialized = false;
    if (initialized) return;

    // Load real libcuda.so (from system location)
    real_libcuda_handle = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1", RTLD_LAZY);

    // Get real cuInit to initialize driver
    auto real_cuInit = (PFN_cuInit)dlsym(real_libcuda_handle, "cuInit");
    real_cuInit(0);

    // Use cuGetProcAddress to get real functions
    int driverVersion;
    auto real_cuDriverGetVersion = (CUresult(*)(int*))dlsym(real_libcuda_handle, "cuDriverGetVersion");
    real_cuDriverGetVersion(&driverVersion);

    void* funcPtr;
    CUdriverProcAddressQueryResult result;
    auto real_cuGetProcAddress = (CUresult(*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*))
        dlsym(real_libcuda_handle, "cuGetProcAddress");

    real_cuGetProcAddress("cuLaunchKernel", &funcPtr, driverVersion, 0, &result);
    real_cuLaunchKernel = (PFN_cuLaunchKernel)funcPtr;

    initialized = true;
}

// Your hooked cuLaunchKernel
extern "C" CUresult cuLaunchKernel(CUfunction f, ...) {
    init_proxy();

    // YOUR POLICY / CLC WRAPPER LOGIC HERE:
    // 1. Intercept the kernel launch
    // 2. Decide if you need to wrap it with CLC
    // 3. Maybe load a different module with your wrapper
    // 4. Apply scheduling policy
    printf("[CLC-HOOK] Intercepted kernel launch!\n");

    // Call real driver function
    return real_cuLaunchKernel(f, ...);
}

// Repeat for all driver APIs you want to hook...
    )" << std::endl;

    std::cout << "\nUsage:" << std::endl;
    std::cout << "  LD_LIBRARY_PATH=. ./your_cuda_app" << std::endl;
    std::cout << "  (Or install to standard location)" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Key Takeaways:" << std::endl;
    std::cout << "1. cuGetProcAddress() - Official way to get driver function pointers" << std::endl;
    std::cout << "2. No need for LD_PRELOAD hacks" << std::endl;
    std::cout << "3. Handles symbol versioning automatically" << std::endl;
    std::cout << "4. Foundation for building proxy/hook libraries" << std::endl;
    std::cout << "5. Perfect for injecting CLC/policy frameworks" << std::endl;
    std::cout << "========================================" << std::endl;

    return EXIT_SUCCESS;
}
