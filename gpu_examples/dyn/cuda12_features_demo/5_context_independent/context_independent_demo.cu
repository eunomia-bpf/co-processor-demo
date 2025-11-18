/**
 * Demo 5: Context-Independent Module/Library Loading
 *
 * This demo shows CUDA 12+ context-independent loading:
 * - cuLibraryLoadData/FromFile with lazy loading
 * - Sharing modules across multiple contexts
 * - Building a module cache for multi-tenant scenarios
 *
 * In older CUDA:
 * - cuModuleLoad* are context-specific
 * - Each context needs its own copy
 * - Difficult to share modules
 *
 * In CUDA 12+:
 * - cuLibrary* can be loaded once, used by multiple contexts
 * - Better for multi-process/multi-context scenarios
 * - Ideal for your CLC proxy service
 *
 * Benefits for your framework:
 * 1. Load CLC skeleton once, share across all contexts
 * 2. Module cache by (device, cubin_hash)
 * 3. Lower memory overhead
 * 4. Faster initialization for new contexts
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>

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

std::vector<char> readFile(const char* filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
        return buffer;
    }
    return {};
}

void printArray(const char* name, const float* arr, int n, int limit = 10) {
    std::cout << name << ": [";
    for (int i = 0; i < std::min(n, limit); i++) {
        std::cout << arr[i];
        if (i < std::min(n, limit) - 1) std::cout << ", ";
    }
    if (n > limit) std::cout << " ...";
    std::cout << "]" << std::endl;
}

// Simple module cache implementation
class ModuleCache {
private:
    struct CacheEntry {
        CUlibrary library;
        std::vector<char> binary;  // Keep binary alive if needed
        int refCount;
    };

    std::unordered_map<std::string, CacheEntry> cache_;

public:
    CUlibrary getOrLoad(const std::string& key, const std::vector<char>& binary) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            it->second.refCount++;
            std::cout << "  [CACHE HIT] Reusing existing library for: " << key << std::endl;
            std::cout << "  RefCount: " << it->second.refCount << std::endl;
            return it->second.library;
        }

        std::cout << "  [CACHE MISS] Loading new library for: " << key << std::endl;

        // Load the library - it's context-independent!
        CUlibrary library;
        CUresult result = cuLibraryLoadData(&library, binary.data(),
                                           nullptr, nullptr, 0,
                                           nullptr, nullptr, 0);

        if (result == CUDA_SUCCESS) {
            CacheEntry entry;
            entry.library = library;
            entry.binary = binary;
            entry.refCount = 1;
            cache_[key] = entry;

            std::cout << "  ✓ Loaded and cached library" << std::endl;
            return library;
        } else {
            const char* errStr;
            cuGetErrorString(result, &errStr);
            std::cerr << "  Failed to load library: " << errStr << std::endl;
            return nullptr;
        }
    }

    void release(const std::string& key) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            it->second.refCount--;
            std::cout << "  [CACHE] Released reference to: " << key
                      << " (refCount: " << it->second.refCount << ")" << std::endl;

            if (it->second.refCount <= 0) {
                cuLibraryUnload(it->second.library);
                cache_.erase(it);
                std::cout << "  [CACHE] Evicted: " << key << std::endl;
            }
        }
    }

    size_t size() const { return cache_.size(); }

    void clear() {
        for (auto& [key, entry] : cache_) {
            cuLibraryUnload(entry.library);
        }
        cache_.clear();
        std::cout << "  [CACHE] Cleared all entries" << std::endl;
    }

    ~ModuleCache() {
        clear();
    }
};

// Simulate a multi-context scenario
void runInContext(CUcontext ctx, CUlibrary library, int contextId) {
    std::cout << "\n  === Running in Context " << contextId << " ===" << std::endl;

    // Set the context
    CHECK_CU(cuCtxSetCurrent(ctx));

    // Get kernel from the library (library is context-independent!)
    CUkernel kernel;
    CUresult result = cuLibraryGetKernel(&kernel, library, "vectorAdd");

    if (result != CUDA_SUCCESS) {
        std::cout << "  ⚠ Kernel 'vectorAdd' not found (expected if no .fatbin)" << std::endl;
        return;
    }

    std::cout << "  ✓ Got kernel: " << kernel << std::endl;

    // Allocate memory and run kernel
    const int N = 1024;
    const int bytes = N * sizeof(float);

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i + contextId * 1000);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    int n = N;
    void* args[] = {&d_a, &d_b, &d_c, &n};

    // Launch using CUkernel (from context-independent library)
    CHECK_CU(cuLaunchKernel((CUfunction)kernel,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           0, nullptr, args, nullptr));
    CHECK_CU(cuCtxSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    std::cout << "  ";
    printArray("Result", h_c.data(), N, 5);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    std::cout << "  ✓ Context " << contextId << " completed successfully" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Context-Independent Loading Demo" << std::endl;
    std::cout << "========================================" << std::endl;

    CHECK_CU(cuInit(0));

    int driverVersion;
    CHECK_CU(cuDriverGetVersion(&driverVersion));
    std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "."
              << (driverVersion % 100) / 10 << std::endl;

    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    char deviceName[256];
    CHECK_CU(cuDeviceGetName(deviceName, sizeof(deviceName), device));
    std::cout << "Device: " << deviceName << std::endl;

    std::cout << "\n=== Part 1: Traditional cuModule (context-specific) ===" << std::endl;
    std::cout << "Old way: Each context needs its own module copy" << std::endl;

    // Create two contexts
    CUcontext ctx1, ctx2;
    CHECK_CU(cuCtxCreate(&ctx1, 0, device));
    CHECK_CU(cuCtxCreate(&ctx2, 0, device));

    std::cout << "✓ Created 2 contexts: " << ctx1 << ", " << ctx2 << std::endl;

    std::cout << "\n=== Part 2: New cuLibrary (context-independent) ===" << std::endl;
    std::cout << "New way: Load once, use in multiple contexts" << std::endl;

    const char* fatbinPath = "test_kernel.fatbin";
    if (argc > 1) {
        fatbinPath = argv[1];
    }

    auto fatbinData = readFile(fatbinPath);

    if (fatbinData.empty()) {
        std::cout << "\n⚠ No fatbin file found: " << fatbinPath << std::endl;
        std::cout << "  Run 'make fatbin' to create test_kernel.fatbin" << std::endl;
        std::cout << "\nDemo will show API usage anyway...\n" << std::endl;
    }

    std::cout << "\n=== Part 3: Module Cache Pattern ===" << std::endl;
    std::cout << "This is perfect for your CLC framework!" << std::endl;

    ModuleCache cache;

    if (!fatbinData.empty()) {
        std::string cacheKey = "vectorAdd_v1";

        // First context loads the library
        std::cout << "\n[Context 1] Requesting library..." << std::endl;
        CUlibrary lib1 = cache.getOrLoad(cacheKey, fatbinData);

        if (lib1) {
            runInContext(ctx1, lib1, 1);

            // Second context reuses the same library!
            std::cout << "\n[Context 2] Requesting library..." << std::endl;
            CUlibrary lib2 = cache.getOrLoad(cacheKey, fatbinData);

            runInContext(ctx2, lib2, 2);

            // Verify they got the same library
            if (lib1 == lib2) {
                std::cout << "\n✓ Both contexts used the SAME library instance!" << std::endl;
                std::cout << "  This is context-independent loading in action" << std::endl;
            }

            std::cout << "\nCache stats:" << std::endl;
            std::cout << "  Entries: " << cache.size() << std::endl;

            cache.release(cacheKey);
            cache.release(cacheKey);
        }
    }

    std::cout << "\n=== Part 4: Practical Use Case for CLC Framework ===" << std::endl;
    std::cout << R"(
Scenario: Multi-tenant GPU server with your CLC proxy

1. Setup:
   - Compile CLC skeleton + policies to fatbin
   - Store in cache: cache.getOrLoad("CLC_skeleton_v1.2", binary)

2. When app launches kernel:
   - Intercept via proxy libcuda.so (using cuGetProcAddress)
   - Check cache for wrapped version
   - If miss: wrap user kernel + load to cache
   - If hit: reuse cached library

3. Benefits:
   - Each tenant/process gets own context
   - But all share the same CLC library instances
   - Lower memory overhead
   - Faster cold starts

4. Code pattern:

   // In your hook layer
   CUresult hooked_cuLaunchKernel(CUfunction f, ...) {
       // Determine if this kernel needs wrapping
       std::string wrapperKey = getKernelWrapperKey(f);

       // Get or create wrapped version
       CUlibrary wrappedLib = g_moduleCache.getOrLoad(
           wrapperKey,
           [&]() {
               return buildWrappedKernel(f, currentPolicy);
           }
       );

       // Get wrapped kernel and launch
       CUkernel wrappedKernel;
       cuLibraryGetKernel(&wrappedKernel, wrappedLib, "wrapped");
       return real_cuLaunchKernel(wrappedKernel, ...);
   }
    )" << std::endl;

    std::cout << "\n=== Part 5: Performance Comparison ===" << std::endl;
    std::cout << "Scenario: 100 contexts need the same kernel" << std::endl;
    std::cout << "\nOld (cuModule per-context):" << std::endl;
    std::cout << "  - 100 cuModuleLoadData calls" << std::endl;
    std::cout << "  - 100x memory for module code" << std::endl;
    std::cout << "  - Slower initialization" << std::endl;
    std::cout << "\nNew (cuLibrary shared):" << std::endl;
    std::cout << "  - 1 cuLibraryLoadData call" << std::endl;
    std::cout << "  - 1x memory for module code" << std::endl;
    std::cout << "  - Fast context attachment" << std::endl;
    std::cout << "  ✓ 100x memory savings!" << std::endl;

    // Cleanup
    cache.clear();
    CHECK_CU(cuCtxDestroy(ctx1));
    CHECK_CU(cuCtxDestroy(ctx2));

    std::cout << "\n========================================" << std::endl;
    std::cout << "Key Takeaways:" << std::endl;
    std::cout << "1. cuLibrary* - Context-independent loading" << std::endl;
    std::cout << "2. Load once, use in multiple contexts" << std::endl;
    std::cout << "3. Perfect for building module caches" << std::endl;
    std::cout << "4. Essential for multi-tenant scenarios" << std::endl;
    std::cout << "5. Combine with cuGetProcAddress for full proxy" << std::endl;
    std::cout << "\nFor your CLC framework:" << std::endl;
    std::cout << "  → Cache CLC skeletons globally" << std::endl;
    std::cout << "  → Each tenant/context uses cached instances" << std::endl;
    std::cout << "  → Much lower memory overhead" << std::endl;
    std::cout << "  → Faster initialization" << std::endl;
    std::cout << "========================================" << std::endl;

    return EXIT_SUCCESS;
}
