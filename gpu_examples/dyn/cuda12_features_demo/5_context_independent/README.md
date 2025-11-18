# Context-Independent Loading Demo (CUDA 12+)

## Overview

Demonstrates **cuLibrary*** APIs - context-independent module loading that allows sharing loaded code across multiple CUDA contexts, dramatically reducing memory overhead and initialization time.

## What This Demo Shows

1. **cuLibraryLoadData()** - Load module once, use in multiple contexts
2. **Context Independence** - No need to load per-context
3. **Module Cache Pattern** - Efficient caching for multi-tenant scenarios
4. **Memory Savings** - Share code across contexts
5. **Fast Initialization** - Reuse cached libraries

## Key Features

- Load CUDA modules once, use in many contexts
- Massive memory savings in multi-context scenarios
- Perfect for multi-tenant GPU servers
- Fast context attachment vs traditional per-context loading
- Reference counting for automatic cleanup

## Building and Running

```bash
make          # Build the demo
make run      # Build and run
make clean    # Clean build artifacts
```

## How It Works

### Traditional Way (cuModule - Context-Specific)
```c
// Each context needs its own copy
cuCtxSetCurrent(ctx1);
cuModuleLoadData(&module1, binary);  // Load #1

cuCtxSetCurrent(ctx2);
cuModuleLoadData(&module2, binary);  // Load #2 (same code!)
```

Result: 2× memory usage for the same code

### New Way (cuLibrary - Context-Independent)
```c
// Load once
CUlibrary library;
cuLibraryLoadData(&library, binary, ...);

// Use in any context
cuCtxSetCurrent(ctx1);
cuLibraryGetKernel(&kernel, library, "myKernel");
cuLaunchKernel(kernel, ...);

cuCtxSetCurrent(ctx2);
cuLibraryGetKernel(&kernel, library, "myKernel");  // Same library!
cuLaunchKernel(kernel, ...);
```

Result: 1× memory usage, shared across contexts

## Module Cache Pattern

The demo implements a production-ready cache pattern:

```c
class ModuleCache {
    std::unordered_map<std::string, CachedLibrary> cache;

    CUlibrary getOrLoad(const std::string& key,
                       std::function<std::vector<char>()> loader) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            it->second.refCount++;  // Cache hit
            return it->second.library;
        }

        // Cache miss - load and cache
        auto binary = loader();
        CUlibrary library;
        cuLibraryLoadData(&library, binary.data(), ...);
        cache[key] = {library, 1};
        return library;
    }

    void release(const std::string& key) {
        // Reference counting with auto-cleanup
        if (--cache[key].refCount == 0) {
            cuLibraryUnload(cache[key].library);
            cache.erase(key);
        }
    }
};
```

## Use Cases

- **Multi-Tenant GPU Servers**: Each tenant gets own context, shares code
- **Multi-Process GPU Apps**: Share libraries across processes
- **Plugin Systems**: Cache frequently used plugins
- **CLC Frameworks**: Cache policy wrappers globally
- **Resource Optimization**: Minimize GPU memory footprint

## Output Example

```
=== Part 3: Module Cache Pattern ===

[Context 1] Requesting library...
  [CACHE MISS] Loading new library for: vectorAdd_v1

[Context 2] Requesting library...
  [CACHE HIT] Reusing existing library for: vectorAdd_v1
  RefCount: 2

✓ Both contexts used the SAME library instance!

Cache stats:
  Entries: 1
  [CACHE] Released reference (refCount: 1)
  [CACHE] Evicted: vectorAdd_v1
```

## Performance Comparison

**Scenario**: 100 contexts need the same kernel

| Approach | Load Calls | Memory Usage | Initialization |
|----------|------------|--------------|----------------|
| **cuModule** (old) | 100× | 100× | Slow |
| **cuLibrary** (new) | 1× | 1× | Fast |

**Memory savings**: 100× less!

## Related APIs

- `cuLibraryLoadData()` - Load from memory (context-independent)
- `cuLibraryLoadFromFile()` - Load from file
- `cuLibraryGetKernel()` - Get kernel from library
- `cuLibraryEnumerateKernels()` - List all kernels
- `cuLibraryGetModule()` - Get context-specific module handle
- `cuLibraryUnload()` - Unload library

## Integration with CLC Framework

Perfect for CLC (Compile-Load-Check) policy caching:

```c
// Global cache shared across all contexts
static ModuleCache g_policyCache;

CUresult hooked_cuLaunchKernel(CUfunction f, ...) {
    // Determine wrapper needed
    std::string policyKey = getKernelPolicyKey(f);

    // Get or build wrapped version
    CUlibrary wrappedLib = g_policyCache.getOrLoad(
        policyKey,
        [&]() {
            // Build wrapped kernel with policy
            return buildPolicyWrapper(f, currentPolicy);
        }
    );

    // Get wrapped kernel
    CUkernel wrappedKernel;
    cuLibraryGetKernel(&wrappedKernel, wrappedLib, "wrapped");

    // Launch wrapped version
    return real_cuLaunchKernel(wrappedKernel, ...);
}
```

Benefits:
- Each tenant/process gets own context
- All share the same CLC policy wrapper libraries
- Lower memory overhead
- Faster cold start times

## Multi-Tenant Server Example

```
GPU Server
├── Tenant 1 (Context 1) ─┐
├── Tenant 2 (Context 2) ─┼── All use cached library
├── Tenant 3 (Context 3) ─┤
└── Tenant 4 (Context 4) ─┘

Single CLC Wrapper Library (loaded once)
  ├── Policy enforcement code
  ├── Resource management
  └── Monitoring hooks
```

## Context-Specific vs Context-Independent

### When to Use cuModule (Context-Specific)
- Single context applications
- Context-specific optimizations needed
- Legacy code compatibility

### When to Use cuLibrary (Context-Independent)
- Multi-context applications
- Multi-tenant systems
- Plugin/policy frameworks
- Memory-constrained environments
- Fast initialization required

## Cache Key Strategies

Choose cache keys based on your needs:

```c
// 1. Simple version-based
cacheKey = "wrapper_v1.2.3";

// 2. Policy-based
cacheKey = "policy_" + policyHash;

// 3. Kernel-specific
cacheKey = "kernel_" + kernelName + "_" + policyVersion;

// 4. Tenant-specific
cacheKey = "tenant_" + tenantId + "_wrapper";
```

## Memory Layout Comparison

**Traditional (cuModule)**:
```
Context 1: [Module Copy 1]  ← 10 MB
Context 2: [Module Copy 2]  ← 10 MB
Context 3: [Module Copy 3]  ← 10 MB
Total: 30 MB
```

**Context-Independent (cuLibrary)**:
```
Global: [Shared Library]  ← 10 MB
Context 1: [Reference] ────┘
Context 2: [Reference] ────┘
Context 3: [Reference] ────┘
Total: 10 MB
```

## Requirements

- CUDA 12.0 or later
- Driver API: `-lcuda`
- Multiple contexts for full benefit demonstration

## Thread Safety

The demo's cache implementation includes thread safety:
```c
std::mutex cacheMutex;
std::lock_guard<std::mutex> lock(cacheMutex);
```

Important for multi-threaded applications where different threads manage different contexts.

## Best Practices

1. **Use Reference Counting**: Track library usage to know when to unload
2. **Version Cache Keys**: Include version info in keys for safe updates
3. **Implement Eviction**: Add LRU or similar policy for large caches
4. **Monitor Memory**: Track cache size to prevent unbounded growth
5. **Thread Safety**: Protect cache with mutexes in multi-threaded apps

## Combining with Other CUDA 12 Features

This pairs perfectly with:
- **cuGetProcAddress** - Hook kernel launches
- **nvJitLink** - Build wrapped kernels
- **nvFatbin** - Package multi-arch code
- **cudaLibraryLoad*** - Runtime dynamic loading

Together, these enable complete dynamic policy frameworks!
