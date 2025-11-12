# UVMBench: A Comprehensive Benchmark Suite for Researching Unified Virtual Memory in GPUs

**Authors:** Yongbin Gu, Wenxuan Wu, Yunfan Li and Lizhong Chen
**Affiliation:** School of Electrical Engineering and Computer Science, Oregon State University, Corvallis, USA
**Contact:** {guyo, wuwen, liyunf, chenliz}@oregonstate.edu
**arXiv:** 2007.09822v2 [cs.AR] 20 Oct 2020

---

## Abstract

The recent introduction of Unified Virtual Memory (UVM) in GPUs offers a new programming model that allows GPUs and CPUs to share the same virtual memory space, which shifts the complex memory management from programmers to GPU driver/hardware and enables kernel execution even when memory is oversubscribed. Meanwhile, UVM may also incur considerable performance overhead due to tracking and data migration along with special handling of page faults and page table walk. As UVM is attracting significant attention from the research community to develop innovative solutions to these problems, in this paper, we propose a comprehensive UVM benchmark suite named UVMBench to facilitate future research on this important topic. The proposed UVMBench consists of 32 representative benchmarks from a wide range of application domains. The suite also features unified programming implementation and diverse memory access patterns across benchmarks, thus allowing thorough evaluation and comparison with current state-of-the-art. A set of experiments have been conducted on real GPUs to verify and analyze the benchmark suite behaviors under various scenarios.

---

## I. INTRODUCTION

GPUs have been gaining great attention in accelerating traditional and emerging workloads, such as machine learning, bioinformatics, electrodynamics, etc. due to GPU’s massively parallel computing capability. However, there are two major issues in the mainstream GPU programming model that severely limit further utilization. First, the physical memory separation between a GPU and a CPU requires explicit memory management in conventional GPU programming model. Programmers have to explicitly copy data between CPU and GPU memories to the location where the data is used (i.e. copy-then-execute). Second, the conventional GPU programming model does not allow a kernel to be executed if it needs more memory that what the GPU memory can provide (i.e., memory oversubscription). This has greatly limited the use of GPUs in large data-intensive machine learning applications nowadays.

Recently, GPU vendors have proposed and started to employ a new approach, **Unified Virtual Memory (UVM)**, in the newly released products. UVM allows GPUs and CPUs to share the same virtual memory space, and offloads memory management to the GPU driver and hardware, thus eliminating explicit copy-then-execute by the programmers. The GPU driver and underlying hardware automatically migrate the needed data to destinations. Moreover, UVM enables GPU kernel execution while memory is oversubscribed by automatically evicting data that is no longer needed in the GPU memory to the CPU side.

This is extremely important and helpful in facilitating large workloads (especially deep learning models) and GPU virtualization [9], [12] with limited memory sizes.

### UVM Challenges

However, the advantages of UVM may come at a price. Analogous to virtual machines that offer great flexibility over physical machines but sacrifice performance in some degree, UVM also incurs performance overhead. In order to implement automatic data migration between a CPU and a GPU, the GPU driver and the GPU Memory Management Unit (MMU) have to track data access information and determine the granularity of data migration over the PCIe link. This may reduce performance. For example, UVM needs special page table walk and page fault handling that introduce extra latency for memory accesses in GPUs. In addition, the fluctuated page migration granularity may also under-utilize PCIe bandwidth.

### Motivation for UVMBench

Due to the large potential benefits of UVM and its associated performance issues, UVM has recently drawn significant attention from the research community. Several optimization techniques have been proposed to mitigate the side effects of UVM. However, most prior works have used their own modified versions of existing benchmark suites (e.g., Rodinia, Parboil, Polybench) or several in-house workloads. Our further inspection of these benchmarks shows that they lack unified implementation and no paper so far has provided a thorough analysis of the memory behaviors of these benchmarks. This can be a serious limitation for researchers and developers who aim to propose new optimizations for UVM and who would like to make comparison with existing research works.

The earliest work is Zheng et al. [24], which enables on-demand GPU memory and proposes prefetching techniques to improve UVM performance. As the work predates the release of UVM, the developed on-demand memory APIs are quite different from the version in the current UVM practice. More recently, Ganguly et al. [8], Yu et al. [22] and Li et al. [11] study prefetching and/or eviction techniques for UVM in more detail. However, their evaluation includes only benchmarks with limited number of access patterns, which makes it difficult to assess the effectiveness of their schemes on a broader range of benchmarks with diverse memory access patterns. In fact, comprehensive benchmarks (or the lack thereof) have become a common issue in these and other prior works on GPU UVM.

The developed benchmarks are evaluated on a Nvidia GTX 1080 Ti GPU with 11GB memory capacity. The code volume is reduced by removing explicit memory management APIs thanks to UVM. Evaluation results show that, if we directly implement/convert benchmarks to the UVM programming model, there is an average of 34.2% slowdown than the non-UVM benchmarks. However, if we augment with proper manual optimizations on data prefetching and data reuse, the performance can be restored to almost the same as the non-UVM programming model. This indicates that there is substantial room for UVM research on developing autonomous memory management to close the gap between UVM and non-UVM models and possibly exceed the performance of non-UVM.

We have discussed the importance of GPU UVM research and the motivation for a benchmark suite in this section. In the remaining of this paper, Section II describes the proposed benchmark suite in more detail. Section III explains our evaluation methodology. Section IV presents and analyzes test results. Key observations drawn from the results and suggestions for future UVM research are highlighted sporadically in that section. Finally, Section V concludes the paper.

### Main Contributions

The main contributions of this paper are the following:

- Identifying the need for a benchmark suite for UVM
- Developing a comprehensive UVM benchmark suite to facilitate the research on UVM
- Profiling memory access patterns of the benchmark suite, and studying the relevance of the patterns to performance under memory oversubscription
- Conducting thorough analysis of performance difference between the UVM and non-UVM programming models

---

## II. UVMBENCH

Benchmarks play an important role in evaluating the effectiveness and generalization when an architecture optimization is proposed. We develop a comprehensive UVM benchmark suite to facilitate the research on the GPU UVM. This suite covers a wide range of application domains marked in Table I. The benchmarks exhibit diverse memory access patterns (more in Section IV-A) to help evaluate memory management strategies in GPU UVM. The suite also includes several auxiliary python-based programs to help create and test memory oversubscription cases. The benchmark suite is referred to as **UVMBench**, and has been made available to the GPU research community for both non-UVM and UVM versions at: https://github.com/OSU-STARLAB/UVM_benchmark

### Benchmark List

The UVMBench suite consists of 32 benchmarks across various domains:

| Application | Abbr. | Domain | Kernels | Threads Per Block | Type |
|-------------|-------|--------|---------|-------------------|------|
| 2D Convolution | 2DCONV | Machine Learning | 1 | 256 | R |
| 2 Matrix Multiplications | 2MM | Linear Algebra | 2 | 256 | R |
| 3D Convolution | 3DCONV | Machine Learning | 1 | 256 | R |
| 3 Matrix Multiplications | 3MM | Linear Algebra | 3 | 256 | R |
| Matrix Transpose Vector Multiplication | ATAX | Linear Algebra | 2 | 256 | I |
| Backpropagation | BACKPROP | Machine Learning | 2 | 256 | R |
| Breadth First Search | BFS | Graph Theory | 6 | 1024 | I |
| BiCGStab Linear Solver | BICG | Linear Algebra | 2 | 256 | I |
| Bayesian Network | BN | Machine Learning | 2 | 256 | R |
| Convolutional Neural Network | CNN | Machine Learning | 6 | 64 | R |
| Correlation Computation | CORR | Statistics | 4 | 256 | I |
| Covariance Computation | COVAR | Statistics | 3 | 256 | I |
| Discrete Wavelet Transform 2D | DWT2D | Media Compression | 2 | 256 | R |
| 2-D Finite Different Time Domain | FDTD-2D | Electrodynamics | 3 | 256 | I |
| Gaussian Elimination | GAUSSIAN | Linear Algebra | 2 | 512/16 | I |
| Matrix-multiply | GEMM | Machine Learning | 1 | 256 | I |
| Scalar, Vector Matrix Multiplication | GESUMMV | Machine Learning | 1 | 256 | I |
| Gram-Schmidt decomposition | GRAMSCHM | Linear Algebra | 3 | 256 | I |
| HotSpot | HOTSPOT | Physics Simulation | 1 | 256 | R |
| HotSpot 3D | HOTSPOT3D | Physics Simulation | 1 | 256 | R |
| Kmeans | KMEANS | Machine Learning | 5 | 1/3 | I |
| K-Nearest Neighbors | KNN | Machine Learning | 4 | 256 | R |
| Logistic Regression | LR | Machine Learning | 1 | 128 | R |
| Matrix Vector Product Transpose | MVT | Linear Algebra | 2 | 256 | I |
| Needleman-Wunsch | NW | Bioinformatics | 2 | 16 | I |
| Particle Filter | PFILTER | Medical Imaging | 1 | 128 | R |
| Pathfinder | PATHFINDER | Grid Traversal | 1 | 256 | R |
| Speckle Reducing Anisotropic Diffusion | SRAD | Image Processing | 2 | 256 | R |
| Stream Cluster | SC | Data Mining | 1 | 512 | I |
| Support Vector Machine | SVM | Machine Learning | 2 | 1024 | I |
| Symmetric rank-2k operations | SYR2K | Linear Algebra | 1 | 256 | I |
| Symmetric rank-k operations | SYRK | Linear Algebra | 1 | 256 | R |

**Type:** R = Regular memory access pattern, I = Irregular memory access pattern

### Comparison with Other Benchmarks

| Benchmarks / Benchmark Suite | # of Workloads | Test in Real Hardware | Machine Learning Workloads | Diverse Memory Access Patterns | Oversubscription Support |
|------------------------------|----------------|----------------------|---------------------------|-------------------------------|-------------------------|
| Workloads in [8] | 14 | ✗ | ✗ | ✗ | ✓ |
| Workloads in [5] | 6 | ✓ | ✗ | ✗ | ✓ |
| Nvidia SDK [2] | 1 | ✓ | ✗ | ✗ | ✗ |
| **UVMBench** | **32** | **✓** | **✓** | **✓** | **✓** |

### Development Efforts

The development of the benchmark suite includes the following major efforts:

#### (1) Re-implement existing benchmarks

We start with combining three existing popular GPU benchmark suites (Rodinia, Parboil, and Polybench), removing redundant workloads and workload types, and converting into the UVM-based programming model. To implement UVM for these benchmarks, we replace all the host pointers (CPU side) and device pointers (GPU side) with a unified pointer allocated by the UVM API `cudaMallocManaged`. Also, because the GPU driver is now responsible for data migration, all the explicit memory data migration APIs in each original program need to be removed. This may involve rewriting part of the code around the API calls in some benchmarks to achieve the equivalent functionalities. Moreover, the non-UVM data allocation structure should be adapted to the UVM version (e.g., flatten 2D arrays into 1D arrays, as no 2D array allocation API is provided in the UVM programming model).

#### Code Example: SVM Benchmark

**Listing 1: Sigma Update function in SVM with non-UVM**
```c
Sigma_update(int *iters, float *alpha, float *sigma, float *K, int *y, int l, int C)
{
    // Define variables on the device
    float *dev_alpha = 0;
    float *dev_sigma = 0;
    float *dev_K = 0;
    int *dev_y = 0;
    int *dev_block_done = 0;
    float *dev_delta = 0;
    void *args[10] = {&dev_iters, &dev_alpha, &dev_sigma, &dev_K, &dev_y,
                      &dev_block_done, &grid_dimension, &dev_delta, &l, &C};

    // Allocate memory space on the device memory
    cudaMalloc(&dev_iters, sizeof(int));
    cudaMalloc(&dev_alpha, l*sizeof(float));
    cudaMalloc(&dev_sigma, l*sizeof(float));
    cudaMalloc(&dev_K, l*l*sizeof(float));
    cudaMalloc(&dev_y, l*sizeof(int));
    cudaMalloc(&dev_block_done, grid_dimension*sizeof(int));
    cudaMalloc(&dev_delta, 1*sizeof(float));

    // Data migration: Host to Device
    cudaMemcpy(dev_K, K, l*l*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, l*sizeof(int), cudaMemcpyHostToDevice);

    /*Kernel Launch*/

    // Data migration: Device to Host
    cudaMemcpy(iters, dev_iters, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(alpha, dev_alpha, l*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sigma, dev_sigma, l*sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory space
    cudaFree(dev_block_done);
    cudaFree(dev_delta);
    cudaFree(dev_y);
    cudaFree(dev_K);
    cudaFree(dev_sigma);
    cudaFree(dev_alpha);
    cudaFree(dev_iters);
}
```

**Listing 2: Sigma Update function in SVM with UVM**
```c
Sigma_update(int *iters, float *alpha, float *sigma, float *K, int *y, int l, int C)
{
    int *dev_block_done = 0;
    float *dev_delta = 0;
    void *args[10] = {&iters, &alpha, &sigma, &K, &y,
                      &dev_block_done, &grid_dimension, &dev_delta, &l, &C};

    cudaMallocManaged(&dev_block_done, grid_dimension*sizeof(int));
    cudaMallocManaged(&dev_delta, 1*sizeof(float));

    /*Kernel Launch*/

    cudaFree(dev_block_done);
    cudaFree(dev_delta);
}
```

The UVM programming model greatly reduces the code complexity by eliminating explicit memory management.

Listings 1 and 2 show the partial code of the sigma update function in the SVM benchmark, which demonstrates the re-implementation process and newly added benchmarks. Several unrelated variables are omitted for simplicity. Listing 1 is the code without UVM, while Listing 2 is the code with UVM during runtime. As the traditional programming model requires explicit memory management, the program in Listing 1 has to allocate memory space on the device by calling `cudaMalloc` (lines 12–20). It also needs to call `cudaMemcpy` APIs (lines 22–24 and lines 28–31) before and after the kernel launch to explicitly migrate the required data between the host and the device. In contrast, the UVM programming model in Listing 2 unifies the memory space of the host and the device. By calling `cudaMallocManaged` APIs (lines 6–7), the code allocates bytes of managed memory. The allocated variables can be accessed by the host and the device directly, and are managed by the Unified Memory system of the GPU. In Listing 2, when this Sigma update function is called in the main function (line 1), the variables, defined by `cudaMallocManaged`, are passed into the function, and the device kernels can directly access these variables. Therefore, device variable definitions and memory management APIs are removed (i.e., lines 6–7 in Listing 2 vs. lines 12–20 & 22–24 & 28–31 in Listing 1). It can be seen that the UVM programming model greatly reduces the code complexity.

#### (2) Develop machine learning workloads

We add more machine learning related workloads:

- **Bayesian Network (BN)**: A probabilistic-based graphical model for predicting the likelihood of several possible causes given the occurrence of an event. Based on the SJTU version, retaining two GPU-accelerated phases: preprocessing and score calculation.

- **Convolutional Neural Network (CNN)**: Most commonly applied to image recognition. Includes forward propagation (convolutional operations, activation operations, fully connected operations) and back propagation (error calculations, weight and bias update operations) accelerated on GPU.

- **Logistic Regression (LR)**: Used to predict the probability of the existence of a certain class or event. The cost calculation is accelerated on the GPU. Uses document-level sentiment polarity annotations.

- **Support Vector Machine (SVM)**: Finds support vectors that form a hyper plane to separate different classes. The kernel matrix calculation is accelerated on the GPU. Based on the Julia project and converted to UVM.

This would help researchers to understand better the role that GPU UVM plays in machine learning acceleration.

#### (3) Optimize data prefetch

We add asynchronous prefetching optimization before each kernel launch by calling `cudaMemPrefetchAsync`. This optimization exemplifies that hardware prefetchers may bring considerable performance improvement in UVM. Users can easily enable or disable this optimization by changing the macro definition in the Makefile.

**Listing 3: Enable Prefetching in Backprop with UVM**
```c
// Create streams for asynchronous prefetch
cudaStream_t stream1;
cudaStream_t stream2;
cudaStream_t stream3;
cudaStream_t stream4;
cudaStream_t stream5;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamCreate(&stream3);
cudaStreamCreate(&stream4);
cudaStreamCreate(&stream5);

cudaMemPrefetchAsync(input_cuda, (in + 1)*sizeof(float), 0, stream1);
cudaMemPrefetchAsync(output_hidden_cuda, (hid + 1)*sizeof(float), 0, stream2);
cudaMemPrefetchAsync(input_hidden_cuda, (in + 1)*(hid + 1)*sizeof(float), 0, stream3);
cudaMemPrefetchAsync(hidden_partial_sum, num_blocks*WIDTH*sizeof(float), 0, stream4);

// Performing GPU computation
bpnn_layerforward_CUDA<<<grid, threads, 0, stream5>>>(
    input_cuda, output_hidden_cuda, input_hidden_cuda, hidden_partial_sum, in, hid);
cudaDeviceSynchronize();

cudaMemPrefetchAsync(input_prev_weights_cuda, (in + 1)*(hid + 1)*sizeof(float), 0, stream1);
cudaMemPrefetchAsync(hidden_delta_cuda, (hid + 1)*sizeof(float), 0, stream2);

bpnn_adjust_weights_cuda<<<grid, threads, 0, stream5>>>(
    hidden_delta_cuda, hid, input_cuda, in, input_hidden_cuda, input_prev_weights_cuda);
cudaDeviceSynchronize();
```

Listing 3 shows the code in the Backprop benchmark after enabling the above asynchronous prefetching. The program uses CUDA streams to manage concurrency in GPU applications. Different streams can execute their corresponding commands concurrently. To prepare asynchronous prefetching, it first creates different streams (lines 2–11). With different streams, the prefetching APIs (lines 13–16 and 23–24) prefetch the required data asynchronously. As the data have been fetched in the device before the kernel is launched, the Unified Memory system does not need to stall the kernel and handle page faults. Therefore, the data migration overhead in the UVM is mitigated under asynchronous prefetching.

#### (4) Optimize data reuse

Data reuse can also mitigate performance overhead of UVM. If useful data resides in the device memory for longer time, fewer page faults may occur. We add the option to run multiple iterations of a kernel execution to create data reuse opportunities. Users can change the number of iterations (≥ 1) by modifying the macro in each benchmark program file.

Benchmarks in the proposed UVMBench are all implemented in CUDA and can be run on Nvidia GPUs. This suite includes both the non-UVM version (original) and the UVM version implementation for performance comparison. There are no algorithmic changes when developing the UVM version of the benchmarks. This ensures fair comparison between the traditional programming model and the UVM programming model. Consequently, the observed performance changes are mostly attributed to the difference between programming models rather the algorithms.

---

## III. EVALUATION METHODOLOGY

To conduct the above experiments, we employ an Nvidia GTX 1080 Ti GPU with the Pascal architecture. We use the Nvidia Binary Instrumentation Tool (NVBit) to extract the global memory access patterns of the UVMBench suite and two Nvidia official profiling tools to profile the performance related data of benchmarks.

### Evaluation Platform Setup

| Component | Specification |
|-----------|--------------|
| CPU | Intel Xeon E5-2630 V4 10 Cores 2.2 GHz |
| Memory | DDR4 16GB x 4 |
| PCIe | PCIe Gen3x16 16GB/s |
| Operating System | Ubuntu 18.04 64bit |
| GPU | Nvidia GTX1080Ti |
| Driver version | 440.33.01 |
| CUDA | CUDA 10.2 |
| Profiling Tools | nvprof, Nvidia Visual Profiler, NVBit |

### Methodology

Our evaluation methodology is designed to enable a set of experiments that test the proposed benchmark suite:

1. **Memory access pattern profiling**: Use Nvidia Binary Instrumentation Tool (NVBit) to extract global memory access patterns
2. **Performance comparison**: Direct comparison between UVM and non-UVM implementations
3. **PCIe bandwidth impact**: Examine the impact on PCIe bandwidth under UVM
4. **Memory oversubscription**: Evaluate UVM performance under memory oversubscription scenarios

---

## IV. RESULTS AND ANALYSIS

### A. Memory Access Pattern Profiling

To study the relationship between memory behaviors and UVM efficiency, we first profile memory access patterns of each benchmark. NVBit is used to generate memory reference traces by injecting the instrumentation function before performing each global load/store. The memory traces show that benchmarks in the UVMBench suite exhibit diverse memory access patterns.

**Classification:**
- **Regular (R)**: Benchmarks access only a small number of memory pages at any point of time
- **Irregular (I)**: Benchmarks with large unique memory pages access at a given time

**Regular benchmarks** exhibit a streaming access pattern (e.g., 2DCONV, 2MM, 3DCONV, BACKPROP). These benchmarks access only a small number of memory addresses and seldom exhibit data reuse within the kernel.

**Irregular benchmarks** show very different memory access patterns:
- Accessing many memory addresses at a given time (e.g., ATAX, BICG, GAUSSIAN)
- Repeatedly accessing the same memory address over time (e.g., COVAR, GRAMSCHM)
- Accessing random addresses (e.g., SC, SVM)

Note that benchmark NW is classified as irregular, as it exhibits a sparse, localized and repeated memory accesses, although this is not quite visible in the figure due to the scale.

### B. UVM vs. non-UVM Performance

#### a. Performance of Direct UVM Conversion

Direct conversion to UVM shows an **average of 34.2% slowdown** compared with the non-UVM version. These results are expected as page fault handling causes large performance overhead for kernel execution. Under UVM, when required data does not reside in the GPU DRAM (page fault occurrence), the kernel has to be stalled while waiting for the data to be fetched from the CPU side.

**Benchmarks with most significant performance drop:**
- 2DCONV
- BACKPROP
- HOTSPOT
- GESUMMV
- PATHFINDER

For these benchmarks, data migration time accounts for majority of the entire execution (over 80%), and their kernels have little to no data reuse and are only invoked once.

#### b. Restoring UVM Performance via Data Reuse

Data reuse can mitigate UVM performance degradation by reducing the occurrence of page faults. Experiments show that performance of benchmarks under UVM rapidly improves with more kernel invocations and eventually approaches the performance of non-UVM. Except for the first executed kernel, following kernels may reuse the data that has been fetched during the execution of the first kernel, and fewer page faults would occur.

**Observation/Suggestion:** Although data reuse is artificially introduced in the software program in this experiment, it prompts us that if applications exhibit significant data reuse opportunities, either inherent or created through architecture optimizations, UVM can be an attractive model that provides flexibility while having little performance overhead.

#### c. Restoring UVM Performance via Data Prefetch

Using asynchronous prefetching via `cudaMemPrefetchAsync`, the performance of benchmarks improves considerably and is close to the performance of the non-UVM version. The geometric mean of the slowdown decreased from **95.8% to merely 0.7%**. The improvement comes from the fact that kernel execution is now rarely stalled as data has already been fetched in the device memory before being accessed.

**Observation/Suggestion:** Besides data reuse, another alternative to restore performance degradation of UVM is data prefetching by employing the runtime API cudaAsyncPrefetch. In theory, page faults can be completely eliminated if there is an oracle prefetcher that is able to load any required data into the GPU memory before the data is accessed.

It is important to note that we achieve data reuse and data prefetch by manually modifying the software programs. What is needed is innovation in architecture research that can achieve similar level of data reuse and prefetch but is transparent to programmers.

### C. Effect of Data Migration on PCIe Bandwidth

On average, the achieved PCIe bandwidth of UVM is **15.2% lower** than that of non-UVM. In general, the larger the transferred data size is, the higher the effective PCIe bandwidth can achieve. Since the non-UVM model copies the entire allocated data chunk to the GPU memory before execution, this results in relatively high effective bandwidth. In contrast, the migrated data size in UVM is usually much smaller than the non-UVM one as only on-demand data is migrated through the PCIe bus (usually smaller than 1MB).

Note that benchmarks BN and CNN in UVM and non-UVM both exhibit low effective PCIe bandwidth, because the sizes of allocated variables in these two benchmarks are all small (less than 4KB), and even the entire chunk of allocated variable transmission cannot fully utilize the PCIe bandwidth.

The variation in effective PCIe bandwidth among UVM benchmarks is mainly caused by the hardware prefetcher inside the GPU. Nvidia has implemented a tree-based hardware prefetcher in their GPUs, which heuristically adjusts the prefetching granularity based on access locality. The difference in memory access patterns across benchmarks puts the hardware prefetcher in different degrees of efficacy.

**Observation/Suggestion:** The above results on the effective PCIe bandwidth indicate that hardware prefetchers currently employed in GPUs cannot fully utilize PCIe bandwidth. Future research is much needed to continue developing and optimizing GPU hardware prefetchers that are UVM-aware.

### D. Oversubscription

A major advantage of UVM is to enable kernel execution when memory is oversubscribed. Performance under memory oversubscription can be significantly reduced since part of the data now needs to be brought from the CPU memory. To quantify the performance degradation, we run all benchmarks with required memory footprint set to be **110% and 125%** of the available memory space in the GPU physical memory. As different benchmarks have different required memory footprint, to create memory oversubscription, we modify the available memory space through the `cudaMalloc` runtime API during setup to emulate 110% and 125% usage.

**Results:**
- All benchmarks suffer considerable performance degradation under memory oversubscription
- The more memory is oversubscribed, the more performance degrades
- Many benchmarks can complete execution with **2-3x slowdown** under memory oversubscription
- Other benchmarks suffer from significant performance penalty or even crash (marked as >100X)

**Benchmarks with 2-3x slowdown** usually have a streaming access pattern. With this pattern and the LRU eviction policy in Nvidia GPUs, the evicted data does not affect kernel execution as the evicted data is not reused anymore. Therefore, the performance overhead mainly comes from the waiting time of page eviction.

**Benchmarks with large performance penalty** mainly suffer from severe page thrashing, which repeatedly migrates the page back and forth between the GPU and the CPU. This usually occurs when a benchmark has a short data reuse distance so the evicted data is needed/reused within a short time.

Note that oversubscription is not possible under non-UVM, which does not allow kernels to run at all if the memory is oversubscribed.

Note: Some benchmarks may even crash under oversubscription (e.g., LR uses the cuBLAS library which cannot support memory oversubscription and leads to crash), reflected as >100X in our plots.

**Observation/Suggestion:** The significant performance degradation under memory oversubscription suggests that the current eviction policies are doing a poor job at selecting the best candidate pages to evict, thus causing severe page thrashing and limiting the amount of memory that can be oversubscribed. This may be possibly because existing eviction policies are not designed specifically with supporting UVM in mind. We urge researchers to develop more effective eviction policies that can select evicted data more accurately or even proactively to make space for expected data accesses.

---

## V. CONCLUSION

The Unified Virtual Memory (UVM) programming model has been introduced recently in GPUs to ease the programming efforts and to allow kernel execution under memory oversubscription. This paper identifies the need for representative benchmarks for GPU UVM, and proposes a comprehensive benchmark suite to help researchers understand and study various aspects of GPU UVM. Several observations and suggestions have been drawn from the evaluation results to guide the much needed future research on UVM:

1. **Data reuse opportunities** can make UVM an attractive model with little performance overhead
2. **Prefetching techniques** can restore UVM performance to near non-UVM levels
3. **Hardware prefetchers** need to be optimized to fully utilize PCIe bandwidth
4. **Eviction policies** need to be redesigned specifically for UVM to reduce page thrashing

The UVMBench suite is available at: https://github.com/OSU-STARLAB/UVM_benchmark

---

## REFERENCES

[1] “Radeons next-generation vega architecture,” 2017. [Online]. Available: https://radeon.com/downloads/vega-whitepaper-11.6.17.pdf

[2] “Cuda toolkit,” 2020. [Online]. Available: https://developer.nvidia.com/cuda-downloads

[3] S. Che, M. Boyer, J. Meng, D. Tarjan, J. W. Sheaffer, S.-H. Lee, and K. Skadron, “Rodinia: A benchmark suite for heterogeneous computing,” in 2009 IEEE international symposium on workload characterization (IISWC). IEEE, 2009, pp. 44–54.

[4] S. Che, J. W. Sheaffer, M. Boyer, L. G. Szafaryn, L. Wang, and K. Skadron, “A characterization of the rodinia benchmark suite with comparison to contemporary cmp workloads,” in IEEE International Symposium on Workload Characterization (IISWC’10). IEEE, 2010, pp. 1–11.

[5] S. Chien, I. Peng, and S. Markidis, “Performance evaluation of advanced features in cuda unified memory,” in 2019 IEEE/ACM Workshop on Memory Centric High Performance Computing (MCHPC). IEEE, 2019.

[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

[7] D. Ganguly, Z. Zhang, J. Yang, and R. Melhem, “Adaptive page migration for irregular data-intensive applications under gpu memory oversubscription,” in Proc. of the Int. Conf. on Parallel and Distributed Processing (IPDPS).

[8] D. Ganguly, Z. Zhang, J. Yang, and R. Melhem, “Interplay between hardware prefetcher and page eviction policy in cpu-gpu unified virtual memory,” in Proceedings of the 46th International Symposium on Computer Architecture, 2019, pp. 224–235.

[9] M. Gu, Y. Park, Y. Kim, and S. Park, “Low-overhead dynamic sharing of graphics memory space in gpu virtualization environments,” Cluster Computing, pp. 1–12, 2019.

[10] H. Kim, J. Sim, P. Gera, R. Hadidi, and H. Kim, “Batch-aware unified memory management in gpus for irregular workloads,” in Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems, 2020, pp. 1357–1370.

[11] C. Li, R. Ausavarungnirun, C. J. Rossbach, Y. Zhang, O. Mutlu, Y. Guo, and J. Yang, “A framework for memory oversubscription management in graphics processing units,” in Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems, 2019, pp. 49–63.

[12] Q. Lu, J. Yao, H. Guan, and P. Gao, “gqos: A qos-oriented gpu virtualization with adaptive capacity sharing,” IEEE Transactions on Parallel and Distributed Systems, vol. 31, no. 4, pp. 843–855, 2019.

[13] A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts, “Learning word vectors for sentiment analysis,” in Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. Portland, Oregon, USA: Association for Computational Linguistics, June 2011, pp. 142–150. [Online]. Available: http://www.aclweb.org/anthology/P11-1015

[14] P. Markthub, M. E. Belviranli, S. Lee, J. S. Vetter, and S. Matsuoka, “Dragon: breaking gpu memory capacity limits with direct nvm access,” in SC18: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 2018, pp. 414–426.

[15] L.-N. Pouchet et al., “Polybench: The polyhedral benchmark suite,” URL: http://www.cs.ucla.edu/pouchet/software/polybench, 2012.

[16] Y. Qin, “a stochastic decomposition implementation of support-vector machine training,” https://github.com/qin-yu/julia-svm-gpu-cuda, 2019.

[17] N. Sakharnykh, “Unified memory on pascal and volta,” May 2017. [Online]. Available: http://on-demand.gputechconf.com/gtc/2017/presentation/s7285-nikolay-sakharnykh-unified-memory-on-pascal-and-volta.pdf

[18] J. A. Stratton, C. Rodrigues, I.-J. Sung, N. Obeid, L.-W. Chang, N. Anssari, G. D. Liu, and W.-m. W. Hwu, “Parboil: A revised benchmark suite for scientific and commercial throughput computing,” Center for Reliable and High-Performance Computing, vol. 127, 2012.

[19] O. Villa, M. Stephenson, D. Nellans, and S. W. Keckler, “Nvbit: A dynamic binary instrumentation framework for nvidia gpus,” in Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, 2019, pp. 372–383.

[20] Y. Wang, W. Qian, S. Zhang, X. Liang, and B. Yuan, “A learning algorithm for bayesian networks and its efficient implementation on gpus,” IEEE Transactions on Parallel and Distributed Systems, vol. 27, no. 1, pp. 17–30, 2015.

[21] W. Wu, Z. Qi, and L. Fuxin, “Pointconv: Deep convolutional networks on 3d point clouds,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 9621–9630.

[22] Q. Yu, B. Childers, L. Huang, C. Qian, and Z. Wang, “A quantitative evaluation of unified memory in gpus,” The Journal of Supercomputing, pp. 1–28, 2019.

[23] W. Zhang, M. Zhu, T. Gong, L. Xiao, L. Ruan, Y. Mei, Y. Sun, and X. Ji, “Performance degradation-aware virtual machine live migration in virtualized servers,” in 2012 13th International Conference on Parallel and Distributed Computing, Applications and Technologies. IEEE, 2012, pp. 429–435.

[24] T. Zheng, D. Nellans, A. Zulfiqar, M. Stephenson, and S. W. Keckler, “Towards high performance paged memory for gpus,” in 2016 IEEE International Symposium on High Performance Computer Architecture (HPCA). IEEE, 2016, pp. 345–357.
