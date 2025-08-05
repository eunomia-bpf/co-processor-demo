# trace NPU with bpftrace

```bash
sudo bpftrace -e 'kprobe:intel_vpu:ivpu_* { printf("%s\n", probe); }'
```






Based on the VTune profiling data and the kernel trace, I can map the Level Zero API calls to the kernel driver
  operations:

  API to Kernel Mapping

  1. zeMemAllocHost (3 calls, 0.003s)

  Maps to kernel operations:
  zeMemAllocHost() → ioctl(DRM_IOCTL_IVPU_BO_CREATE) → ivpu_bo_create_ioctl()
    → ivpu_gem_create_object()
    → ivpu_mmu_context_map_page() (multiple calls)
    → ivpu_bo_pin()
  The 3 host memory allocations likely correspond to:
  - Input matrix A
  - Input matrix B
  - Output matrix C

  2. zeCommandQueueCreate (1 call)

  Maps to:
  zeCommandQueueCreate() → ioctl(DRM_IOCTL_IVPU_GET_PARAM) → ivpu_get_param_ioctl()
  Creates command queue for submitting work to NPU.

  3. zeCommandListCreate (1 call)

  No direct kernel call - this is managed in user-space by libze_intel_vpu.so.

  4. zeCommandQueueExecuteCommandLists (1 call, 0.003s)

  Maps to:
  zeCommandQueueExecuteCommandLists() → ioctl(DRM_IOCTL_IVPU_SUBMIT) → ivpu_submit_ioctl()
    → Triggers IPC communication:
      → ivpu_ipc_irq_handler() (946 times)
      → ivpu_ipc_receive() (945 times)
  This submits the matrix multiplication job to the NPU.

  5. zeFenceHostSynchronize (1 call, 0.001s)

  Maps to polling/waiting operations:
  zeFenceHostSynchronize() → Polls fence status via IPC
    → ivpu_ipc_irq_handler()
    → ivpu_hw_ip_ipc_rx_count_get()

  Complete Workflow for npu_matrix_mul:

  1. Initialization:
    - Open device: open(/dev/accel/accel0) → ivpu_open()
    - Query device params: multiple ivpu_get_param_ioctl() calls
  2. Memory Allocation:
    - Allocate 3 buffers for matrices: zeMemAllocHost() × 3
    - Each allocation triggers ~1,377 ivpu_mmu_context_map_page() calls (4,131 total ÷ 3)
  3. Command Setup:
    - Create command queue: zeCommandQueueCreate()
    - Create command list: zeCommandListCreate()
    - Append matrix multiplication kernel to command list
  4. Execution:
    - Submit job: zeCommandQueueExecuteCommandLists() → ivpu_submit_ioctl()
    - NPU firmware executes the matrix multiplication
    - ~945 IPC messages exchanged during execution
  5. Synchronization:
    - Wait for completion: zeFenceHostSynchronize()
    - Continuous IPC polling until job completes

  The total execution time of ~15 seconds with most time spent in IPC communication suggests the NPU was actively
  processing the matrix multiplication workload during this period.


vtune: Warning: Cannot locate debugging information for file `/opt/intel/oneapi/vtune/2025.4/lib64/libtpsstool.so'.
vtune: Warning: Cannot locate debugging information for file `/lib/x86_64-linux-gnu/libnpu_driver_compiler.so'.
vtune: Executing actions 75 % Generating a report                              Elapsed Time: 14.814s

        Top Hotspots when GPU was idle
        Function                                                                                         Module        
        -----------------------------------------------------------------------------------------------  --------------
        operator new                                                                                     libc++abi.so  
        __cxa_demangle                                                                                   libc++abi.so  
        std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::operator=  libc++.so     
        std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::append     libc++.so     
        matrix_multiply_cpu                                                                              npu_matrix_mul
        [Others]                                                                                         N/A           

Hottest Host Tasks
Host Task                          Task Time  % of Elapsed Time(%)  Task Count
---------------------------------  ---------  --------------------  ----------
zeCommandQueueExecuteCommandLists     0.005s                  0.0%           1
zeMemAllocHost                        0.004s                  0.0%           3
zeFenceHostSynchronize                0.002s                  0.0%           1
zeCommandQueueCreate                  0.000s                  0.0%           1
zeCommandListCreate                   0.000s                  0.0%           1
Collection and Platform Info
    Application Command Line: ./npu_matrix_mul "npu_analysis_vtune.txt" 
    Operating System: 6.13.0-rc4+ DISTRIB_ID=Ubuntu DISTRIB_RELEASE=25.04 DISTRIB_CODENAME=plucky DISTRIB_DESCRIPTION="Ubuntu Plucky Puffin (development branch)"
    Computer Name: victoryang00-ASUS-Zenbook-S-14-UX5406SA-UX5406SA
    Result Size: 5.0 MB 
    Collection start time: 03:53:04 05/08/2025 UTC
    Collection stop time: 03:53:19 05/08/2025 UTC
    Collector Type: User-mode sampling and tracing
    CPU
        Name: Intel(R) microarchitecture code named Lunarlake-M
        Frequency: 3.302 GHz
        Logical CPU Count: 8

If you want to skip descriptions of detected performance issues in the report,
enter: vtune -report summary -report-knob show-issues=false -r <my_result_dir>.
Alternatively, you may view the report in the csv format: vtune -report
<report_name> -format=csv.
vtune: Executing actions 100 % done                   
