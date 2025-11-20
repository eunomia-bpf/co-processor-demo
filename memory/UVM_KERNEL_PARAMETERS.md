# NVIDIA UVM 内核参数配置指南

## 概述

NVIDIA UVM (Unified Virtual Memory) 驱动提供了丰富的内核参数（module parameters），用于动态调优性能策略。这些参数通过 Linux 的 sysfs 接口暴露，大部分可以在运行时读取，部分支持动态写入。

## 参数访问路径

所有 UVM 内核参数位于：
```
/sys/module/nvidia_uvm/parameters/
```

## 权限说明

内核参数有三种权限类型：

| 权限标志 | 八进制 | 含义 | 是否可动态修改 |
|---------|--------|------|---------------|
| `S_IRUGO` | 0444 | 只读 | ❌ 不可（需重新加载模块） |
| `S_IRUGO|S_IWUSR` | 0644 | 读写 | ✅ 可动态修改 |
| `0644` | 0644 | 读写 | ✅ 可动态修改 |

## 一、Prefetch（预取）相关参数

### 1.1 基础参数

| 参数名 | 默认值 | 权限 | 当前值 | 说明 |
|--------|--------|------|--------|------|
| `uvm_perf_prefetch_enable` | 1 | 只读 (S_IRUGO) | 1 | 全局预取开关 |
| `uvm_perf_prefetch_threshold` | 51 | 只读 (S_IRUGO) | 51 | 子区域 occupancy 阈值（%），范围 1-100 |
| `uvm_perf_prefetch_min_faults` | 1 | 只读 (S_IRUGO) | 1 | 触发预取的最小 fault 数，范围 1-20 |

**代码位置**: `uvm_perf_prefetch.c:39-61`

### 1.2 工作原理

```c
// 核心算法: compute_prefetch_region() - Line 102
// 遍历 bitmap tree 的每个节点
if (counter * 100 > subregion_pages * g_uvm_perf_prefetch_threshold)
    prefetch_region = subregion;  // 超过阈值则预取整个子区域
```

**对应论文**: IPDPS'20 "Adaptive Page Migration" 的 Tree-based Prefetcher
- 2MB 大页分解为 64KB basic blocks 的满二叉树
- 默认 51% occupancy 阈值，可配置
- 自底向上遍历，选择最大满足阈值的子区域

### 1.3 动态修改限制

⚠️ **当前所有 prefetch 参数均为只读 (S_IRUGO)**

要修改这些参数，需要：
1. 卸载模块：`sudo rmmod nvidia_uvm`
2. 重新加载并设置参数：
   ```bash
   sudo modprobe nvidia_uvm \
       uvm_perf_prefetch_enable=0 \
       uvm_perf_prefetch_threshold=75 \
       uvm_perf_prefetch_min_faults=3
   ```

## 二、Thrashing（抖动检测）相关参数

### 2.1 基础参数

| 参数名 | 默认值 | 权限 | 当前值 | 说明 |
|--------|--------|------|--------|------|
| `uvm_perf_thrashing_enable` | 1 | 只读 (S_IRUGO) | 1 | 全局 thrashing 检测开关 |
| `uvm_perf_thrashing_threshold` | 3 | 只读 (S_IRUGO) | 3 | Thrashing 检测阈值 |
| `uvm_perf_thrashing_pin_threshold` | 10 | 只读 (S_IRUGO) | 10 | Pin 页面的阈值 |
| `uvm_perf_thrashing_lapse_usec` | 200 (默认) | 只读 (S_IRUGO) | - | Thrashing 时间窗口（微秒） |
| `uvm_perf_thrashing_nap` | - | 只读 (S_IRUGO) | - | Throttle 睡眠时间 |
| `uvm_perf_thrashing_epoch` | - | 只读 (S_IRUGO) | - | Epoch 计数器阈值 |
| `uvm_perf_thrashing_pin` | - | 只读 (S_IRUGO) | - | Pin 策略开关 |
| `uvm_perf_thrashing_max_resets` | - | 只读 (S_IRUGO) | - | 最大重置次数 |

**代码位置**: `uvm_perf_thrashing.c:307-314`

### 2.2 缓解策略

1. **Throttle 策略**: CPU 睡眠（nap），减少迁移频率
2. **Pin 策略**: 将页面固定到当前位置，避免反复迁移

⚠️ **所有 thrashing 参数均为只读**

## 三、Memory Management（内存管理）相关参数

### 3.1 Oversubscription（超额订阅）

| 参数名 | 默认值 | 权限 | 当前值 | 说明 |
|--------|--------|------|--------|------|
| `uvm_global_oversubscription` | 1 | 只读 (S_IRUGO) | 1 | 全局超额订阅支持 |

**代码位置**: `uvm_pmm_gpu.c:183`

### 3.2 CPU Chunk 分配

| 参数名 | 默认值 | 权限 | 当前值 | 说明 |
|--------|--------|------|--------|------|
| `uvm_cpu_chunk_allocation_sizes` | UVM_CPU_CHUNK_SIZES | **可写** (S_IRUGO\|S_IWUSR) | - | CPU chunk 分配大小掩码 |

**代码位置**: `uvm_pmm_sysmem.c:31`

✅ **这是少数支持动态修改的参数之一！**

```bash
# 动态修改 CPU chunk 分配大小
echo 2 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_cpu_chunk_allocation_sizes
```

## 四、Page Fault 处理相关参数

| 参数名 | 默认值 | 权限 | 说明 |
|--------|--------|------|------|
| `uvm_perf_fault_batch_count` | 8 (默认) | 只读 | Fault batch 处理数量 |
| `uvm_perf_fault_coalesce` | 1 | 只读 | Fault 合并开关 |
| `uvm_perf_fault_max_batches_per_service` | - | 只读 | 每次服务的最大 batch 数 |
| `uvm_perf_fault_max_throttle_per_service` | - | 只读 | 每次服务的最大 throttle |
| `uvm_perf_fault_replay_policy` | UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT | 只读 | Fault replay 策略 |
| `uvm_perf_fault_replay_update_put_ratio` | - | 只读 | Replay 更新 PUT 指针比例 |
| `uvm_perf_reenable_prefetch_faults_lapse_msec` | - | 只读 | 重新启用 prefetch faults 的时间间隔 |

**代码位置**: `uvm_gpu_replayable_faults.c:70-116`

## 五、Debug 和测试相关参数

### 5.1 可动态修改的调试参数

| 参数名 | 默认值 | 权限 | 说明 |
|--------|--------|------|------|
| `uvm_debug_prints` | 0 | **可写** (S_IRUGO\|S_IWUSR) | 启用调试打印 |
| `uvm_release_asserts` | 0 | **可写** (S_IRUGO\|S_IWUSR) | 启用 release build 的断言 |
| `uvm_release_asserts_dump_stack` | 0 | **可写** (S_IRUGO\|S_IWUSR) | 断言失败时 dump stack |
| `uvm_release_asserts_set_global_error` | 0 | **可写** (S_IRUGO\|S_IWUSR) | 断言失败时设置全局错误 |
| `uvm_debug_enable_push_desc` | 0 | **可写** (S_IRUGO\|S_IWUSR) | Push 描述追踪 |
| `uvm_debug_enable_push_acquire_info` | 0 | **可写** (S_IRUGO\|S_IWUSR) | Push acquire 信息追踪 |

**代码位置**: `uvm_common.c:34-72`, `uvm_push.c:37-41`

✅ **这些参数支持运行时动态修改！**

```bash
# 启用调试打印
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_debug_prints

# 启用断言检查
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_release_asserts
```

### 5.2 只读测试参数

| 参数名 | 默认值 | 权限 | 说明 |
|--------|--------|------|------|
| `uvm_enable_builtin_tests` | 0 | 只读 | 启用内置测试 |
| `uvm_enable_debug_procfs` | 0 (release) / 1 (debug) | 只读 | 启用 debug procfs 接口 |

## 六、其他可动态修改的参数

| 参数名 | 默认值 | 权限 | 说明 |
|--------|--------|------|------|
| `uvm_fault_force_sysmem` | 0 | **可写** (S_IRUGO\|S_IWUSR) | 强制使用系统内存存储 faulted 页面 |
| `uvm_block_cpu_to_cpu_copy_with_ce` | 0 | **可写** (S_IRUGO\|S_IWUSR) | 使用 GPU CE 进行 CPU-to-CPU 迁移 |
| `uvm_downgrade_force_membar_sys` | 1 | **可写** (0644) | 强制 TLB 失效使用 MEMBAR_SYS |

**代码位置**:
- `uvm_va_block.c:63, 70`
- `uvm_hal.c:53`

## 七、参数访问示例

### 7.1 查看所有参数

```bash
# 列出所有参数及其权限
ls -la /sys/module/nvidia_uvm/parameters/

# 查看 prefetch 相关参数
cat /sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable
cat /sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_threshold
cat /sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_min_faults
```

### 7.2 动态修改可写参数

```bash
# 启用调试功能（运行时修改）
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_debug_prints

# 修改 CPU chunk 分配策略（运行时修改）
echo 4 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_cpu_chunk_allocation_sizes

# 强制页面使用系统内存（运行时修改）
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_fault_force_sysmem
```

### 7.3 修改只读参数（需要重新加载模块）

```bash
# 方法 1: 在 /etc/modprobe.d/ 中创建配置文件
sudo cat > /etc/modprobe.d/nvidia-uvm.conf <<EOF
options nvidia_uvm uvm_perf_prefetch_enable=0
options nvidia_uvm uvm_perf_prefetch_threshold=75
options nvidia_uvm uvm_perf_thrashing_enable=0
EOF

# 重新加载模块
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

# 方法 2: 直接在命令行指定参数
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm \
    uvm_perf_prefetch_enable=0 \
    uvm_perf_prefetch_threshold=75 \
    uvm_perf_thrashing_enable=0
```

## 八、Procfs 接口

UVM 还提供了 procfs 接口，用于查看运行时统计信息：

```bash
# 查看 procfs 目录结构
ls -la /proc/driver/nvidia-uvm/

# 主要目录
/proc/driver/nvidia-uvm/
├── gpus/          # 每个 GPU 的详细信息和统计
└── cpu/           # CPU 相关信息
```

**代码位置**: `uvm_procfs.c:30-40`

## 九、关键参数总结

### 9.1 可动态修改的参数（运行时调优）

✅ 以下参数支持通过 sysfs 动态写入：

1. **调试参数**:
   - `uvm_debug_prints` - 调试输出
   - `uvm_release_asserts*` - 断言控制
   - `uvm_debug_enable_push*` - Push 追踪

2. **内存管理**:
   - `uvm_cpu_chunk_allocation_sizes` - CPU chunk 大小
   - `uvm_fault_force_sysmem` - 强制系统内存
   - `uvm_block_cpu_to_cpu_copy_with_ce` - CPU-to-CPU 迁移策略

3. **硬件层**:
   - `uvm_downgrade_force_membar_sys` - TLB 失效策略

### 9.2 只读参数（需要重新加载模块）

❌ 以下参数只能在模块加载时设置：

1. **性能关键参数**:
   - `uvm_perf_prefetch_enable` - 预取开关
   - `uvm_perf_prefetch_threshold` - 预取阈值
   - `uvm_perf_thrashing_enable` - Thrashing 检测
   - `uvm_global_oversubscription` - 超额订阅

2. **原因分析**:
   - 这些参数在模块初始化时被拷贝到内部全局变量
   - 代码中使用的是内部变量（如 `g_uvm_perf_prefetch_threshold`）
   - 修改 sysfs 文件不会影响已运行的代码

## 十、未来改进方向

根据代码中的 TODO 注释：

1. **Line 1487-1488 (uvm_pmm_gpu.c)**:
   ```c
   // TODO: Bug 1766651: Also track read pages and update the LRU on mapping
   // Currently only updates LRU on allocation, not on actual page access
   ```
   未来可能在页面映射时也更新 LRU，提高追踪精度。

2. **Line 42 (uvm_perf_prefetch.c)**:
   ```c
   // TODO: Bug 1778037: [uvm] Use adaptive threshold for page prefetching
   ```
   未来可能实现自适应阈值，根据工作负载自动调整。

## 十一、验证方法

### 11.1 确认参数是否可写

```bash
# 查看文件权限
ls -l /sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable
# -r--r--r-- 表示只读 (0444)
# -rw-r--r-- 表示可写 (0644)

# 尝试写入测试
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable
# 如果返回 "Permission denied" 或写入后值未改变，则为只读参数
```

### 11.2 验证修改是否生效

```bash
# 修改前查看
cat /sys/module/nvidia_uvm/parameters/uvm_debug_prints

# 修改
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_debug_prints

# 修改后确认
cat /sys/module/nvidia_uvm/parameters/uvm_debug_prints

# 观察内核日志（如果启用了调试输出）
sudo dmesg | grep nvidia-uvm
```

## 十二、性能调优建议

### 12.1 禁用 Prefetch（减少带宽开销）

```bash
# 需要重新加载模块
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm uvm_perf_prefetch_enable=0
```

**适用场景**:
- 随机访问模式，预取效果差
- 带宽受限环境
- 需要精确控制数据迁移

### 12.2 调整 Prefetch 阈值

```bash
# 更激进的预取（阈值降低到 30%）
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm uvm_perf_prefetch_threshold=30

# 更保守的预取（阈值提高到 80%）
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm uvm_perf_prefetch_threshold=80
```

### 12.3 禁用 Thrashing 检测

```bash
# 如果确定不会发生 thrashing，可以禁用以减少开销
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm uvm_perf_thrashing_enable=0
```

### 12.4 运行时调试

```bash
# 启用详细日志（运行时）
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_debug_prints

# 启用断言检查（运行时）
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_release_asserts

# 查看日志
sudo dmesg -w | grep nvidia-uvm
```

## 十三、注意事项

1. ⚠️ **模块依赖**: 卸载 `nvidia_uvm` 前需要确保没有应用在使用（所有 CUDA 进程已退出）

2. ⚠️ **参数验证**: 某些参数有取值范围限制，写入非法值会被拒绝或使用默认值

3. ⚠️ **性能影响**: 修改性能相关参数可能显著影响应用性能，建议先在测试环境验证

4. ⚠️ **内核版本**: 不同版本的驱动可能有不同的参数集合和默认值

## 十四、参考资料

- **源代码**: `kernel-open/nvidia-uvm/`
- **论文**: IPDPS'20 "Adaptive Page Migration for Irregular Data-Intensive Applications under GPU Memory Oversubscription"
- **相关文档**:
  - `UVM_LRU_POLICY.md` - LRU 替换策略详解
  - `UVM_EVICTION_POLICY_FRAMEWORK.md` - 驱逐策略框架

---

**文档版本**: v1.0
**创建日期**: 2025-11-19
**验证环境**: NVIDIA open-gpu-kernel-modules (当前加载版本)
