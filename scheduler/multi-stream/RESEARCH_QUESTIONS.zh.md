你这版工具 + 当前 `compute_metrics` / raw CSV 输出，已经足够支撑一整套 **OSDI 级别**的 eval：

* 可以从 **并发度、吞吐、延迟分布、优先级语义、公平性、到达过程、内存工作集** 七八个维度发力；
* 完全可以组织出十几二十个 sub‑RQ，每个对应一个 subplot，而不需要再改 C++ 代码（复杂分析都在 Python 里做）。

下面我帮你把「基于当前输出能做的 RQ + 图」系统性整理一遍。
约定：每个 **Sub‑RQ = 1 个 subplot**，但一个 subplot 里可以有多条线 / 多组 bar。

---

## 0. 你现在到底有那些指标可以用？

从当前代码：

* aggregated（`compute_metrics`）每次 run 一行 CSV：

  * config：`streams, kernels_per_stream, total_kernels, type, grid_size, block_size`
  * 整体：

    * `wall_time_ms, throughput`
    * Service time：`svc_mean, svc_p50, svc_p95, svc_p99, svc_stddev`
    * E2E latency：`e2e_mean, e2e_p50, e2e_p95, e2e_p99`
    * queue：`avg_queue_wait, max_queue_wait`
    * 并发与利用率：`concurrent_rate, util, max_concurrent, avg_concurrent`
    * 公平性：`jains_index`（按 per‑stream 总执行时长）
    * priority：`inversions, inversion_rate, per_priority_avg/p50/p99`
    * memory：`working_set_mb, fits_in_l2`

* raw per‑kernel CSV（`--csv-output`）：

  * `stream_id, kernel_id, priority, kernel_type`
  * `enqueue_time_ms, start_time_ms, end_time_ms`
  * `duration_ms, launch_latency_ms, e2e_latency_ms`
  * `host_launch_us, host_sync_us`

> 也就是说：**concurrency / throughput / latency percentiles / fairness / priority inversion / L2 边界** 你都已经有了。
> 后处理（Python）可以随便算 CDF、per‑stream/per‑priority/per‑type 任何东西。

下面所有 RQ 都严格只用这些东西。

---

## RQ1：GPU 流并发能力与可扩展性（Stream Scalability & Concurrency）

### Sub‑RQ1.1：`max_concurrent` vs #streams（不同 kernel 大小）

**问题**：在不同 kernel 时间长度下，实际能同时跑的 kernel 数（`max_concurrent`）如何随 stream 数变化？
**配置**：

* 固定 kernel type（比如 `COMPUTE` 或 `GEMM`）
* sweeps：

  * `streams ∈ {1,2,4,8,16,32,64}`
  * `workload_size`/`kernel_iterations` 调成几档典型 **单 kernel 时长**（例如 50µs, 200µs, 1ms, 5ms）
* 其他参数 default（无优先级、无 heterogeneity、无 load imbalance）

**图**：

* Plot type：折线图
* X 轴：`streams`（建议 log2）
* Y 轴：`max_concurrent`（来自 aggregated CSV）
* 多条线：不同 kernel size（或 service time 档位）

这个 subplot 直接回答：

* 真正的并发度上限是多少（经常 < stream 数）；
* 小 kernel vs 大 kernel 的并发差异。

---

### Sub‑RQ1.2：`concurrent_rate` vs #streams（多大比例时间在“真正并发”？）

**问题**：GPU 在有工作的时候，有多大比例时间是 **≥2** 个 kernel 同时跑？
**配置**：同 1.1（可以重用数据）。

**图**：

* 折线图
* X：`streams`
* Y：`concurrent_rate`（aggregated CSV，百分比）
* 多条线：不同 kernel size

这张图支撑 “并发质量” 的结论：

* streams 过少 → `concurrent_rate` 低；
* streams 增加到一定程度以后，`concurrent_rate` 饱和甚至下降（调度/launch 开销）。

---

### Sub‑RQ1.3：`util` vs #streams（GPU 利用率）

**问题**：在不同 stream 数下，GPU 忙碌时间占整个时间的比例？
**配置**：同 1.1。

**图**：

* 折线图
* X：`streams`
* Y：`util`（GPU utilization, aggregated）
* 多条线：不同 kernel size

OSDI 水平的点：

* 结合 1.2 一起讨论：

  * 如果 `util` 高但 `concurrent_rate` 低，说明靠单 kernel 吃满；
  * 如果 `util` 高且 `concurrent_rate` 也高，说明流并发确实在发挥作用。

---

### Sub‑RQ1.4：`avg_concurrent` vs #streams

**问题**：长期平均同时活跃 kernel 数随 stream 数怎么变？
**配置**：同 1.1。

**图**：

* 折线图
* X：`streams`
* Y：`avg_concurrent`
* 多条线：不同 kernel size

这张图是对 1.1/1.2 的 summary：

* 多数 GPU 上你会看到 **平均并发度远低于 stream 数**；
* 可以直接给出结论：“仅仅堆 streams 并不能线性提升并行度”。

---

## RQ2：吞吐随 stream / workload type / working set 的变化

### Sub‑RQ2.1：`throughput` vs #streams（不同 kernel size）

**问题**：多开 stream 对整体吞吐有多大帮助？什么时候饱和甚至回退？
**配置**：同 RQ1.1 的 sweep。

**图**：

* 折线图
* X：`streams`
* Y：`throughput (kernels/sec)`
* 多条线：不同 kernel size

你可以把这张图和 1.3（util）放一起解读：

* `throughput` 上升 + `util` 上升 → streams 的确提高了资源利用；
* `util` 高但 `throughput` 不再上升 → stream 继续加只是增加调度/launch overhead。

---

### Sub‑RQ2.2：`throughput` vs #streams（不同 `kernel_type`）

**问题**：compute / memory / mixed / gemm 这几类 kernel 的吞吐 scaling 行为有何差异？
**配置**：

* sweeps：

  * `streams ∈ {1..64}`
  * `kernel_type ∈ {COMPUTE, MEMORY, MIXED, GEMM}`
* `workload_size` 调到让四种 type 的 **单 kernel 时长大致落在同一数量级**，避免完全 incomparable。

**图**：

* 折线图
* X：`streams`
* Y：`throughput`
* 多条线：不同 kernel_type

OSDI 级别解读可以是：

* memory 型 kernel 在高 stream 数时 throughput 提升有限（带宽瓶颈）；
* compute/mixed/gemm 在某个 stream 区间内 scaling 更好；
* 结合 1.x 的并发指标，看不同类型 under the hood 的调度行为。

---

### Sub‑RQ2.3：`throughput` vs offered load（通过 `launch_frequency`）

**问题**：在增加请求到达速率时，GPU 的吞吐如何从 underload → saturation？
**配置**：

* 固定 `streams`（例如 8 或 16）
* 对每个 stream 设置相同 `launch_frequency`，全局 offered load ∝ frequency
* sweeps： `freq ∈ {低 → 高 → 明显超载}`，有 jitter 与无 jitter 两组

**图**：

* 折线图
* X：offered load（可以直接用 `total_kernels / total_wall_time` 或 sum over 1/freq 近似）
* Y：`throughput`（实际完成的 kernels/sec）
* 两条线：无 jitter vs 有 jitter

用来说明：

* 在轻载区，throughput 随 offered load 增长；
* 在 saturate 区，throughput flatten；
* jitter 对 saturation 区的影响（乱序到达是否导致更差/更好的 lumping）。

---

## RQ3：延迟分布与排队行为（Latency & Queueing）

### Sub‑RQ3.1：E2E 延迟 CDF（低并发 vs 高并发）

**问题**：增加 streams / 提高 arrival rate 对 tail latency 的影响有多大？
**配置**：

* 选一个固定 kernel type & size（比如 MIXED, ~0.5–1ms）
* 两个对比配置：

  * 低并发：`streams=2`, 中等 arrival rate
  * 高并发：`streams=32`, 高 arrival rate（但未明显超载）
* 多 run 取中位数或 overlay 多条 CDF（但 subplot 还是一张）

**图**（需要 raw CSV）：

* Plot type：CDF（线图）
* X：`e2e_latency_ms`
* Y：`CDF`
* 两条线：低并发 vs 高并发

这张图是 OSDI 最爱的一种：

* 可直接读出中位数变化、P95/P99 变化；
* 可以引用为“高并发场景下 tail latency 变成原来的 X 倍”。

---

### Sub‑RQ3.2：E2E P99 vs #streams（不同 kernel_type）

**问题**：不同 kernel 类型对 tail latency 的敏感性如何？
**配置**：

* sweeps：`streams ∈ {1..64}`
* 多次 run，取每个配置的 P99（aggregated CSV 的 `e2e_p99`）
* 对每种 kernel_type 单独跑一轮

**图**：

* 折线图
* X：`streams`
* Y：`e2e_p99`
* 多条线：`kernel_type ∈ {COMPUTE,MEMORY,MIXED,GEMM}`

这张图可以和 RQ2.2 做对照：

* throughput 上去了，tail latency 是否可接受？
* memory 型 workload tail 增长是否更剧烈？

---

### Sub‑RQ3.3：平均 / 最大 queue wait vs #streams

**问题**：排队等候时间（`launch_latency`）如何随 stream 数增长？
**配置**：同 3.2 或 1.1。

**图**：

* 折线图（或 error bar）
* X：`streams`
* Y：`avg_queue_wait` 或 `max_queue_wait`
* 多条线：不同 kernel size 或类型

用来量化“系统内在的 queuing effect”，可以直接说：

* 在高 stream 数下，平均队列等待时间达到 X ms，占 e2e latency 的 Y%。

---

### Sub‑RQ3.4：Queue wait CDF 在轻载 vs 重载

**问题**：轻载（利用率低）和 heavy load 时，queue wait 的分布有何不同？
**配置**：

* 固定 `streams` 和 workload
* 用 `launch_frequency` 控制 light load / heavy load / overload 三档
* 多 run

**图**（raw CSV）：

* CDF
* X：`launch_latency_ms`
* Y：CDF
* 三条线：light / medium / heavy

这张图能很直观地解释“为什么在 trace 驱动 workload 下 tail 会炸”：排队分布直接给你看。

---

## RQ4：CUDA stream priority 的实际语义（Priority Behavior）

### Sub‑RQ4.1：Priority inversion rate vs #streams（有/无 priority）

**问题**：CUDA priority 对 **“谁先上 GPU”** 的排序能力有多大？
**配置**：

* 两组：

  * baseline：所有 stream priority 一样（比如 0）
  * priority：一半 stream 设为高（数值更接近 greatest_priority），一半为低
* sweeps：`streams ∈ {4,8,16,32}`
* 其他相同（同一种 kernel，同样 arrival pattern）

**图**：

* 柱状图/折线图均可
* X：`streams`
* Y：`inversion_rate`（aggregated CSV），0–1
* 两组 series：无 priority vs 有 priority

这张图就是你之前诊断 “priority inversion” 的量化版本：

* 如果有 priority 时 inversion_rate 仍然很高，说明 driver 对 priority 支持很弱；
* 如果 inversion_rate 明显下降，说明至少在 queue ordering 层面是有效的。

---

### Sub‑RQ4.2：按 priority class 的 P99 延迟 vs offered load

**问题**：在负载上升时，高优先级流的 tail latency 相对低优先级能被保护到什么程度？
**配置**：

* 两类 stream：High priority（短 kernel） vs Low priority（相对长 kernel）
* sweep：通过 `launch_frequency` 调整 offered load，直到明显接近 saturation
* 多次 run，使用 aggregated CSV 里的 `per_priority_p99` 字段（按 priority 升序）

**图**：

* 折线图
* X：**launch frequency (Hz)**（直接用 `launch_freq` 列，log scale）
* Y：P99 e2e（高优先级一条线，低优先级一条线）

> **实现注意**：当前 analyzer 已改为直接用 `launch_freq` 列并按频率排序，避免使用派生的 offered_load。

从图可以读出：

* 高优先级 P99 是否随着负载增长而增长得慢得多；
* 某些 GPU 上 priority 完全不 work，高低优先级 P99 几乎重合，这就是可以写进 paper 的负面结论。

---

### Sub‑RQ4.3：Fast kernel P99 在 RT vs BE 混合场景

**问题**：短小"前景任务"在长 kernel 背景下，priority 能否改善短任务 tail latency？

**配置**：

* 两类 kernel：
  * Fast：size=65536（compute type, ~几十 µs）
  * Slow：size=4194304（memory type, ~几 ms）
* 三种配置：
  1. Only Fast: 4 streams, 40 kernels each, 所有 fast kernels
  2. Fast+Slow, no priority: 4 streams（前 2 个 fast, 后 2 个 slow）, load-imbalance=40,40,10,10
  3. Fast+Slow, with priority: 同配置 2，但 fast streams priority=-5, slow streams priority=0

**测量方法**：

* 使用 raw per-kernel CSV（`--csv-output`）
* **关键**：只统计 fast kernels（stream_id in {0, 1}）的 P99
* 指标：`e2e_latency_ms` 的第 99 百分位（仅 fast kernels）
* 聚合：按配置分组，计算 mean/std across runs

**图**：

* 柱状图
* X：三个配置
* Y：Fast kernels 的 `e2e_p99`（ms）
* Error bar：多 run 标准差

---

### Sub‑RQ4.4：Jain's Fairness vs priority 配置

**问题**：priority 对不同 stream 之间 GPU 时间占用的公平性有什么影响？

**配置**：

* 8 streams, 30 kernels each, size=1M
* 多种 priority pattern：
  * All Equal: 0,0,0,0,0,0,0,0
  * 1H-7L: -5,0,0,0,0,0,0,0
  * 2H-6L: -5,-5,0,0,0,0,0,0
  * 4H-4L: -5,-5,-5,-5,0,0,0,0
  * Multi-Level: -10,-8,-5,-3,0,0,0,0

**测量方法**：

* 使用 aggregated CSV 的 `jains_index`（基于 per-stream total duration）
* 每种 pattern 多 run，计算 mean/std
* Pattern 名称直接存储到结果 CSV（避免 position-based 推断）

**图**：

* 柱状图
* X：priority pattern
* Y：`jains_index`（0-1, 越接近 1 越公平）
* Error bar：std across runs

---

## RQ5：Preemption Latency（抢占延迟分析）

**问题**：测量高优先级 kernel 到达时 GPU 正忙于低优先级 kernel 的情况下，等待调度的延迟

**设计核心**：

* 反转 workload：少量高优先级 kernel，大量低优先级 kernel saturate GPU
* 只统计"被阻塞的 kernel"：高优先级 kernel 到达时 GPU 正在执行低优先级 work
* 配置：2 high-prio streams（稀疏）+ 6 low-prio streams（密集）

**过滤方法**：

```python
# 1. 构建低优先级 kernel 执行时间区间
lo_intervals = [(start_ms, end_ms) for each low-prio kernel]

# 2. 判断高优先级 kernel 到达时 GPU 是否被占用
def is_gpu_busy_with_low_prio(enqueue_time):
    return any(start <= enqueue_time < end for start, end in lo_intervals)

# 3. 只对被阻塞的高优先级 kernel 计算 preemption latency
hi_blocked = hi[hi['enqueue_time_ms'].apply(is_gpu_busy_with_low_prio)]
preempt_latency = hi_blocked['start_time_ms'] - hi_blocked['enqueue_time_ms']
```

**关键指标**：

* `preempt_latency_ms`：start_time - enqueue_time（仅 blocked kernels）
* `blocked_ratio`：被阻塞的高优先级 kernel 占比
* `num_blocked_kernels`：实际被阻塞的样本数

---

### Sub‑RQ5.1：Preemption latency vs background kernel duration

**问题**：背景低优先级 kernel 的执行时长如何影响抢占延迟？

**配置**：

* 2 high-prio streams (priority=-5, 20 kernels each)
* 6 low-prio streams (priority=0, 100 kernels each)
* Sweep background kernel size: [65536, 262144, 1048576, 4194304]

**测量方法**：

* 使用 raw CSV，过滤 blocked high-prio kernels
* 指标：`preempt_latency_ms` 的 mean, P99, max
* 附加指标：`blocked_ratio`, `num_blocked_kernels`

**图**：

* 折线图（log-log scale）
* X：Background kernel size (elements)
* Y：Preemption latency (ms)
* 三条线：Mean, P99, Max

---

### Sub‑RQ5.2：Preemption latency vs offered load

**问题**：高优先级 kernel 到达频率对抢占延迟的影响

**配置**：

* 2 high-prio streams (priority=-5, 50 kernels each)
* 6 low-prio streams (priority=0, 50 kernels each, fixed 500Hz)
* High-prio frequency sweep: [20, 50, 100, 200, 500, 1000] Hz
* Background kernel size: 1M

**测量方法**：

* 使用 raw CSV，过滤 blocked high-prio kernels
* 指标：`preempt_latency_ms` 的 mean, P50, P95, P99, max
* 附加指标：`blocked_ratio`, `num_blocked_kernels`

**图**：

* 折线图
* X：High-priority launch frequency (Hz, log scale)
* Y：Preemption latency (ms)
* 三条线：P99, Mean, P95

---

### Sub‑RQ5.3 & 5.4：Preemption latency CDF

**问题**：用 CDF 展示 preemption latency 的完整分布

**Sub-RQ5.3**: Small vs large background kernels
* 数据：从 RQ5.1 的 raw CSV 中提取
* 对比：65k (small) vs 4M (large) 的所有 blocked kernels
* 图：CDF，X 轴 preemption latency (ms), Y 轴 CDF

**Sub-RQ5.4**: Low vs high offered load
* 数据：从 RQ5.2 的 raw CSV 中提取
* 对比：20Hz (low load) vs 1000Hz (high load) 的所有 blocked kernels
* 图：CDF，X 轴 preemption latency (ms), Y 轴 CDF

---

## RQ6：异构 / 负载不均衡与公平性（Heterogeneity & Imbalance）

这里主要利用：

* `load-imbalance`（不同 stream kernel 数）
* `kernel_types_per_stream`（heterogeneous）
* aggregated 的 Jain 指数 + raw per‑stream metrics。

### Sub‑RQ6.1：Jain's index vs 负载不均衡程度

**问题**：给不同 stream 分配不同任务量时，GPU 时间分配的公平性随不均衡程度如何变化？
**配置**：

* 用 `--load-imbalance` 指定几种 pattern，比如：

  * 均匀：[20,20,20,20]
  * 轻度：[10,20,30,40]
  * 重度：[5,10,40,80]
* 每种 pattern 下跑多次，记录 `jains_index`（aggregated）

**图**：

* 折线或柱状图
* X：不均衡程度（比如用 CV of kernels_per_stream 表示）
* Y：`jains_index`

这张图用来讲：“即使 driver 尝试公平调度，负载不均衡也天然导致时间占用偏斜”。

---

### Sub‑RQ6.2：Per‑stream P99 延迟（load imbalance 场景）

**问题**：在 load imbalance 场景下，不同 stream 的 tail latency 差异有多夸张？
**配置**：

* 选一个典型不均衡配置（比如 [5,10,40,80]）
* 用 raw CSV 离线按 `stream_id` 聚合，算每个 stream 的 P99 e2e latency。

**图**：

* 柱状图
* X：`stream_id`
* Y：该 stream 的 P99 e2e latency

这张图非常直观，可以用在 paper 里当“heat map/bar chart”展示 per‑stream SLO violation。

---

### Sub‑RQ6.3：并发 & throughput 在 heterogeneity 下的变化

**问题**：不同 stream 跑不同 kernel type 时，整体 concurrency 与 throughput 如何变化？
**配置**：

* `--heterogeneous`：例如 `[memory, memory, compute, compute, gemm, mixed,...]`
* baseline：所有 stream 同一个 type（比如 compute）
* 指标：`max_concurrent`, `concurrent_rate`, `throughput`（aggregated）

**图**：

* 折线图
* X：`streams`
* Y：`throughput` 或 `concurrent_rate`
* 两条线：homogeneous vs heterogeneous

用来说明：“类型异构 + 不均衡访问 pattern”下 scheduler 表现与简单 homogeneous case 差距多大。这是典型 OS 论文会写的一点。

---

## RQ7：到达过程与 jitter 对系统行为的影响（Arrival Pattern & Jitter）

你已经用 `launch_frequency_per_stream` + `random_seed` + jitter 实现了 pseudo‑Poisson 到达。

### Sub‑RQ7.1：`concurrent_rate` vs jitter（固定平均频率）

**问题**：在相同平均 arrival rate 下，引入 jitter 会让实际并发时间比例发生什么变化？
**配置**：

* 固定 `streams`, `launch_frequency`
* 通过 `random_seed` 控制：

  * 0 → 无 jitter（pure periodic）
  * 多个非 0 → 有 jitter（随机）
* 多 run，取每种模式的 `concurrent_rate` 平均值

**图**：

* 柱状 / 折线图
* X：arrival pattern 类型（periodic / jittered）
* Y：`concurrent_rate`

结论可以是：jitter 反而增加了/减少了 overlap，或者对 queueing 产生更大的 burst，影响 tail latency。

---

### Sub‑RQ7.2：E2E P99 vs jitter（固定平均频率）

**问题**：在相同平均 load 下，arrival jitter 对 tail latency 的影响是多少？
**配置**：同 7.1。

**图**：

* 柱状图
* X：pattern（periodic / jittered 异常模式）
* Y：`e2e_p99`

这两张一起很适合出现在“robustness / sensitivity”小节。

---

## RQ8：工作集 vs L2 cache 边界（Memory Working Set & Caching）

你已经在 metrics 里算了：

* `working_set_mb`（粗略）
* `l2_cache_mb`
* `fits_in_l2`（bool）

虽然估算比较粗，但完全可以支撑一个"系数"层面的分析。

### Sub‑RQ8.1：`throughput` / `util` vs working_set/L2 ratio

**问题**：工作集大小是否跨过 L2 容量会显著改变 throughput / 利用率 / 并发度？
**配置**：

* sweeps：`workload_size`，让 `working_set_mb` 覆盖 `< L2`、`≈L2`、`>L2` 几档
  * **RTX 5090 实现**：L2=96MB，workload_size ∈ {1M, 2M, 3M, 4M, 6M, 8M} → WS ∈ {32, 64, 96, 128, 192, 256} MB
* 记录 aggregated 的 `throughput`, `util`, `concurrent_rate`
* 可分 kernel_type（memory-heavy vs compute-heavy）

**图**：

* 折线图
* X：`working_set_mb / l2_cache_mb`（>1 表示工作集超出 L2）
* Y：`throughput` 或 `util`
* 多条线：不同 kernel_type

OSDI 级 story：

* “当工作集远大于 L2 时，memory 型 workload concurrency 下降，throughput 限于 DRAM 带宽；”
* 这一类分析在调度 paper 里是加分项（说明你不是只看黑盒时间线）。

---

## RQ9：多进程 / 多租户（Multi-Process vs Single-Process）

**问题**：在相同总 stream 数和总工作量下，将并发 stream 分配到不同数量的进程时，CUDA scheduler 在 throughput、tenant fairness、tail latency 上的行为差异？

**设计核心**：

* 固定总 stream 数为 32，总工作量为 960 kernels（每 stream 30 个）
* 变化进程粒度：1×32, 4×8, 8×4, 16×2
* 每个进程视为一个 **tenant**，测量 per-tenant 指标
* 使用真实并发执行（`subprocess.Popen`）而非串行

---

### Sub‑RQ9.1：Throughput、Fairness、Latency vs Process Configuration

**问题**：在不同进程粒度下，整体吞吐、tenant 公平性、worst-case tail latency 如何变化？

**配置**：

| 配置 | 进程数 | 每进程 streams | 总 streams | 每 stream kernels | Kernel size | Type |
|------|--------|----------------|------------|-------------------|-------------|------|
| 1×32 | 1      | 32             | 32         | 30                | 1M          | mixed |
| 4×8  | 4      | 8              | 32         | 30                | 1M          | mixed |
| 8×4  | 8      | 4              | 32         | 30                | 1M          | mixed |
| 16×2 | 16     | 2              | 32         | 30                | 1M          | mixed |

**执行方式**：

* **单进程**（1×32）：使用 C++ aggregated CSV 输出
* **多进程**（4×8, 8×4, 16×2）：
  * 并发启动所有进程（`subprocess.Popen`）
  * 每个进程写独立的 raw CSV（`--csv-output`, `--no-header`）
  * 等待所有进程完成
  * 如果任何进程失败，丢弃整个 run

**测量方法（多进程）**：

1. **Global Stream ID 分配**：
   * 每个进程的 stream_id 加上 `proc_id * streams_per_process` 偏移
   * 保证每个进程的 stream 在全局唯一

2. **Throughput**（全局）：
   * 全局时间窗口：从所有 kernel 的 `min(start_time_ms)` 到 `max(end_time_ms)`
   * `throughput = total_kernels / wall_time_sec`

3. **Global P99 Latency**（全局）：
   * 所有 kernel 的 `e2e_latency_ms` 的第 99 百分位

4. **Per-Process (Tenant) Fairness**：
   * 计算每个进程消耗的总 GPU 时间（sum of `duration_ms`）
   * 基于 GPU 时间计算 Jain's fairness index
   * `jains_index = (sum x)^2 / (n * sum x^2)`

5. **Worst-Tenant P99 Latency**：
   * 计算每个进程（tenant）内所有 kernel 的 P99
   * 取所有 tenant P99 的最大值（worst-case）
   * 展示最受影响 tenant 的 SLO violation

6. **GPU Utilization**：
   * 使用 sweep-line 算法计算 GPU 有 kernel 运行的时间占比
   * 基于所有 kernel 的 start/end 事件

**测量方法（单进程）**：

* 使用 C++ 的 aggregated CSV 输出
* `per_proc_p99` = `e2e_p99`（因为只有 1 个进程）
* `jains_index` 为 per-stream fairness（C++ 计算）
  * 对于 1 个 tenant 的场景，tenant fairness 退化为 trivial 1.0
  * 这里的 Jain 值主要作为 sanity check

**图**（2×2 subplots）：

* **(a) Throughput vs Configuration**：
  * X 轴：配置（1×32, 4×8, 8×4, 16×2）
  * Y 轴：Throughput (kernels/sec)
  * 柱状图 + error bar (std across runs)

* **(b) Per-Process Fairness vs Configuration**：
  * X 轴：配置
  * Y 轴：Jain's Fairness Index (per-process)
  * 柱状图 + error bar
  * Y 范围：[0, 1.1]

* **(c) Global P99 Latency vs Configuration**：
  * X 轴：配置
  * Y 轴：Global E2E P99 Latency (ms)
  * 柱状图 + error bar

* **(d) Worst-Tenant P99 vs Configuration**：
  * X 轴：配置
  * Y 轴：Worst-Tenant P99 Latency (ms)
  * 柱状图 + error bar
  * **关键**：展示 SLO violation for the most affected tenant

**指标语义**：

* **Throughput**：整体系统吞吐（kernels/sec）
* **Per-Process Fairness**：各 tenant 获得的 GPU 时间是否均衡（Jain index）
* **Global P99**：从所有 kernel 视角看的 tail latency
* **Worst-Tenant P99**：从最差 tenant 视角看的 tail latency（SLO violation）

**CSV 输出**：

* 文件：`results/rq9_multiprocess.csv`
* 额外字段：`per_proc_p99`（追加到最后一列）
* 单进程行：使用 C++ aggregated 格式 + `per_proc_p99`
* 多进程行：Python 计算的 metrics + `per_proc_p99`

---

## 总结一下结构（方便你写 evaluation 大纲）

你可以把整个 eval 切成几节，每节是一个 top-level RQ，每节下面 3–4 个 sub‑RQ，每个 sub‑RQ 1 张图：

* **RQ1：Stream scalability & concurrency**

  * 1.1 `max_concurrent` vs streams
  * 1.2 `concurrent_rate` vs streams
  * 1.3 `util` vs streams
  * 1.4 `avg_concurrent` vs streams

* **RQ2：Throughput & workload type**

  * 2.1 throughput vs streams (size sweep)
  * 2.2 throughput vs streams (type sweep)
  * 2.3 throughput vs offered load

* **RQ3：Latency & queueing**

  * 3.1 e2e CDF (low vs high concurrency)
  * 3.2 e2e P99 vs streams
  * 3.3 avg/max queue wait vs streams
  * 3.4 queue wait CDF (light vs heavy load)

* **RQ4：Priority semantics**

  * 4.1 inversion_rate vs streams (w/ & w/o priority)
  * 4.2 per-priority P99 vs load
  * 4.3 fast kernels P99 (single / no‑prio / prio)
  * 4.4 Jain fairness vs priority pattern

* **RQ5：Preemption latency**

  * 5.1 preemption latency vs background kernel duration
  * 5.2 preemption latency vs offered load
  * 5.3 preemption latency CDF (small vs large background kernels)
  * 5.4 preemption latency CDF (low vs high offered load)

* **RQ6：Heterogeneity & imbalance**

  * 6.1 Jain index vs load imbalance
  * 6.2 per-stream P99 bar chart
  * 6.3 throughput/concurrency in homogeneous vs heterogeneous

* **RQ7：Arrival pattern & jitter**

  * 7.1 concurrent_rate vs jitter
  * 7.2 e2e P99 vs jitter

* **RQ8：Working set vs L2**

  * 8.1 throughput/util vs working_set/L2 ratio

* **RQ9：Multi-process vs single-process**

  * 9.1 fairness/throughput vs process count（offline merge）

每个 sub‑RQ 都有明确：

* 问什么；
* 用哪些已有字段；
* vary 哪些 config 参数；
* 图是什么形式。
