对，你说得对：**只拿 seq_read / random_read 这种玩具 kernel 上 OSDI 肯定不买账**。这些顶多是“补充 microbench”，主角还是要贴近 SC’21、UVMBench、MGG、InfiniGen 那种“real kernel + clear memory pattern”。

我重写一套设计，把 microbench 分成三层：

* Tier‑0：极简 synthetic kernel（保留，但只是用来解释机制）
* Tier‑1：典型 “real kernel”（GEMM、Stencil、SpMV、BFS、Conv 等）
* Tier‑2：领域特定 kernel（Transformer layer、GNN layer、一层 CNN）

下面全部按“你准备写 OSDI measurement/system paper”的标准来设计。

---

## 一、整体设计思路

### 1.1 目标和 RQ（重新整理）

你这套 benchmark 最终要支撑的是类似这样的一组问题：

* RQ1：**在真实 kernel 上，UVM 相对显式 GPU 内存管理的性能损失有多大？**（HPC / DL / GNN / LLM 各一类）
* RQ2：**UVM 在不同访存模式和计算强度的 kernel 上表现是否一致？** dense vs stencil vs sparse vs graph vs attention。
* RQ3：**在 realistic oversubscription（1.0×–2.0× 显存）下，哪些 kernel 还能“勉强可用”，哪些直接被 thrash 掉？**
* RQ4：**简单 prefetch（如 cudaMemPrefetchAsync）在 real kernel 上的收益/副作用有多大？** 对哪些访问 pattern 有用，对哪些完全救不了。

Tier‑0 synthetic 只用来给这几类 real kernel 找“解释工具”；**所有结论必须在 Tier‑1/2 上复现**，否则 OSDI reviewer 会直接说“你这个只在合成负载上 work”。

---

## 二、Workload 设计：从玩具到 real kernel 的三层结构

### 2.1 Tier‑0：保留少量 synthetic kernel（只做“显微镜”）

这一层你可以保留 2–3 个最基础的访问模式就够了，用来解释 UVM 行为本身：

* `seq_stream`：顺序读一大段 managed 数组（带计算），看 cold fault + steady‑state 吞吐。
* `rand_stream`：完全随机读写 managed 数组，看最坏场景下每页只用几个字节的情况。
* `pointer_chase`：典型 TLB + pointer-chasing 场景。

这层不再是 main evaluation，而是在 Section “UVM Behavior Characterization” 里给 real kernel 提供解读 basis——比如说明为什么 SpMV 这么烂，而 GEMM 相对没那么惨。

---

### 2.2 Tier‑1：real kernel family（和 SC’21 / UVMBench 一样的级别）

这一层是重点，目标是用一批**覆盖不同 *memory behavior* 的真实 kernel**，但每个 kernel 本身足够小、单一：

#### 2.2.1 Dense Linear Algebra：GEMM（高重用、高算密度）

* 对应 SC’21 的 cuBLAS sgemm  和 UVMBench 里的 GEMM/SGEMM
* 工作负载：

  * 单精度 GEMM：`C = A×B`，A,B,C 都是 `cudaMallocManaged`；
  * 用 cuBLAS 或 CUTLASS 实现（不用自己写 naive GEMM，否则 reviewer 直接问“为什么不用 cuBLAS？”）。
* 参数控制：

  * 矩阵大小 N×N，从 fit‑in 到 oversub（确保 A+B+C 合起来是 {0.25×,0.5×,1×,1.25×,1.5×,2×} GPU mem）；
  * 访问模式固定是 dense + cache friendly。
* 意义：

  * 这是 **“best‑case real kernel”**：算密度高、数据重用多，看 UVM 在极友好场景下的 overhead 下限。

#### 2.2.2 Stencil / 2D/3D Convolution：高空间局部性、有限重用

* 对应 UVMBench 里的 2DCONV/3DCONV、SC’21 的 Gauss‑Seidel / HPGMG‑FV。

* 工作负载：

  1. **2D 5-point stencil**（或 9‑point）：

     * 大数组 `A[N][N]` 和 `B[N][N]` 是 managed；
     * kernel：对每个 internal cell 做固定模板更新；
     * 时间步 t 可以做 1–10 步（增加 compute reuse）。
  2. **3D 7‑point stencil**（可选，增加内存压力）。

* 参数控制：

  * N 调到数组大小 {0.25×–2×} GPU mem；
  * 时间步数 t ∈ {1, 5}，看算密度变化对 UVM 影响。

* 意义：

  * 代表 HPC / PDE 类型应用：有空间局部性，但重用窗口远不及 GEMM；
  * 对 UVM 来说是中等难度 case。

#### 2.2.3 Sparse Linear Algebra：SpMV / SpMM（典型 irregular）

* 对应 UVMBench / SC 系列大量用的 SpMV、以及许多 graph/GNN 底层算子。
* 工作负载：

  * 用 cuSPARSE 的 CSR SpMV：`y = A x`，A 是 sparse matrix；
  * 稀疏矩阵来源：

    * SuiteSparse 的标准测试矩阵（如 webbase‑1M、kron_g500 等）；
    * 或 OGBN 的 graph adjacency 转成 CSR（和 MGG 中图数据接轨）。
  * A, x, y 全部 managed。
* 参数控制：

  * 控制 matrix size / nnz，使得 `sizeof(A) + sizeof(x) + sizeof(y)` 在 {0.25×–2×} GPU mem；
  * 矩阵类型区分：

    * “结构规整”（banded、block‑diag）；
    * “结构随机”。
* 意义：

  * 这是典型 **“TLB / page‑fault hell”**；
  * 跟 MGG 的 irregular graph pattern完全一脉相承，用来解释为什么 GNN + UVM 会被打爆。

#### 2.2.4 Graph Traversal：BFS / PageRank（一阶 graph kernel）

* 对应 UVMBench 的 BFS、SC/Graph500 测试中最常见的 kernel。
* 工作负载：

  * 选一个简化 BFS / PageRank kernel（可以直接基于 Gunrock 的实现，或者用 UVMBench 改的版本）；
  * 图数据用：

    * scale‑free graph（Twitter, com‑orkut 等）；
    * mesh‑like graph（road network）；
  * 顶点属性 / frontier 等结构全部用 UVM。
* 参数控制：

  * 图规模调到顶点数 / 边数使得 graph + aux arrays 占 {0.25×–2×} GPU mem；
  * 多次 BFS/PR 迭代，稳定阶段采样。
* 意义：

  * 这就是 MGG 前面的“单 GPU graph kernel”版本；
  * 显示 UVM 在 neighbor‑exploration 上表现如何，对比 SpMV。

#### 2.2.5 CNN 层 / Conv+BN+ReLU（DL 真实 kernel）

* 对应 PipeSwitch/TGS 那种 DL 模型的基本 building block。
* 工作负载：

  * 模仿 ResNet‑50 的某个 conv block（如 3×3 conv + BN + ReLU）：

    * 输入 feature map: `N×C_in×H×W`；
    * filter: `C_out×C_in×K×K`；
    * 输出: `N×C_out×H'×W'`；
  * 用 cuDNN 或 PyTorch C++ frontend 跑 forward，只是把所有 tensor 改为 `cudaMallocManaged` 分配。
* 参数控制：

  * 调 batch size N、feature map 分辨率，使整个 layer 的 working set 覆盖 {0.25×–2×} GPU mem；
  * 也可以固定 N、H、W，增大 C_in / C_out。
* 意义：

  * 对应真实 DL 推理 kernel，空间局部性较好，但中间 activation 大、权重 reuse 高；
  * 跟 PipeSwitch / TGS 里的场景有直接对齐。

---

### 2.3 Tier‑2：领域特定 kernel（LLM / GNN）

这一层用**简化但真实的“层”**，直接对上 InfiniGen / MGG 里的 evaluation 场景：

#### 2.3.1 Transformer decoder block（LLM‑style KV heavy kernel）

* 对应 InfiniGen 的单层 KV cache 使用模式。
* 工作负载：

  * 实现一个简化的 decoder block：

    * multi‑head attention（Q,K,V projection + softmax + matmul）；
    * MLP（两层全连接）；
  * KV cache 以 `[layers][heads][seq_len][head_dim]` 存放，全部用 `cudaMallocManaged`。
* 参数控制：

  * 固定 hidden size/head 数，扫 seq_len，让 KV cache 占 {0.5×–4×} GPU mem；
  * 可选：只把 KV cache 放 UVM，weights 用普通 device mem，复现实际场景。
* 意义：

  * 直接对应 InfiniGen 的 baseline：UVM 做 KV offloading。
  * 把 micro‑bench 跟完整 LLM 系统实验连起来。

#### 2.3.2 GNN layer（GCN message passing）

* 对应 MGG 的单层 GCN 聚合逻辑。
* 工作负载：

  * 给定 CSR graph + node feature matrix `H`，实现：

    * `H' = σ(Ā H W)`（典型的 GCN 一层）；
  * graph 结构、feature matrix 统统用 UVM；
  * 单 GPU 版本即可（multi‑GPU 留给系统部分）。
* 参数控制：

  * graph 用 ogbn-products / ogbn-papers100M 的子图（sampling 不要太 aggressive）；
  * 扫 feature 维度 d，使得 `|H| + |H'| ≈ {0.25×–2×} GPU mem`。
* 意义：

  * Irregular neighbor aggregation + medium compute intensity；
  * 直接给你解释 MGG 里 MGG‑UVM 那条惨不忍睹的曲线的 micro‑bench 版本。

---

## 三、实验矩阵重写：real kernel 为主，synthetic 为辅

### 3.1 维度 1：内存模式（还是那四个）

对**每个 Tier‑1 / Tier‑2 kernel**（至少）跑四种模式：

1. **Device‑only baseline（显式显存 + memcpy）**

   * 所有数据 `cudaMalloc`；
   * host side 用 pinned mem + `cudaMemcpyAsync` 预拷贝；
   * kernel 本身只碰 device mem；
   * optional：对 oversub 实验用 “滑动窗口 + double buffering” 版本做 best‑effort out‑of‑core baseline。

2. **UVM（默认，无 prefetch）**

   * 全部 `cudaMallocManaged`；
   * 不用 `cudaMemPrefetchAsync` / `cudaMemAdvise`；
   * 完全交给 UVM driver。

3. **UVM + prefetch to GPU**

   * kernel 前调用 `cudaMemPrefetchAsync(ptr, size, device)` 对所有主要数组 prefetch；
   * 对 LLM/GNN 这类带 KV/feature big tensor 的，考虑只对热数据 prefetch。

4. **UVM + oversubscription**

   * working set > GPU mem；
   * 模拟 “KV cache / graph / activation 放不下” 的真实情况。

这四个是 **固定组合**，贯穿所有 kernel，这样 reviewer 会觉得你是在做系统性对比，而不是 cherry-pick。

---

### 3.2 维度 2：working set / GPU mem 比例

对每个 kernel 至少扫这些点：

* `S / GPU_mem ∈ {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}`
* 对 LLM / KV cache，可以再加 `3.0, 4.0`，模拟特别长 context。

不要每个 kernel 都把所有点跑齐，但至少：

* dense GEMM / stencil：跑到 1.5× 或 2×；
* SpMV / BFS / GNN：重点看 1.0×–1.5× 这一段，说明 “刚开始 oversub 的时候已经很惨”；
* Transformer：跑全程，说明 LLM KV 场景下 UVM 有多不可用。

---

### 3.3 维度 3：并发度 / 算密度

这里不要搞到 combinatorial explosion，只挑关键两个维度：

* **并发度**：

  * 给每个 kernel 选两种 launch config：

    * “低 occupancy”（gridDim ≈ #SM, blockDim 小）；
    * “高 occupancy”（gridDim > 4×#SM, blockDim = 256/512）；
  * 用来验证：简单多发几个 warps 并不能掩盖 UVM latency（援引 SC’21 的结论）。

* **算密度（compute intensity）**（只对 stencil / GNN / Transformer 做）：

  * stencil：调整 time steps t = 1 vs t = 10；
  * GNN：增加 feature dim / 增加 nonlinearity；
  * Transformer：加入/去掉 MLP 部分。
    目的：看看 compute heavy kernel 是否对 UVM 更“宽容”。

---

## 四、实现建议：怎么把 real kernel 塞进同一个 microbench 框架

### 4.1 框架结构

你可以设计成一个统一的 `uvmbench` 可执行程序，形态类似 UVMBench，但更偏 “mechanism measurement”：

* 命令行参数：

  * `--kernel={gemm,stencil2d,spmv,bfs,conv,transformer,gnn,...}`
  * `--mode={device,uvm,uvm_prefetch,uvm_oversub}`
  * `--size_factor=0.25..4.0`（按 GPU mem 比例）
  * `--occupancy={low,high}`
* 内部每个 kernel 是一个 C++/CUDA 函数：

  * `run_gemm(const Config&, Result*)`
  * `run_stencil2d(...)`
  * …
* 每个函数负责：

  1. 按 `size_factor` 推导实际问题规模（N, nnz, H×W, seq_len 等）；
  2. 按 `mode` 选择 `cudaMalloc` 还是 `cudaMallocManaged` + prefetch；
  3. 调用 cuBLAS/cuDNN/cuSPARSE 或自己的 kernel；
  4. 用 `cudaEvent` 计时，跑多次取 median；
  5. 收集 profiler/driver 统计（后面说）。

### 4.2 具体 kernel 实现建议

* GEMM：

  * 用 cuBLAS `cublasSgemm`，调用前后插 event 计时；
  * 确保 A,B,C 都是 managed 或 device。
* Stencil：

  * 写自己的 2D/3D kernel（教材级别那个），用 shared mem 做 basic 优化，保证不是完全 naive；
  * 迭代 t 次。
* SpMV：

  * cuSPARSE `cusparseSpMV`，CSR + managed arrays；
  * 对 nnz 很大的矩阵（百万以上）才有意义。
* BFS / PageRank：

  * 可以直接拿 Gunrock 的 BFS kernel 抽出 main loop，简化成单 GPU 版本；
  * 或者用 UVMBench 改好的 BFS 变种。
* Conv layer：

  * cuDNN `cudnnConvolutionForward` + BN + ReLU；
  * 或者直接在 PyTorch C++ API 里 instantiate 一层 conv 模型，跑 forward；
  * 反正内存模式对他们都是透明的。
* Transformer：

  * 可以复用 HuggingFace/DeepSpeed 中某个 decoder layer 的 CUDA kernel，或者简单用 PyTorch Eager 的一层，前提是你能控制 memory 分配方式（managed vs device）。
* GNN layer：

  * 用 DGL/PyG 的 GCNConv/GATConv 内核，自己写 wrapper 控制 memory；
  * 或者照 MGG 的 pseudo‑code 实現一层 gather‑aggregate kernel。

这部分的策略很简单：**凡是有成熟库的，就用库**，不要自己造一个 naive 版本然后被 reviewer 问“为啥你不跑 cuBLAS/cuDNN”。

---

## 五、测量与图：重写版（以 real kernel 为主）

### 5.1 指标

和之前说的类似，但强调要在每个 real kernel 上都收：

* Runtime：每次 kernel 或每个 iteration 的 wall‑clock（取 median 或 P95）；
* Effective throughput：

  * 对 GEMM：GFLOPS；
  * 对 stencil/SpMV/BFS：GB/s 或 GEdges/s；
  * 对 conv：images/s；
  * 对 Transformer/GNN：tokens/s 或 nodes/s；
* Slowdown vs device‑only baseline；
* UVM‑specific：

  * migrated bytes；
  * #page faults；
  * batch 处理时间（如果你 patch 了 driver）。
* GPU counters：

  * 内存带宽利用率；
  * SM stall breakdown。

### 5.2 推荐图（这次全部以 real kernel 为主）

#### 图 A：不同 kernel 的整体性能损失

* X 轴：kernel 类型（GEMM, stencil2d, SpMV, BFS, Conv, Transformer, GNN）
* Y 轴：slowdown (UVM vs device‑only) 在 `S = 1.0× GPU mem` 时的值；
* 每个 kernel 画两个 bar：

  * UVM no prefetch；
  * UVM + prefetch。

**解释方向：**

* 可以直接写一句非常“硬”的话：

  * “On average, UVM slows down real‑world kernels by 1.3×–8× even when data fits in GPU memory; sparse and graph kernels suffer the most.”
* 把 SC’21 “管理开销 > memcpy”的结论搬出来佐证。

#### 图 B：Oversubscription 扫描（per kernel）

选代表性的 3–4 个 kernel（比如 GEMM, stencil2d, SpMV, Transformer），画 4 张类似的图：

* X 轴：`S / GPU_mem`；
* Y 轴：throughput（normalize 到 device‑only fit‑in mem 情况 = 1.0）；
* 曲线：device sliding‑window baseline（如果有）、UVM、UVM+prefetch。

**你能讲的故事：**

* GEMM 在 1.25× 之前还能苟一苟，2× 基本折半甚至更差；
* SpMV / BFS 在 1.1× 左右就开始疯狂 thrashing；
* Transformer/LLM 在 >1× 时完全 PCIe‑bound，延迟线性/超线性炸裂（对应 InfiniGen 里 UVM baseline 的行为）。

#### 图 C：算密度对 UVM 的影响（stencil / GNN / Transformer）

* 例如对 stencil2d，画：

  * X 轴：time steps t；
  * Y 轴：slowdown (UVM vs device‑only)，分别在 `S=1.0×` 和 `1.25×` 两种规模下。
* 类似地，对 GNN/Transformer 改变 feature dim / 是否包含 MLP。

**解释：**

* 如果算密度加大，对 GEMM/Conv 来说 UVM 的额外 latency 可以部分隐藏；
* 对极度 memory bound 的 SpMV/BFS 基本帮不上忙；
* 这直接支持后面系统设计里“某些 kernel 适合作为 out‑of‑core candidate，某些完全不适合”。

#### 图 D：UVM 行为解剖（从 real kernel 映射回 synthetic）

这一步是把 Tier‑0 synthetic 拿回来发挥作用：

* 先给例如 SpMV/BFS/Transformer 分析他们的访问 pattern：

  * 通过 driver instrumentation 或 trace 统计「每次 kernel 实际访问的 unique pages / total accesses」；
  * 推出一个“等价 stride / randomness”。
* 然后把这些 pattern 映射回你之前的 `seq_stream` / `rand_stream` synthetic 实验中对应的点，展示：

  * “the behavior of real kernels falls between synthetic cases X and Y”；
* 用来证明你对 UVM 行为的解释不是拍脑袋，而是有系统 mapping 的。

Reviewer 看这个会觉得你不是瞎凑 microbench，而是**用 synthetic 作为分析工具，真正服务 real workload**。

---

## 六、怎么在论文里“讲”这套设计（而不是像 benchmark paper 那样罗列）

你最后的 narrative，应该不是“我们设计了一套很酷的 benchmark”，而是类似：

1. **先定义 scope：**

   * “We focus on four representative application domains where UVM is widely used or frequently proposed as a fallback: dense HPC (GEMM), stencil/HPC simulations, sparse/graph analytics, and DL/LLM/GNN workloads.”
   * 引 SC’21、UVMBench、PipeSwitch、MGG、InfiniGen，证明这四类不是瞎挑的。

2. **然后说选择的 kernel 是 “realistic building blocks”：**

   * “Our benchmark suite consists of seven real‑world kernels that are directly reused or abstracted from widely‑deployed libraries and systems (cuBLAS/cuDNN/cuSPARSE, Gunrock, DGL, HuggingFace), plus a small set of synthetic kernels used only for low‑level analysis.”

3. **接着强调你不是搞 workload zoo，而是为了回答前面那 4–5 个 RQ：**

   * 每个 kernel 类别对应某个 RQ 的极端点（best‑case / worst‑case）。

4. **最后用图 A–D 把故事串起来：**

   * “UVM’s best‑case overhead even on dense GEMM is X–Y×…”
   * “Irregular kernels such as SpMV and BFS suffer Z× slowdown even when their working set fits in GPU memory…”
   * “Naive prefetch helps only in dense/stencil workloads; in LLM‑style KV access, UVM becomes completely PCIe‑bound beyond 1.25× oversubscription…”

你要真按这个路数来写，**microbench 部分就已经是一个非常 solid 的 measurement section**，接下来你可以自然引出任何你想做的机制（比如 device‑driven migration、sub‑page UVM、multi‑GPU coordinated eviction 等等）。

---

如果你接下来想，我可以直接把这一套 Tier‑1/Tier‑2 kernel 列成一个“实验 checklist”（比如：行 = kernel，列 = mode × size_factor × occupancy，标出必须跑的组合、optional 的组合），你丢给学生去实现的时候就不会乱。
