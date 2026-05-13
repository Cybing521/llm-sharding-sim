# LLM Sharding Simulator

面向**异构边缘设备集群**的大语言模型推理分片模拟器。给定一个 LLM 的层级结构和一组具有不同算力/内存/带宽的边缘设备，本工具通过求解器决定**模型层放置（placement）**和**请求路由（routing）**策略，并通过离散时间仿真引擎验证端到端推理延迟。

## 目录

- [项目结构](#项目结构)
- [核心概念](#核心概念)
- [数学建模](#数学建模)
- [模块说明](#模块说明)
- [配置文件](#配置文件)
- [快速开始](#快速开始)
- [实验说明](#实验说明)
- [输出示例](#输出示例)

## 项目结构

```
llm-sharding-sim/
├── main.py                     # 入口：加载配置 → 求解 → 仿真 → 可视化
├── requirements.txt            # Python 依赖
├── config/
│   ├── model.yaml              # 模型结构（LLaMA-3.2-3B）
│   ├── devices.yaml            # 设备集群 & 带宽/延迟矩阵
│   └── requests.yaml           # 推理请求定义
├── simulator/
│   ├── model.py                # ModelConfig / LayerInfo — 模型层级描述
│   ├── device.py               # Device / DeviceCluster — 设备与网络拓扑
│   ├── request.py              # Request — 推理请求
│   ├── cost.py                 # 解析式代价函数（t_prc, t_dec, t_trans）
│   └── engine.py               # 离散时间仿真引擎（KV Cache、传输延迟）
├── solver/
│   ├── base_solver.py          # BaseSolver / SolverResult 抽象基类
│   ├── greedy_solver.py        # 贪心求解器（连续层填充）
│   └── ilp_solver.py           # ILP 求解器（PuLP/CBC 整数线性规划）
├── analysis/
│   ├── compare.py              # 求解器对比报告
│   └── visualize.py            # 可视化（内存时间线、热力图等）
└── results/                    # 实验输出（报告 + 图表）
    ├── exp1_single_request/
    ├── exp2_multi_request/
    ├── exp3_prompt_sweep/
    └── exp4_bandwidth_sweep/
```

## 核心概念

### 层级放置（Layer Placement）

将 LLM 的各层（embedding → transformer blocks → lm_head）分配到异构设备上，受限于每台设备的内存容量。放置用二值矩阵 `x[u,k]` 表示：层 `u` 是否部署在设备 `k` 上。

### 请求路由（Request Routing）

对于每条推理请求，决定其各层在哪台设备上执行。路由用三维二值矩阵 `z[q,u,k]` 表示：请求 `q` 的层 `u` 在设备 `k` 上执行。路由必须服从放置约束（`z[q,u,k] ≤ x[u,k]`）。

### 代价模型

每条请求的端到端延迟分解为三部分：

| 符号 | 含义 | 计算方式 |
|------|------|----------|
| `t_prc` | Prefill 延迟 | 所有层处理 \|r_q\| 个 prompt token |
| `t_dec` | Decode 延迟 | (g_q - 1) 步自回归，每步各层处理 1 token（跳过 embedding 层） |
| `t_trans` | 传输延迟 | 相邻层位于不同设备时的 activation 传输代价 |

### KV Cache

每个 transformer 层在处理完 token 后会产生 KV Cache。仿真引擎会逐时间步追踪每台设备上的 KV Cache 占用，请求完成后释放。

## 数学建模

### 决策变量

- **`x[u,k]`** ∈ {0,1} — 层 `u` 是否放置在设备 `k`
- **`z[q,u,k]`** ∈ {0,1} — 请求 `q` 的层 `u` 是否由设备 `k` 执行
- **`y[q,u,k,k']`** ∈ {0,1} — 辅助变量，线性化 `z[q,u,k] × z[q,u+1,k']` 的乘积项

### 目标函数

最小化所有请求的总延迟：

```
min  Σ_q ( t_prc(q) + t_dec(q) + t_trans(q) )
```

其中：

```
t_prc(q)   = Σ_u Σ_k  z[q,u,k] · (r_q / c_k)
t_dec(q)   = Σ_{u≥1} Σ_k  z[q,u,k] · ((g_q - 1) / c_k)
t_trans(q) = Σ_u Σ_{k≠k'} y[q,u,k,k'] · g_q · transfer_time(k, k')
```

- `r_q` = prompt 长度，`g_q` = 输出长度，`c_k` = 设备 k 的每层推理速度 (tokens/s)

### 约束条件

1. **内存约束**：`Σ_u x[u,k] · size(u) ≤ h_k`（设备 k 的内存上限）
2. **放置约束**：每层至少部署在一台设备上
3. **路由唯一性**：每条请求的每层恰好由一台设备执行
4. **路由从属放置**：`z[q,u,k] ≤ x[u,k]`
5. **乘积线性化**：标准 McCormick 不等式约束 `y` 变量

## 模块说明

### `simulator/model.py`

- **`LayerInfo`** — 单层元数据（索引、大小 MB、类型）
- **`ModelConfig`** — 完整模型描述，从 YAML 构建层列表（embedding + N × transformer + lm_head），提供 KV Cache 和 activation 大小参数

### `simulator/device.py`

- **`Device`** — 单设备属性：内存容量、算力（GFLOPS）、每层推理速度（tokens/s/layer）
- **`DeviceCluster`** — 设备集合 + 带宽矩阵 `bandwidth_mbps[i,j]` + 延迟矩阵 `latency_ms[i,j]`
- **`transfer_time_s(src, dst, data_bytes)`** — 计算跨设备传输耗时

### `simulator/request.py`

- **`Request`** — 推理请求：prompt 长度、预估输出长度、到达设备、到达时间
- **`load_requests()`** — 从 YAML 加载请求列表

### `simulator/cost.py`

- **`compute_request_delay()`** — 给定单条请求的路由 `z_q`，解析计算 `t_prc`、`t_dec`、`t_trans`
- **`compute_total_cost()`** — 汇总所有请求的总延迟
- **`check_memory_feasibility()`** — 检查放置方案是否满足静态内存约束

### `simulator/engine.py`

离散时间仿真引擎，以固定时间步 `tick_s` 推进：

- 请求生命周期：`WAITING → PREFILL → DECODE → COMPLETE`
- Prefill 阶段：逐层处理所有 prompt token，层间可能需要跨设备传输 activation
- Decode 阶段：自回归逐 token 生成，每步从 layer 1 遍历到 lm_head（跳过 embedding）
- 逐时间步追踪 KV Cache 增长与释放
- 输出仿真时间 vs 解析时间的对比结果

### `solver/greedy_solver.py`

贪心求解器策略：
1. 按推理速度降序排列设备（速度相同则按内存降序）
2. 将连续的层贪心填充到当前设备，直到内存不足则切换下一台设备
3. 未放置的层通过回退扫描分配
4. 路由：每层仅有唯一放置时直接路由；多副本时选择距到达设备延迟最低的

### `solver/ilp_solver.py`

整数线性规划（ILP）求解器：
- 使用 PuLP 库建模，CBC 后端求解
- 精确建模目标函数与所有约束（见上方数学建模）
- 通过辅助变量 `y` 线性化相邻层跨设备的乘积项
- 支持设置求解时间上限 `time_limit`

### `analysis/compare.py`

生成文本格式的求解器对比报告，包含内存使用检查、各请求的延迟分解、层分配方案。

### `analysis/visualize.py`

Matplotlib 可视化工具：

| 函数 | 说明 |
|------|------|
| `plot_memory_timeline()` | 各设备内存（权重 + KV Cache）随时间变化的面积图 |
| `plot_delay_comparison()` | 不同求解器的请求延迟柱状图 |
| `plot_layer_assignment_heatmap()` | 层-设备分配热力图 |
| `plot_prompt_length_sweep()` | 延迟随 prompt 长度变化的折线图 |

## 配置文件

### `config/model.yaml`

```yaml
name: "LLaMA-3.2-3B"
precision: "fp16"
hidden_size: 3072
num_attention_heads: 24
num_kv_heads: 8
head_dim: 128
layers:
  embedding: { index: 0, size_mb: 751 }
  transformer: { count: 28, index_start: 1, index_end: 28, size_per_layer_mb: 192 }
  lm_head: { index: 29, size_mb: 751 }
kv_cache:
  bytes_per_layer_per_token: 4096   # 2 * num_kv_heads * head_dim * bytes_per_param
activation_size_bytes: 6144         # hidden_size * bytes_per_param
```

模型共 30 层（1 embedding + 28 transformer + 1 lm_head），总权重约 6878 MB。

### `config/devices.yaml`

定义 5 台 Jetson 系列边缘设备：

| 设备 | 内存 | 推理速度 (tok/s/layer) |
|------|------|----------------------|
| jetson_nano_0/1/2 | 3072 MB | 15.0 |
| jetson_xavier_0/1 | 6144 MB | 200.0 |

设备间带宽 12.5 MB/s（Xavier 间 25.0 MB/s），延迟 1-2 ms。

### `config/requests.yaml`

```yaml
requests:
  - { id: "r0", prompt_length: 64,  estimated_output_length: 128, arrival_device: 0, arrival_time: 0.0 }
  - { id: "r1", prompt_length: 32,  estimated_output_length: 64,  arrival_device: 1, arrival_time: 0.5 }
  - { id: "r2", prompt_length: 128, estimated_output_length: 256, arrival_device: 3, arrival_time: 1.0 }
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖项：`pulp`、`pyyaml`、`matplotlib`、`numpy`

### 运行全部实验

```bash
python main.py
```

### 运行指定实验

```bash
python main.py --experiment 1        # 仅运行实验 1
python main.py --experiment 3        # 仅运行实验 3
```

### 自定义路径

```bash
python main.py --config-dir my_config --results-dir my_results
```

## 实验说明

### 实验 1：单请求 — ILP vs Greedy

取第一条请求，分别用贪心和 ILP 求解器求解最优放置与路由。对比两种方案的解析延迟和仿真延迟，生成：

- 内存时间线图（各设备权重 + KV Cache 占用）
- 层分配热力图
- 延迟对比柱状图
- 文本对比报告

### 实验 2：多请求并发 — KV Cache 压力测试

使用全部请求，贪心求解器求解。多请求并发仿真，观察 KV Cache 在设备上的动态增长与释放，验证内存压力。

### 实验 3：Prompt 长度扫描

固定输出长度为 64 token，扫描 prompt 长度 {32, 64, 128, 256}，对比 Greedy 和 ILP 的解析延迟随 prompt 长度的变化趋势。

### 实验 4：带宽扫描

固定单条请求，将基础带宽按 {0.5×, 1×, 2×, 5×, 10×} 缩放，分析网络带宽对总延迟的影响，揭示传输瓶颈。

## 实验结果

以下为默认配置（LLaMA-3.2-3B, 5 台 Jetson 设备）下的实验运行结果。

### 实验 1 结果：单请求 — ILP vs Greedy

**请求**：r0 (prompt=64, output=128, arrival_device=0)

#### 层分配对比

| 求解器 | 求解时间 | 分配方案 |
|--------|----------|----------|
| Greedy | 0.000s | xavier_0: L0-L28 (6127 MB), xavier_1: L29 (751 MB) |
| ILP | 0.222s | nano_0/1/2: L0 副本, xavier_0: L0-L1, xavier_1: L2-L29 |

Greedy 倾向于将所有层集中在最快设备上；ILP 会在多台设备上放置 embedding 层副本以支持多请求路由。

#### 延迟分解

| 求解器 | Prefill (t_prc) | Decode (t_dec) | Transfer (t_trans) | **Total** |
|--------|-----------------|----------------|--------------------|-----------|
| Greedy | 9.6000s | 18.4150s | 0.1580s | **28.1730s** |
| ILP | 9.6000s | 18.4150s | 0.1580s | **28.1730s** |

两种策略在单请求场景下总延迟一致（28.173s），因为 ILP 的路由选择了与 Greedy 等效的路径：关键层都在 Xavier 设备上执行。

#### 仿真 vs 解析

| 求解器 | 解析延迟 | 仿真延迟 | 差异 |
|--------|----------|----------|------|
| Greedy | 28.173s | 28.655s | +1.7% |
| ILP | 28.173s | 28.655s | +1.7% |

仿真延迟略高于解析值，主要由离散时间步的量化误差引起。

---

### 实验 2 结果：多请求并发 — KV Cache 压力

**层分配**：Greedy — xavier_0 承载 L0-L28, xavier_1 承载 L29

#### 各请求延迟

| 请求 | Prompt | Output | 解析延迟 | 仿真延迟 | 差异 |
|------|--------|--------|----------|----------|------|
| r0 | 64 | 128 | 28.173s | 28.655s | +1.7% |
| r1 | 32 | 64 | 14.014s | 14.255s | +1.7% |
| r2 | 128 | 256 | 56.491s | 57.455s | +1.7% |

**三请求总解析延迟**：98.678s

**延迟分解**：

| 请求 | t_prc | t_dec | t_trans | t_total |
|------|-------|-------|---------|---------|
| r0 | 9.600s | 18.415s | 0.158s | 28.173s |
| r1 | 4.800s | 9.135s | 0.079s | 14.014s |
| r2 | 19.200s | 36.975s | 0.316s | 56.491s |

**内存分布**：xavier_0 静态权重占用 6127/6144 MB，仅剩 17 MB 余量。KV Cache 在仿真中动态增长，r2（128+256 tokens）对设备内存形成显著压力。

---

### 实验 3 结果：Prompt 长度扫描

固定 output_length=64，扫描不同 prompt 长度：

| Prompt 长度 | Greedy 延迟 | ILP 延迟 |
|-------------|------------|----------|
| 32 | 14.014s | 14.014s |
| 64 | 18.814s | 18.814s |
| 128 | 28.414s | 28.414s |
| 256 | 47.614s | 47.614s |

**观察**：
- 延迟与 prompt 长度近似线性增长（主导项为 prefill 的 `r_q / c_k`）
- Greedy 与 ILP 在单请求下延迟完全一致，表明当前设备配置下贪心策略已找到最优路由
- 从 32→256，延迟增长 3.4 倍，说明 prefill 计算是长 prompt 场景的主要瓶颈

---

### 实验 4 结果：带宽扫描

固定请求 r0（prompt=64, output=128），缩放基础带宽：

| 带宽倍率 | Greedy 延迟 | ILP 延迟 | 传输占比 |
|----------|------------|----------|----------|
| 0.5x | 28.203s | 28.203s | 0.11% |
| 1.0x (基线) | 28.173s | 28.173s | 0.06% |
| 2.0x | 28.158s | 28.158s | 0.03% |
| 5.0x | 28.149s | 28.149s | 0.01% |
| 10.0x | 28.146s | 28.146s | <0.01% |

**观察**：
- 带宽从 0.5x 提升到 10x，总延迟仅从 28.203s 降至 28.146s（降幅 0.2%）
- 传输延迟占总延迟不足 0.2%，说明当前场景的瓶颈完全在**计算**而非网络传输
- 当层大多集中在同一设备时，跨设备传输次数很少（仅 L28→L29 一次跳转），带宽影响微弱

---

### 输出文件

实验结果保存在 `results/` 目录下：

```
results/
├── exp1_single_request/
│   ├── report.txt              # 求解器对比报告
│   ├── memory_greedy.png       # Greedy 内存时间线
│   ├── memory_ilp.png          # ILP 内存时间线
│   ├── heatmap_greedy.png      # Greedy 层分配热力图
│   ├── heatmap_ilp.png         # ILP 层分配热力图
│   └── delay_comparison.png    # 延迟对比柱状图
├── exp2_multi_request/
│   ├── report.txt
│   ├── memory_timeline.png     # 多请求内存时间线（含 KV Cache 增长）
│   └── heatmap.png
├── exp3_prompt_sweep/
│   ├── prompt_sweep.png        # 延迟 vs prompt 长度折线图
│   └── sweep_data.txt
└── exp4_bandwidth_sweep/
    ├── bandwidth_sweep.png     # 延迟 vs 带宽倍率折线图
    └── sweep_data.txt
```
