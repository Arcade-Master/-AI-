# 04｜显存估算、ZeRO、并行、Gradient Checkpoint、FlashAttention

对应：**第一批**（DeepSpeed 概要、FP16）；**第二批**（14B/7B 显存、显存因素、ZeRO-1/2/3、训练卡数与异常、DeepSpeed vs FSDP、FlashAttention、Gradient Checkpoint、并行策略）。

---

## 1. 显存大致由什么组成？

训练时大头通常包括：

- **模型参数**（每参数 2 bytes @ BF16/FP16，或 4 @ FP32）
- **梯度**（与可训练参数同量级）
- **优化器状态**（Adam 类：m、v 等，常 FP32 → **每参数约 +8 bytes** 量级理解 AdamW 两矩）
- **激活**（前向中间结果；与 batch、seq_len、hidden、层数相关，常是 **OOM 主因**）
- **临时 buffer**（通信、kernel workspace）

**推理**：参数 + **KV Cache**（随 batch、层数、seq、head 维增长）+ 少量激活与 workspace。

---

## 2. 粗算公式（面试数量级，不是精确厂商实现）

设参数量为 P（如 7B ≈ 7e9），一种 **混合精度（FP16/BF16 训 + Adam 状态 FP32）** 的常见口算：

- **前两个 2**：**参数**、**梯度** 各按 **半精度 2 bytes**（FP16 或 BF16；与「权重用半精度算」一致）。
- **后两个 4**：Adam 的 **m（一阶矩）、v（二阶矩）** 框架里通常 **FP32 存**，各 **4 bytes**——数值稳定，避免半精度下矩估计炸掉；**不是**「优化器也用 FP16」。

若全套 **FP32 训练**（参数、梯度、两矩全是 float32），则更像 `P × (4+4+4+4) = 16P`，与 12P 的口算要分开记。

```
约 = P × (2 + 2 + 4 + 4) bytes = P × 12 bytes（混合精度下梯度可能 FP16，会有出入）
7B → 7e9 × 12 ≈ 84 GB 量级（仅优化器相关+梯度的一种上界谈法）
```

更松的口算：**7B 全量 Adam 训练** 常听到「**至少百 GB 级** 才舒服」——因为还有 **激活、框架开销、并行分片前单卡要扛的峰值** 等。

**推理 14B BF16**：权重约 `14e9 × 2 ≈ 28 GB`；再加 KV、框架、并发 batch，**单卡 40GB+** 更现实。

**INT4 权重**（W4A16 等）：权重大致 **÷4**，但激活与 KV 仍可能吃满带宽；「能装下」≠「跑得爽」。

**KV Cache**（与 02 篇一致）：

```
每层: 2 × batch × seq × num_kv_heads × head_dim × dtype_bytes
乘层数
```

MQA/GQA/MLA 通过减小 **num_kv_heads** 或压缩 KV 表示来省缓存。

### 激活显存怎么粗算（训练）

面试先讲 **「和什么成正比」**，再讲 **attention 是否把 score 整块 materialize 成 `S×S`**——两者差一个 **S²**。

- **主链 hidden 类张量（LN 前后、残差、MLP 中间等）**：量级上常记 **每层若干份** 形状约为 **`[B, S, H]`** 的激活（`B`=micro-batch，`S`=seq，`H`=hidden），再乘 **`dtype_bytes`**、乘 **层数 `L`**。整体与 **`B · S · H · L`** 同阶，前面系数依实现（几份 QKV、是否重算）多在常数级，**不要死背一个倍数**。
- **若标准 Attention 把 score 整块 materialize 到 HBM**：每层多一项约 **`B × n_heads × S × S × elem_bytes`**（FP16 则 `elem_bytes=2`）。**对齐多头**：`Q/K/V` 常按 **`[B, n_heads, S, head_dim]`**，其中 **`head_dim ≈ H / n_heads`**——**切的是特征维，不是把 `S` 拆开给各头**；每个头仍对 **整条 `S`** 做 softmax 注意力，故 **每个头一张 `S×S` logits**，`n_heads` 份并列即 **`× n_heads`**。此时 **`S` 翻倍 → 这一项约 ×4**，长序列下激活常 **压过** 权重+优化器，成为 **OOM 首因**；**FlashAttention / 融合核** 通过不常驻完整 **`S×S` score 矩阵**，把峰值从 **O(S²)** 拉到更接近 **O(S)**（仍随 `B,H,L` 增）。
- **Gradient Checkpointing**：减少 **同一时刻存活** 的激活条数 → **降峰值**；代价是多算前向，**不算子项时也要在口述里提一句**。

### 并行后「单卡峰值」为何还在、量级怎么说

- **数据并行 + ZeRO-1/2**：切的是 **优化器状态 / 梯度**；**每张卡仍跑完整前向/反向图**，**激活按本卡 micro-batch 的 `B,S,H` 长**，**一般不会**因为「世界规模 N 张卡」就 **除以 N**。
- **ZeRO-3 / FSDP**：参数分片后，算子前常要 **短时 all-gather 出整层（或一块）完整权重**，显存会出现 **「常驻分片 + 通信/gather 尖峰」**；小模型、高并行、大层时尖峰更明显。
- **TP**：层内切分算子往往需要 **通信 buffer** 或 **临时拼回的中间张量**，峰值 **不低于**「最胖那个算子」的需求。
- **PP**：调度与 **micro-batch 数** 决定管道里是否 **同时驻留多段激活**；bad schedule 时单卡峰值 **不按直觉下降**。

**口述量级**：先分开估 **状态（12P/16P 那套）** 与 **激活（上两小节）**，再说 **多卡主要省状态、激活仍按单卡 batch 算、ZeRO-3/通信还有 buffer 尖峰**——比报一个固定「百分之多少」更稳。

### 框架 / 运行库开销从哪里来

- **cuBLAS / cuDNN 等 workspace**：大矩阵乘、卷积、部分融合 Attention 会申请 **kernel 临时工作区**；可通过环境变量或 API **设上限**，但可能影响 **可用算法/速度**。
- **PyTorch Caching Allocator**：**缓存池、对齐、碎片** 会让 **`nvidia-smi` 占用 > 把各 tensor `.element_size()*numel()` 手加一遍**；峰值还包含 **已分配未立即归还** 的块。
- **分布式**：梯度 **bucket**、**all-reduce / reduce-scatter staging**；ZeRO **gather/scatter** 的 **临时 buffer**。
- **编译/图优化**（如 `torch.compile`）：可能引入 **额外中间缓冲或编译期峰值**，需实测。

---

## 3. DeepSpeed ZeRO-1 / 2 / 3：各切什么？省什么？通信怎样？

**共同思想**：数据并行时每张卡 **不必存完整冗余** 的优化器状态/梯度/参数，改 **分片 + all-gather / reduce-scatter**。

| 阶段 | 切分对象 | 直觉效果 |
|------|----------|----------|
| ZeRO-1 | **优化器状态** 分片 | 省大量 optimizer state 显存 |
| ZeRO-2 | + **梯度** 分片 | 再省梯度常驻 |
| ZeRO-3 | + **参数** 分片 | 单卡几乎只持有一部分权重；可训更大模型 |

**通信**：分片越激进，前向/反向越常需要 **按算子 all-gather 参数**、reduce-scatter 梯度，**延迟与带宽压力上升**，故 ZeRO-3 往往 **更慢**（尤其小 batch、多卡、网络一般时）。

**「省多少」**：与卡数、分片策略、是否 offload 有关，面试答 **「阶上可接近按卡数分摊状态，但受通信与激活限制」** 比背固定倍数更靠谱。

---

## 4. DeepSpeed vs FSDP（PyTorch）

**相似**：都可做分片式数据并行（sharded DDP），降低每卡峰值显存。

**差异（答思路）**：

- FSDP 是 PyTorch **原生** API，与 torch 生态集成紧；DeepSpeed 是 **独立训练引擎**，插件化 ZeRO、offload、1-bit Adam、pipeline 等 **一体化配置** 多。
- 大团队常 **二者选一或混用组件**；选型看维护成本、已有脚本、是否需 DeepSpeed 特有优化。

---

## 5. Tensor / Pipeline / Data / Sequence Parallel

- **Data Parallel（DP）**：每卡完整模型，不同数据子集；梯度 all-reduce。显存 **不省模型**，省不了单卡扛全模。
- **Tensor Parallel（TP）**：层内矩阵切分，单卡只存部分参数；**卡间高频通信**，适合机内 NVLink。
- **Pipeline Parallel（PP）**：**按层分段**到多卡，用 **micro-batch 流水** 传 **stage 边界激活**；有 **bubble**。实现与约束见 **§5.4**。
- **Sequence Parallel（SP）**：**按 token 维（序列维 `S`）** 在同层内切开激活，配合 **all-gather / split** 与 **TP** 使用；不是 DP 那种「每卡不同样本」。见 **§5.4**。

**场景口诀**：机内超大层 → TP；跨机超大模 → PP + ZeRO；超长上下文 → SP / 重算 / Flash。

---

## 5.1 数据并行 vs 模型并行：数据流与梯度如何更新？

（题面里「模型冰箱」一般为 **模型并行** 的笔误，下面按 **模型并行** 讲解。）

### 数据并行（Data Parallel, DP）

- **每卡一份完整模型**，输入 **不同 micro-batch**（数据切分）。  
- **前向**：各卡独立算自己的 loss。  
- **反向**：各卡得到 **本地梯度** `g_i`。  
- **同步**：对所有卡的梯度做 **all-reduce（求平均或求和后再除 world size）**，使各卡 **持有相同 `ḡ`**。  
- **更新**：各卡用 **相同 `ḡ`** 更新本地参数 → **下一轮各卡参数仍一致**（若实现正确）。

**数据流**：数据 **分散**；激活与梯度 **局部**；梯度 **汇聚** 后广播式等价更新。

### 模型并行（Model Parallel, MP）

**张量并行（TP）** 与 **流水线并行（PP）** 都属于模型并行大类：**单份模型拆到多卡**，每卡只存 **部分参数**。

- **前向**：卡间传递 **中间激活**（如 attention 里切分的 QKV 块、MLP 的部分 hidden）。  
- **反向**：链式法则下 **梯度沿相同通信边回传**（reduce-scatter / all-to-all 依切分方式）。  
- **数据**：同一 batch（PP 下常切成 **micro-batch** 流水）在设备间 **流水或广播**，与 DP 的「每卡不同子 batch」**概念不同**。

**梯度**：对每个参数子集 **局部计算**，再通过 **collective** 与相邻分片 **对齐**（具体算子依赖切分方案）。

**一句话对比**：DP 是 **同构复制 + 数据分片 + 梯度 all-reduce**；MP 是 **异构分片 + 激活/梯度跨卡通信**。

---

## 5.2 DP 与 DDP 的区别与实现要点

| | **DP（DataParallel）** | **DDP（DistributedDataParallel）** |
|---|-------------------------|-------------------------------------|
| **进程** | 常为 **单进程多 GPU**，主卡聚合 | **每 GPU 一进程**（推荐） |
| **梯度同步** | 单进程内 **gather 到主卡** 再广播 | **`torch.distributed` all-reduce** |
| **性能** | Python GIL、主卡瓶颈 | **更接近线性扩展**（受通信与 batch 限制） |
| **适用** | 单机小玩具 / 老代码 | **正式训练默认** |

**DDP 实现要点（口述）**：

1. `torchrun` / `mp.spawn` 启动 **world_size** 个进程，各绑 **一卡**。  
2. `dist.init_process_group(backend="nccl")`。  
3. `model = DDP(model, device_ids=[local_rank])`。  
4. 每进程 **不同 DistributedSampler** 保证 **数据不重复**。  
5. `loss.backward()` 后梯度 **自动 all-reduce**；`optimizer.step()` 各进程一致。

---

## 5.3 Pipeline Parallel（PP）与 Tensor Parallel（TP）再讲透一点

**TP（层内切）**

- 把 **大矩阵乘** 按列/行切到多卡；每步需 **all-gather 或 reduce-scatter**（依算子）。  
- **优点**：单卡装不下单层时 **可训**；**缺点**：**通信极频繁**，强依赖 **机内 NVLink**。

**PP（层间切）**

- GPU0 管层 1～L1，GPU1 管 L1+1～L2… **微批次流水** 填满管道。  
- **优点**：**跨机** 也能扩；**缺点**：**bubble 空泡**、调试难；需调 **micro-batch 数**。

**常组合**：**TP 机内 + PP 跨机 + ZeRO 数据并行** 混成 3D 并行。

---

## 5.4 Pipeline Parallel（PP）与 Sequence Parallel（SP）：具体是啥、和配置约束

### PP：在切什么、数据怎么流

- **切什么**：**网络深度方向** 按 **stage** 切，例如 Stage0 持有层 `0…L₁-1`，Stage1 持有 `L₁…L₂-1`… **每张卡只有一段层的参数**；不是把 **同一个矩阵** 横竖切开（那是 TP）。
- **一次迭代怎么走**：把 **一个优化步里的 batch** 拆成 **`num_micro_batches` 个 micro-batch**。micro-batch 在 Stage0 上前向到边界，得到 **发给下一 stage 的激活**（形状依该层输出，常见仍是 **`[B_micro, S, H]`** 量级），经 **点对点 send/recv（或 NCCL P2P）** 交给 Stage1；Stage0 同时可接下一个 micro-batch，形成 **流水线**。
- **bubble（空泡）**：管道 **刚启动/将结束** 时，部分 GPU **在等上下游**，算力闲置；**micro-batch 个数太少** 时 bubble **占比**高。
- **调度名（面试点到为止）**：**GPipe** 式（先灌满再反传）与 **1F1B**（交错前反向以平衡显存）等；不同框架默认不同，影响 **峰值显存 vs 吞吐**。

**配置约束（口述）**

- **`pp_size`（stage 数）**：与 **总层数 / 每 stage 最少层数**、框架是否支持 **virtual pipeline（交错排层）** 有关；要能对上 **设备拓扑**（连续 rank 常排成一条链）。
- **`global_batch` ↔ micro-batch**：框架里常有 **`global_batch = micro_batch_size × num_micro_batches × data_parallel_degree`**（或再乘其它维）；要满足 **整除关系**，否则最后一段要 **drop/pad**。
- **`num_micro_batches` 调大**：一般 **减 bubble、提吞吐**，但管道里 **同时活着的 micro-batch 变多 → 激活峰值上升**；和 **Gradient Checkpointing** 一起是常见取舍。
- **混维**：PP 常与 **TP、DP、ZeRO** 组合，**总 rank = 各并行维乘积**（具体排布依 `torchrun` / Megatron / DeepSpeed 的 rank 映射），改一个维可能要 **重排 device mesh**。

### SP：在切什么、和 DP / TP 的差别

- **切什么**：在 **同一层的前向里**，把某张 **与 `S` 成正比** 的大激活（典型是 **LayerNorm / Dropout / 残差边** 上的 **`[B, S, H]`**）按 **序列维** 切成 **`[B, S/p, H]`**，**`p` 张卡各持一段 token**（`p` 在 Megatron 系里常与 **TP 组大小一致**，不是任意多加卡）。
- **为啥省显存**：上述算子的 **中间结果规模 ~ `B·S·H`** 随 **`S/p`** 缩小；长序列时 **LN 一带** 常可观。Attention 主体内是否同时省 **`S×S`**，**依赖具体实现**（是否与 Flash、TP 分头混排），面试答 **「减与 S 线性相关的激活与部分通信，不等价于自动消灭 S²」** 更稳。
- **怎么「拼回去」**：凡算子 **必须看全长 `S`** 时，框架在算子前后插入 **`all-gather`（序列维）** 或等价 collective，再 **scatter/split** 回分片；本质是 **用通信换显存**。
- **和 DP 区别**：DP 是 **每卡完整模型 + 不同样本子 batch**；SP 是 **同一样本的一条序列被拆开、多卡协作算一层**，仍属 **模型/算子并行** 思路。
- **和 TP 区别**：TP 主要切 **参数/矩阵乘的列或行**；SP 切 **token 维上的激活布局**。工业训练里 **SP 常依赖 TP group**（例如 Megatron `sequence_parallel=True` 与 TP 同组），不要理解成「单独一种只加卡数的开关」。

**配置约束（口述）**

- **框架开关**：常见是 **在 TP 已配置的前提下** 打开 `sequence_parallel` / 等价选项；**不支持「只有 SP 没有 TP」** 的组合要看对应版本文档。
- **`S` 与 `p`**：有效长度 **`S` 能被 `p` 整除**，或框架用 **pad** 到整除；变长 batch 时注意 **padding 策略** 与 **有效 token mask**。
- **拓扑与带宽**：SP 增加 **序列维 all-gather** 等；与 TP 叠加后仍偏 **机内 NVLink 友好**，弱网多机要慎重。
- **数值与随机性**：Dropout 等若在分片上算，需 **与框架一致的种子 / 分片规则** 才与「不切 SP」对齐；面试提一句 **「依赖框架实现细节」** 即可。

---

## 5.5 Megatron-LM / Megatron-Core：卡数、Global Batch、Micro-batch、TP / PP / DP / SP 怎么对上（防报错）

以下按 **当前 Megatron-LM `megatron/training/arguments.py` 里 `validate_args` 的常见逻辑** 归纳（**版本会演进**，MoE、FSDP、Context Parallel 等开了以后 **总并行度公式会变**；报错时以 **终端 assert 原文 + 官方文档** 为准）。

### 谁由谁推出来（不要「手填 DP」和 world 对不上）

- **`world_size`**：`torchrun --nproc_per_node=... * NNODES` 等与启动一致。
- **`tensor_model_parallel_size`（TP）**、**`pipeline_model_parallel_size`（PP）**、**`context_parallel_size`（CP）**：你显式设；未用 CP 时 **CP=1**。
- **`total_model_size`（占模并行的卡数）**：Megatron-Core 路线里常见  
  **`total_model_size = TP × PP × CP`**。  
  **硬约束**：**`world_size % total_model_size == 0`**，否则直接 assert。
- **`data_parallel_size`（DP）**：**框架推导**  
  **`DP = world_size / total_model_size`**。  
  面试/实操：**不要**再假设「我自己定一个 DP」；**改 TP/PP/CP 或加卡** 都会改 DP。

### Global batch 与 micro-batch（训练步）

- 未显式给 **`global_batch_size`** 时，常见默认：  
  **`global_batch_size = micro_batch_size × DP`**（等价 **每个 DP rank 每步只跑 1 个 micro-batch**）。
- 显式给了 **`global_batch_size`** 时，Megatron-Core 用 **micro-batch 计算器** 推出 **`num_microbatches`**，核心整除关系（与 eval 里 assert 同型）记为：  
  **`global_batch_size % (micro_batch_size × DP) == 0`**  
  记 **`num_microbatches = global_batch_size / (micro_batch_size × DP)`**，须为 **≥1 的整数**。  
  不满足 → **初始化阶段报错**或无法稳定流水。
- **经验**：**PP>1** 时常把 **`num_microbatches` 调大** 减轻 bubble，但仍须满足上式；过大则 **显存峰值**上去。

### PP、层数、交错流水

- **`num_layers`（或等价总层数）** 需能被 **`transformer_pipeline_model_parallel_size`（通常等于 PP）** 等分；开 **virtual / interleaved pipeline** 时还有 **「每层 stage 内再细分」** 的整除 assert（版本里有明文）。
- **PP 与别的并行**：例如 **Hybrid context parallel** 在部分版本里 **不与 PP>1 同开**（代码里会直接 assert）；以当前仓库为准。

### SP（`--sequence-parallel`）与 TP

- Megatron 里 **`tensor_model_parallel_size == 1`** 时，会 **关掉 sequence parallel**（避免数值/路径与 TP=1 不一致），并 **warn**。
- **`tp_comm_overlap`** 类优化常 **要求 `sequence_parallel=True`**（否则 assert）。
- **模型维**：`hidden_size`、`num_attention_heads` 等需满足 **TP 列/行切分**（各版本在 `validate_args` / transformer config 里有一批 assert）；**`num_heads % TP == 0`** 是最常被问到的 **必要条件之一**（具体还看是否 GQA/MQA）。

### Context Parallel（CP）与序列长

- 若 **`context_parallel_size > 1`**，Megatron 对 **`seq_length`** 有常见约束：  
  **`seq_length % (2 × context_parallel_size) == 0`**（assert 原文即此意）。
- **CP 乘进 `total_model_size`** 后，**DP 会变小**；**global batch 整除式里的 DP** 用的是 **推导后的 DP**。

### 实操怎样少报错

1. 先定 **`TP×PP×(CP)`**，保证 **`world_size` 能整除**。  
2. 用 **`DP = world / (TP×PP×CP)`** 看数据并行还有几路。  
3. 选 **`micro_batch_size`**，再定 **`global_batch_size`**，检查 **`GBS % (micro×DP) == 0`**。  
4. 开 **SP** 前先保证 **TP>1**；开 **CP** 再对 **`seq_length`** 做 **2×CP** 整除。  
5. **MoE / EP / Megatron FSDP** 等会再引入 **`expert_model_parallel_size` 等因子** 改写 `total_model_size`，需 **单独读该版本 README / arguments**。

---

## 6. Gradient Checkpointing（激活重算）

**原理**：前向时不存所有中间激活，反向需要时再 **重算一遍前向** 到该点。

**代价**：**算力** 这里指 **GPU 上多做的前向计算（额外 FLOPs / 更长的有效前向时间）**——反向缺激活时要把 **一段前向再算一遍**，同一次迭代里 **部分层的 matmul/激活函数等会执行两次**，总 **wall-clock** 常 **慢约 20%～50%**（依 checkpoint 粒度）；**不是**「参数量变大」或「优化器多算一遍」那种含义。显存侧 **显著降激活峰值**。

**哪些层划算**：大块高维激活（Transformer block 内 attention、MLP 中间层）；**不要无脑全 checkpoint** 以免算力爆炸。

---

## 7. FlashAttention：为什么省显存、online softmax 是什么、IO-aware

**标准实现问题**：materialize 完整 `(L,L)` attention 矩阵读写显存带宽大。

**FlashAttention**：在 **片上 SRAM**（SM 附近的高速缓存）里 **分块** 算 attention，融合 softmax 与乘 V，**减少 HBM 往返**；用 **online softmax**（分块维护 running max 与 running sum）在 **不存完整 scores 矩阵** 的情况下得到等价 softmax 结果。

**HBM**：**High Bandwidth Memory**，焊在 GPU 封装上的 **大容量显存**（相对 **SRAM** 容量大、**延迟高**）。Attention 若把大块 `S×S` 矩阵在 **HBM** 里反复读写，算子再快也 **吃带宽**；FlashAttention 尽量在 **SRAM** 上完成分块计算，少把中间结果 **写回/读回 HBM**。

**IO-aware**：优化目标不仅是 FLOPs，而是 **内存层次**（HBM vs on-chip），故长序列上常大幅提速。

---

## 8. 训练变慢 / GPU 利用率低（与「用了几张卡」类问题衔接）

**排查顺序**（可口述）：

1. `nvidia-smi dmon` / DCGM 看 **SM 占用、PCIe、NVLink**。
2. 若 SM 低、数据加载线程 idle：**DataLoader** `num_workers`、**存储 I/O**、**tokenize 是否在主进程**。
3. 若通信占比高：**batch/global batch**、梯度累积、ZeRO 阶段、是否跨机慢网。
4. **同步点**：日志、eval、barrier、Python GIL。

**显存碎片化**：长期分配释放导致 **大块连续显存不足**；可重启进程、用 **内存池/缓存分配器**、减少频繁 `empty_cache` 迷信（有时适得其反）。

---

## 小结

- 显存要会 **拆项估算**：**状态（参/梯/优化器）** + **激活（与 `B,S,H,L` 及是否 S² attention 强相关）** + **KV（推理）** + **框架与并行 buffer 尖峰**。
- ZeRO 是 **用通信换显存** 的连续谱；**多卡主要分摊状态**，**激活峰值仍按单卡 micro-batch 看**；ZeRO-3 最省参数驻留但 **gather 尖峰与通信**更明显。
- FlashAttention 的核心是 **分块 + online softmax + 减少 HBM IO**，不是「换了个 softmax 公式」。
