# 04｜显存估算、数据/模型并行、ZeRO、FlashAttention 与训练排障

**建议阅读顺序（本篇内部）**：先 **单卡显存由哪些项组成**（否则后面「并行省了什么」说不清）→ **数据并行 / DDP**（最常见）→ **模型并行：TP、PP、SP**（为何会 bubble、为何要流水）→ **和 ZeRO 的关系**（ZeRO 仍是「数据并行 + 分片状态」）→ **Megatron 类配置**（进阶）→ **Checkpoint / FlashAttention** → **变慢与碎片**。

术语：英文 **pipeline** 本文统一叫 **「流水线并行」** 或保留 **pipeline parallel**，不译成含混的「管道法」。**HBM** 指 GPU 上容量大、相对慢的那层显存（相对片上 SRAM）。

---

## 1. 单卡训练：显存大致由什么组成

一块卡上，训练一步通常要扛：

1. **权重**（BF16/FP16 常 2 byte/参数）
2. **梯度**（常与权重同量级、同 dtype 策略）
3. **优化器状态**（Adam 的 m、v 常见 **FP32**，约 **8 byte/参数** 量级理解两矩）
4. **激活**（与 batch、序列长、隐藏维、层数有关；**长序列 + 标准 Attention 物化 L×L 分数矩阵** 时常是 OOM 首因）
5. **临时区**：通信 buffer、cuBLAS workspace、分配器碎片等

**推理**：主要是 **权重 + KV Cache + 少量激活**。

### 面试怎么答

「训练 OOM 先分清是状态（参/梯/优化器）还是激活；长文先看注意力有没有爆 L²。」

---

## 2. 粗算与关键比例（数量级）

**混合精度 + Adam 常见口算**：参数量 P，

```
约 P × (2 + 2 + 4 + 4) = 12P 字节
```

含义：权重、梯度各按半精度 **2**；m、v 按 **FP32 各 4**（与「优化器状态要稳」的工程习惯一致，具体实现略有出入）。

**7B**：`7e9×12 ≈ 84 GB` **只是状态量级**，真实还要加 **激活、通信尖峰**。全量训练常说 **百 GB 级** 才宽裕。

**推理 14B BF16**：权重约 `14e9×2 ≈ 28 GB`，再加 KV 与框架，**40GB+** 更现实。

**KV Cache（与 02 一致）**：每层约 `2 × batch × seq × num_kv_heads × head_dim × dtype_bytes`，乘层数。MQA/GQA/MLA 通过 **少算少存 K/V 头** 或 **压缩 KV** 省钱。

### 2.1 激活：为何与 L、L² 有关

- 与 **B（batch）·S（序列）·H（隐藏）·层数** 成正比的 **张量链** 总要占一份。  
- 若标准实现把注意力 **分数矩阵整块物化**（materialize，即真的分配出整块显存）为 `**B × heads × S × S`**，则 **S 翻倍约 ×4** 这一项。  
- **FlashAttention** 用分块在 **片上 SRAM** 多算少写 **HBM**，降低 **峰值与带宽压力**；不是把渐近复杂度 magically 变成 O(S)。

### 2.2 多卡之后：激活为何常常「不除以卡数」

**数据并行 + ZeRO-1/2**：每张卡仍跑 **完整前向/反向图**，micro-batch 仍是本卡的 **B、S** → **激活按本卡 batch 算**，**不会**因为世界规模 N 张卡就自动 **÷N**。

**ZeRO-3 / FSDP**：参数分片了，但算子前常要 **短时 all-gather 整层权重**，出现 **常驻分片 + 通信尖峰**。

**TP**：层内切分常伴随 **通信 buffer / 临时拼张量**。

### 面试怎么答

「ZeRO 主要分的是优化器/梯度/参数状态；激活还是跟本卡 micro-batch 和是否物化 L² 走。」

---

## 3. 并行在解决什么问题（一句话地图）

- **装不下整模**：**模型并行**（TP 切层内大矩阵；PP 按层切段；SP 切序列维激活配合 TP）。  
- **装得下但想加速**：**数据并行**（多卡各吃不同数据，梯度对齐）。  
- **状态太大**：**ZeRO / FSDP**（在数据并行之上 **分片优化器/梯度/参数**）。

---

## 4. 数据并行与 DDP（默认该会）

**数据并行**：每卡 **一份完整模型副本**，各卡输入 **不同子 batch**；反向得到本地梯度后做 **all-reduce 求平均（或先求和再除）**，保证各卡 **下一步参数一致**。

**DP（DataParallel，老接口）**：常 **单进程多 GPU**，梯度 **先 gather 到主卡** 再广播 → **Python GIL + 主卡瓶颈**。

**DDP（DistributedDataParallel）**：**每 GPU 一进程**，`torchrun` 起，`DistributedSampler` 保证样本不重复，`backward` 后 **梯度自动 all-reduce**。这是 **正式训练默认**。

### 面试怎么答

「DDP 是多进程一对一绑卡；DP 是单进程多卡老路，别在生产用。」

---

## 5. 模型并行：TP、PP、SP

### 5.1 TP（张量并行，tensor parallel）

把 **单层里的大矩阵乘** 按列/行切到多卡；前向/反向需要 **频繁 all-gather / reduce-scatter**（依算子）。**机内 NVLink** 友好；跨慢网很痛苦。

### 5.2 PP（流水线并行，pipeline parallel）

按 **层** 切段：卡0算前几层，卡1算后几层……一个 batch 切成 **多个 micro-batch** 在时间上 **流水** 喂进去，提高利用率。

**bubble（流水线空泡）**：管道刚启动/将结束时，有的卡 **在等上下游**，算力空转；**micro-batch 太少** 时 bubble **占比**高。调度有 **1F1B** 等（知道「为了减少空泡、平衡显存」即可）。

### 5.3 SP（序列并行，sequence parallel）

把 **LayerNorm / Dropout / 残差** 一带与 **S 成正比** 的大激活，按 **序列维 S** 切开多卡存；需要看全长时再 **all-gather**。常与 **TP 同组** 使用，不是「单独多加卡就能开」的万能开关。

### 面试怎么答

「TP 切矩阵；PP 切层用流水填 GPU；SP 切序列维省 LN 一带激活，一般要配合 TP。」

---

## 6. 数据并行 vs 模型并行：数据流与梯度（对照记）


|      | 数据并行                        | 模型并行（TP/PP 等）                  |
| ---- | --------------------------- | ------------------------------ |
| 每卡模型 | **完整副本**                    | **一块子结构**                      |
| 数据   | **不同子 batch**               | **同一条样本在卡间传激活（PP）或协作算（TP/SP）** |
| 梯度   | **对本地数据算完 → all-reduce 对齐** | **在分片边界用 collective 拼出完整梯度语义** |


题面若写「模型冰箱」，一般是把 **模型并行** 打错字。

---

## 7. ZeRO-1 / 2 / 3 与 FSDP

**共同思想**：数据并行团队里，**优化器状态 / 梯度 / 参数** 不必每卡全存，改 **分片 + 需要时 all-gather / reduce-scatter**。


| 阶段     | 多分片了什么 | 直觉                |
| ------ | ------ | ----------------- |
| ZeRO-1 | 优化器状态  | 省一大块              |
| ZeRO-2 | + 梯度   | 再省                |
| ZeRO-3 | + 参数   | 单卡常驻最少，但 **通信最多** |


**FSDP**：PyTorch 原生 **分片 DDP**，思想接近 ZeRO，生态集成好；**DeepSpeed** 更像「训练框架 + 一堆开关（ZeRO、offload、pipeline…）」。

### 面试怎么答

「ZeRO 是用通信换显存；越激进越慢；激活不见得随 ZeRO 变少。」

---

## 8. Megatron 三维并行：TP、PP、CP、DP 怎么「乘满 world」

先把 **四种并行扮演的角色** 钉死（与第 5 节一致，这里换成 **算卡数** 视角）：

- **TP（Tensor Parallel）**：**同一层** 里把大矩阵切到 **TP 张卡** 上协作算；这些卡 **同一步、同一条 micro-batch**，**不**各自持有一份完整层权重。  
- **PP（Pipeline Parallel）**：把 **层堆** 沿深度切成 **PP 段**，每段落在不同 GPU；micro-batch 像 **流水线工件** 在时间维上流过各段。  
- **CP（Context Parallel，可选）**：极长序列时把 **单条样本的上下文维** 再切开；不用时 **CP=1**。  
- **DP（Data Parallel）**：**一整坨**「TP×PP×CP 绑在一起的计算组」是 **一条模型副本**；**DP 份** 这样的副本，每份吃 **不同数据**，梯度做 **数据并行 all-reduce**。

**硬等式（念这个比背报错字符串稳）**：

```
world_size = TP × PP × CP × DP        （Dense 常见写法）
DP = world_size / (TP × PP × CP)
```

**MoE + EP**：是否写成 `world = TP×PP×CP×EP×DP` 一类，**随框架与版本**；见 **§8.3**。

- `**TP×PP×CP**`：一个 **完整前向** 在模型维上占多少张卡（常称 **model-parallel width**）。  
- `**DP`**：同样的模型逻辑 **并行复制多少份** 去吃不同 batch。  
- 手填时 **先选 TP、PP、CP**，**DP 只能由上式推出**；若除不尽，**world 拆不干净**，启动即 assert。

**SP（Sequence Parallel）**：不是上式里的独立因子；开在 **TP 组内部**，省 LN/残差一带与序列成比例的激活；常见 **要求 TP>1**，否则无意义或自动关。

**PP 额外整除**：总 **Transformer 层数** 要能被 **PP 的 stage 数** 整除；**virtual pipeline / interleaved** 再叠倍数——**以当前版本 `arguments.py` 为准**。

### 8.1 Global batch、micro-batch、梯度累积

Megatron 类里 **global_batch_size（一次优化器步等效见过多少条样本）** 口述核心：

```
global_batch_size ≈ micro_batch_size × DP × gradient_accumulation_steps
```

（各 repo 变量名可能叫 `global_batch`、`num_micro_batches` 等，面试说 **结构** 即可。）

- `**micro_batch_size`**：单次前向的 batch 维大小；顶 **激活**，也影响 **算子效率**。  
- `**DP`**：多少路独立数据并行。  
- `**gradient_accumulation_steps`**：多步前向-反向 **累加梯度** 再 `optimizer.step`，**等效放大 global batch** 而不按同比例放大单步激活峰值。

**易错点**：ZeRO / TP **不会自动帮你改 global_batch 定义**；global_batch 是 **数据调度语义**，靠 **sampler + 累积步** 与并行度配平。

### 8.2 微型数值例子

`world_size = 32`，`TP=2, PP=4, CP=1` → 一组模型并行占 **8** 张卡 → `**DP = 4`**。**  
`**micro_batch_size=1`，`gradient_accumulation_steps=8` → `**global_batch_size ≈ 1×4×8 = 32`**。  
把 `**TP=4, PP=2**` 仍得 **8 张一组**，**DP 仍为 4**——说明 **先定 TP×PP×CP 再除 world** 才是硬约束。

### 8.3 Expert Parallel（EP）：干什么、怎么设、常见约束

**干什么**：MoE 每层有 **E 个 expert**，每个 expert 一套 FFN 权重。**EP（Expert Parallel）** 把 **expert 权重切到多张卡** 上存，每张卡只驻 **一部分 expert**。  
前向时，token 被 router 指到某个 expert，若该 expert **不在本卡**，就要通过 **all-to-all / 等价 collective** 把 **token 表示（或梯度）送到持权重的那张卡** 上算，再传回来——所以 EP **省「单卡 expert 权重显存」**，但 **吃机间/机内通信**。

**怎么设（口述口径）**：

1. 先定 **MoE 层数、总 expert 数 `E`、每 token top-k**。  
2. 选 **`expert_model_parallel_size`（EP）**：希望 **每张卡挂几个 expert 的权重**。  
3. **硬约束（Megatron 系里非常常见）**：**`num_experts % EP == 0`**——否则 expert 无法均匀分片，启动会 assert。  
4. **world 与并行维**：带 MoE 时，**是否把 EP 乘进「模型并行宽度」、DP 怎么推**，**随 Megatron / NeMo / Bridge 版本与是否 MoE 而异**；面试答 **「EP 与 TP/PP/CP/DP 同属并行拓扑，必须满足 `world` 被各维乘积整除；细节以当前仓库 `parallel_state.py` + `validate_args` 为准」** 比背一个可能过期的连乘式更稳。  
5. **物理拓扑**：EP 常伴随 **大量 token↔expert 路由通信**；**机内 NVLink** 远好于跨机慢网。和 **TP** 同用时，要留意 **CUDA stream / max connections** 类环境变量冲突（部分版本文档会警告）。

**和 ZeRO 的关系**：ZeRO 主要切 **优化器状态/梯度/参数副本**；**EP 切的是 expert 这一块的「模型分片语义」**。大 MoE 训练里常见 **EP +（ZeRO 或 FSDP）+ TP** 混用，**调参面大**，先跑通小规模再扩。

### 面试怎么答

「world = TP×PP×CP×DP；DP 是除出来的；global_batch 用 micro×DP×累积步配；SP 不是独立一维。」

MoE 训练再加一句：**「EP 把 expert 权重分卡存，省单卡权重大头；要付 all-to-all；`E` 必须能整除 EP；拓扑以仓库 assert 为准。」**

---

## 再讲一遍

总卡数 (World Size) = TP × PP × DP
GBS(梯度累加步数) = MBS × DP × GAS

注意力头数整除 TP：Num_Attention_Heads % TP == 0。
（如果有 GQA/MQA，KV 头数也必须能被 TP 整除）

隐藏层维度整除 TP：Hidden_Size % TP == 0

模型总层数整除 PP：Num_Layers % PP == 0
（PP 要求每个 pipeline stage 分配到相同数量的 Transformer 层）。

(如果开启 SP) 序列长度整除 TP：Seq_Length % TP == 0

能跑起来不报错只是第一步，要跑得快（高 MFU），还需要遵守以下潜规则：

#### 1. Pipeline 气泡约束：GAS 必须足够大

- **制约关系**：流水线并行（PP）是有“气泡”（Bubble）的。为了掩盖气泡，必须有足够多的 Microbatch（即 GAS）在流水线里跑。

**经验公式**：**GAS 必须** ≥ **PP**。

**最佳实践**：为了保持高效率，通常要求 **GAS** ≥ **4 × PP**，最好是 **8 × PP**。

- *防错提醒：* 如果算出来的 GAS < PP，Megatron 可能会抛出警告，且你的 GPU 利用率会极低（都在互相等）。

#### 2. TP（张量并行）的物理边界：不出节点

- **制约关系**：TP 需要极大量的通信（All-Reduce）。因此，**TP 组内的卡必须有极高的带宽（NVLink）**。
- **最佳实践**：**TP 的大小永远不要超过单台机器的 GPU 数量！**（目前主流是单机 8 卡，所以 TP 最大设为 8。跨机做 TP 速度会慢到让人怀疑人生）。

#### 3. SP（序列并行）的开启条件

- **制约关系**：SP 本质上是对 TP 的显存优化（把 LayerNorm 和 Dropout 在序列维度上切分开）。
- **前提条件**：**必须 TP > 1** 才能开启 SP（通常设 --sequence-parallel）。如果 TP=1 开启 SP 会报错。

### 来一道应用题🤡

假设你要在 **64张 A100 (8机8卡)** 上训练一个 **Llama-2-7B**。  
模型参数：32 层 (Layers)，32 个头 (Heads)，算法要求 GBS = 1024。

```
**第一步：确定 TP (Tensor Parallel)**

- TP 最好不出节点（<=8）。看模型参数，32 头可以被 2, 4, 8 整除。
- 假设模型不算太大，我们设 **TP = 4**。

**第二步：确定 PP (Pipeline Parallel)**

- 总层数是 32。PP 可以是 1, 2, 4, 8。
- 假设为了装下模型（或者分摊显存），我们设 **PP = 4**。

**第三步：算出 DP (Data Parallel)**

- 卡数公式：64 = TP(4) × PP(4) × DP
- 算出 **DP = 4**。

**第四步：确定 MBS (Micro Batch Size)**

- MBS 决定了单卡的显存占用。你需要慢慢调大 MBS，直到快要 OOM（爆显存）为止。
- 假设经过测试，**MBS = 4** 时显存利用率最好。

**第五步：验证 GBS 和 GAS**

- Batch 公式：GAS = GBS / (MBS × DP)
- 代入数据：GAS = 1024 / (4 × 4) = 1024 / 16 = **64**
- 检查能否整除？64 是整数，**不报错。**
- 检查流水线效率？GAS (64)≥ 8 × PP (4)，**效率极高。**



**最终安全参数配置：**  
--tensor-model-parallel-size 4  
--pipeline-model-parallel-size 4  
--micro-batch-size 4  
--global-batch-size 1024  
*(DP=4 会由 Megatron 自动推导，不用写)*

```

---

## 9. Gradient Checkpointing 与 FlashAttention

**Checkpointing**：前向少存激活，反向缺了再 **重算一段前向** → **省显存、多花算力**（wall-clock 常慢 **二到五成** 量级，依粒度）。

**FlashAttention**：在 **SRAM** 上分块做 attention，**少把大中间结果写回 HBM**；常配合 **online softmax**。目标是 **带宽与峰值**，不是换 softmax 公式。

---

## 10. 训练变慢、利用率低、碎片化

**排查顺序**：先看 **SM 占用** 还是 **PCIe/NVLink**；再看 **DataLoader**、是否在主线程做重 tokenizer；再看 **通信占比**（小 batch + ZeRO-3）；最后看 **barrier、同步日志、过大 eval**。

**碎片化**：`nvidia-smi` 显示还有空闲却 `malloc` 失败 → **重启进程**、避免无意义频繁 `empty_cache`、注意 bucket 设置。

---

## 11. Megatron 断言速记（与第 8 节同型，考场 30 秒版）

- `**world_size = TP × PP × CP × DP`**；`**DP` 只能推、不要手填打架**。  
- `**global_batch_size`** 与 `**micro_batch_size × DP × gradient_accumulation_steps`** 对齐（具体名字以版本 README 为准）。  
- `**sequence_parallel`** 常与 `**TP>1**` 绑定。  
- **PP / interleaved / MoE-EP** 会再叠整除条件——**以该版本 `arguments.py` 的 assert 为准**（MoE 常见：**`num_experts % EP == 0`**）。

---

## 本篇小结

- **账单**：状态（12P 量级口算）+ 激活（看 **B,S,H** 与是否 **L²**）+ KV（推理）。  
- **并行**：DDP 扩数据；TP/PP 扩模型；ZeRO 分状态。  
- **Flash**：减 **HBM** 往返与峰值。  
- **Megatron**：`**world = TP×PP×CP×DP`**（Dense）；**MoE 时加 EP 等维** 见 **§8.3**；**global_batch ≈ micro×DP×累积步**；SP 不是独立一维。

