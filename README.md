# 面试笔记

正文 Markdown 在 [`面试笔记/`](./面试笔记/) 目录；**手撕代码题** 单独放在 [`手撕代码/`](./手撕代码/)（与笔记分离，便于贴题与维护实现）。下面为 **阅读顺序** 与 **文件索引**（GitHub 打开本仓库首页即可从这里点进各篇）。

本套笔记按 **「先基础、后系统」** 拆开，避免读到后面才发现前面某节本该先读。公式仍用纯文本（`Σ`、`sqrt`），不依赖 LaTeX。

---

## 建议阅读顺序（拓扑）

1. **`01`**：损失、正则、axis、DFS/BFS、第 k 大 —— 后面所有「算 loss、看 shape」都依赖它。
2. **`02`**：Attention 全流程（Q/K/V → mask → KV → 复杂度）— 读懂后再看 04 里与激活、`S×S` 相关的显存。
3. **`03`**：Adam / 浮点 / 溢出 —— 与混合精度训练强相关。
4. **`04`**：单卡显存粗算 → **数据并行（DDP）** → **模型并行（TP/PP/SP）** → **ZeRO / FSDP** → **Megatron：`world=TP×PP×CP×DP`、global_batch≈micro×DP×累积** → Checkpoint / Flash / 变慢排查。
5. **`05`**：SFT（含 **teacher forcing / exposure bias**）、**RLHF 数据与 RM**、**DPO 成对 loss**、**GRPO 与 12 衔接**、**RLAIF 数据流**、LoRA、遗忘；**`12`**：GAE、PPO clip、GRPO 组内优势（在 05 之后读更顺）。
6. **`06`**：MoE 与 MLA 放前，**R1/冷启动** 放后（R1 常依赖「为何用 MoE、如何省 KV」的直觉）。
7. **`07`**：位置编码、长文、量化、评测。
8. **`08` / `14`**：推理侧 prefill/decode、vLLM、SGLang、JSON、asyncio。
9. **`09` / `13` / `15`**：`09` 含 LangChain/LangGraph/Harness 长笔记 + Prompt/幻觉；`13` RAG；`15` Agent —— 可独立阅读。

---

## 文件索引

| 文件 | 内容 |
|------|------|
| [01](./面试笔记/01-损失函数正则与手撕基础.md) | 交叉熵/KL、MSE、sigmoid/softmax、L1/L2、axis、DFS/BFS、第 k 大 |
| [02](./面试笔记/02-注意力Transformer与Mask.md) | Attention、mask、KV、**Prefill/Decode 分阶段**、与 RNN 对比、复杂度、**单头+多头手撕** |
| [03](./面试笔记/03-优化器浮点与数值稳定.md) | Adam/AdamW、BF16/FP16、溢出、loss scaling |
| [04](./面试笔记/04-显存估算训练并行与DeepSpeed.md) | 显存粗算、DDP/TP/PP/SP、ZeRO、**Megatron 并行度乘法与 batch**、Flash、变慢 |
| [05](./面试笔记/05-微调SFT对齐RL与遗忘.md) | SFT、**RLHF/DPO/GRPO 原理与数据**、**RLAIF 介入点**、LoRA、遗忘 |
| [06](./面试笔记/06-DeepSeek-R1-MLA与MoE.md) | MoE、MLA、R1、冷启动 |
| [07](./面试笔记/07-长上下文位置编码与量化评测.md) | RoPE、外推、量化、评测 |
| [08](./面试笔记/08-推理部署Prefill与vLLM.md) | Prefill/Decode、vLLM、PagedAttention |
| [09](./面试笔记/09-Prompt工程数据与幻觉.md) | LangChain/LangGraph/Harness + Prompt、数据、幻觉（整合篇） |
| [10](./面试笔记/10-工程排障与分布式实战.md) | NCCL、OOM、死锁、利用率 |
| [11](./面试笔记/11-多模态对齐QFormer-ViT与RoPE.md) | 对齐层、Q-Former、ViT 训练、RoPE、PPL |
| [12](./面试笔记/12-RL进阶REINFORCE-PPO-GRPO与价值函数.md) | REINFORCE、**GAE**、**PPO clip 目标**、**GRPO 组内优势与 loss**、Value |
| [13](./面试笔记/13-RAG向量化检索与Agent系统设计.md) | PDF、检索、HyDE、记忆、RAG 三问题 |
| [14](./面试笔记/14-推理框架SGLang与API工程.md) | SGLang、asyncio、进程线程、强制 JSON |
| [15](./面试笔记/15-Agent架构上下文与工具调用.md) | Agent 上下文、架构、Tool |

---

## 与题库文件的对应（简表）

若本地仓库根目录还有 `面试答题.md`、`面试题2` 等，可按下表对照章节（未纳入远程仓库时仅作本地索引）。

| 来源 | 主要文档 |
|------|----------|
| `面试答题.md` 前部基础 + 手撕 | 01、02、05、11、12、01 |
| `面试答题.md` 项目中后段 | 13、14、15 |
| `面试题2` | 02～10、06、07、08 等 |

若某篇仍觉跳跃，优先看该篇开头的 **「阅读顺序」** 小节。
