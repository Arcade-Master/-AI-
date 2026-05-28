# 面试笔记

正文 Markdown 在 [`面试笔记/`](./面试笔记/) 目录；**手撕代码题** 在 [`手撕代码/`](./手撕代码/)。写作规约见 [`面试笔记/00-写作与答疑规约.md`](./面试笔记/00-写作与答疑规约.md)。

## 文件索引

| 文件 | 内容 |
|------|------|
| [01](./面试笔记/01-损失函数正则与手撕基础.md) | 交叉熵/KL、MSE、sigmoid/softmax、L1/L2、axis、DFS/BFS、第 k 大 |
| [02](./面试笔记/02-注意力Transformer与Mask.md) | Attention、**RoPE**、mask、手撕（推理见 08） |
| [03](./面试笔记/03-优化器浮点与数值稳定.md) | Adam/AdamW、BF16/FP16、溢出、loss scaling |
| [04](./面试笔记/04-显存估算训练并行与DeepSpeed.md) | 训练显存、DDP/TP/PP/SP、ZeRO、Megatron、**FlashAttention 原理** |
| [05](./面试笔记/05-微调SFT对齐RL与遗忘.md) | SFT、RLHF/DPO/GRPO/GSPO 选型、SFT→RL、奖励形态概要、LoRA、遗忘 |
| [05-1](./面试笔记/05-1-RL工程细节与GRPO动态采样.md) | GAE/Value、稀疏 reward、buffer、异步 Judge、off-policy、动态采样、loss 平均 |
| [06](./面试笔记/06-DeepSeek-R1-MLA与MoE.md) | MoE、MLA、R1、冷启动 |
| [07](./面试笔记/07-长上下文外推量化与评测.md) | 长窗外推、量化、评测（RoPE 见 02） |
| [08](./面试笔记/08-推理部署Prefill与vLLM.md) | **Prefill、Decode、KV Cache、vLLM**（推理专篇） |
| [09](./面试笔记/09-Prompt工程数据与幻觉.md) | LangGraph/Harness、Prompt、幻觉（Agent 模式见 15） |
| [10](./面试笔记/10-工程排障与分布式实战.md) | NCCL、OOM、死锁、利用率、**AutoResearch 流程** |
| [11](./面试笔记/11-多模态对齐QFormer-ViT与RoPE.md) | 对齐层、Q-Former、ViT 训练、RoPE、PPL |
| [13](./面试笔记/13-RAG向量化检索与Agent系统设计.md) | PDF、检索、HyDE、记忆、RAG 三问题 |
| [15](./面试笔记/15-Agent架构上下文与工具调用.md) | ReAct/Plan&Execute/Reflection、Tool、**强制 JSON**、Agentic RL |

---

## 与题库文件的对应（简表）

若本地仓库根目录还有 `面试答题.md`、`面试题2` 等，可按下表对照章节（未纳入远程仓库时仅作本地索引）。

| 来源 | 主要文档 |
|------|----------|
| `面试答题.md` 前部基础 + 手撕 | 01、02、05、11、01 |
| `面试答题.md` 项目中后段 | 13、15（SGLang/asyncio 类 infra 题可略） |
| `面试题2` | 02～10、06、07、08 等 |
