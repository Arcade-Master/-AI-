# 08｜推理两阶段：Prefill / Decode 与 vLLM

**建议阅读顺序（本篇内部）**：先分清 **Prefill 与 Decode 在算什么、谁吃带宽**（否则看不懂 vLLM 在优化什么）→ 再讲 **vLLM 的 PagedAttention 与 continuous batching** → 最后一句 **和 FlashAttention 的分工**（细节算子见 `04`）。

---

## 1. Prefill 与 Decode：延迟里谁是大头

**Prefill（提示阶段）**：把整段 prompt 一次性送进模型。矩阵大、并行度高，往往 **算力更吃满**（口语里仍有人会说 compute-bound）。

**Decode（逐 token 生成）**：每步只多一个新 token，矩阵小，但要 **反复读整模权重 + 读历史 KV**。这一步常常 **显存带宽受限**（memory-bandwidth-bound）：GPU 算力空着，卡在 **读写显存**。

**总延迟直觉**：Prefill **一步重**；Decode **步数多**，总时间里 decode **常常更长**。

### 1.1 Chunked prefill 与 speculative decode（各一句原理）

**Chunked prefill**：极长 prompt 时把 prefill **按段切开** 多次前向，避免 **单次激活峰值** 顶满显存；代价是 **更多 kernel 次数**，要权衡。

**Speculative decode（投机解码）**：用小 **草稿模型** 先 cheap 地多猜几步，大模型 **并行验证** 哪些前缀可接受；接受则 **一步吃掉多个 token**，在 **带宽受限 decode** 下换 **少量额外算力** 换 **有效 tokens/s**。失败则回退单步，**不改变目标分布**（依具体算法如 Medusa/Lookahead 细节而定，面试说到「草稿+验证」即可）。

### 面试怎么答

「Prefill 像一次大矩阵乘；decode 像很多小步，每步都要搬权重和 KV，所以常卡在带宽。」

---

## 2. vLLM 解决的是「服务」问题，不是换模型

线上多用户、**序列长短不一** 时：

- 静态 padding → **算力和显存都浪费**；  
- KV 若按最大长度 **连续预留** → **碎片与浪费**。

**PagedAttention**：借鉴操作系统 **分页**：逻辑上的 KV 序列映射到 **不连续的物理块**，按需申请/回收，提高 **batch 能塞下的条数**。

**Continuous batching（连续批处理）**：某条先 **EOS** 结束就立刻腾槽，新请求可以 **插进正在跑的时间步**，提高 **吞吐（tokens/s）**。

**batch 变大为何吞吐先升后降**：一开始 **摊薄了每次调 kernel 的固定开销**；大到带宽或显存顶满后 **延迟和 OOM** 上来。

### 面试怎么答

「vLLM = 把 KV 当分页管 + 动态拼 batch 提高吞吐；不是改 Transformer 公式。」

---

## 3. 和 FlashAttention 的关系

**FlashAttention**：算子层 **少写 HBM、分块 softmax**（见 `04`）。  
**vLLM**：调度层 **KV 存哪、batch 怎么拼**。  
二者 **叠在一起** 很常见。

---

## 本篇小结

- 先 **Prefill/Decode** 再谈 **vLLM**。  
- 服务优化 = **显存布局 + 批调度**，和「模型聪明度」无关。  
- 长 prefill 看 **chunk**；decode 吞吐可看 **投机验证**。
