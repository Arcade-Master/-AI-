# 08｜推理：Prefill、Decode、KV Cache 与 vLLM serving

**本篇讲清推理侧一条链**：自回归 **两阶段在算什么** → **KV Cache 存什么、为何只存 K/V** → **多用户 serving（vLLM）** → **长 prompt / 投机 decode 等工程手段**。  
注意力公式与训练 mask 见 **`02`**；FlashAttention 算子原理见 **`04` §9**；KV 显存粗算见 **`04` §2**。

---

## 1. Prefill 与 Decode

**场景**：推理时先输入整段 **prompt**，再 **逐 token** 生成续写。工程上拆成两阶段：

- **Prefill**：处理 **已知前缀**（整段 prompt）
- **Decode**：每步只来 **一个新 token**（自回归步进）

### 1.1 Prefill（提示 / 填充阶段）

**在算什么**：对 prompt 全部位置（长度 `L_prompt`）做 **完整前向**。每层 self-attention 里 Q、K、V 的序列维都是整段 prompt，注意力一侧是 **`L_prompt` 量级** 的大矩阵乘，**并行度高**。

**瓶颈**：`L_prompt` 大时偏 **算力受限**（大 GEMM，SM 较满）；同时 **一次性写入** 较长 KV，显存跳一截。

### 1.2 Decode（解码 / 自回归步进）

**在算什么**：第 `t` 步通常只有 **1 个新 token** 的 Q，但要 **attend 到位置 1…t 的全部 K/V**。算力变成：**读历史 KV + 读整模权重 + 算一行 attention + 写新 KV**。

**瓶颈**：矩阵「薄」，GPU 算不饱，时间多在 **HBM 带宽** → 常 **显存带宽受限**。

### 1.3 对照

| | Prefill | Decode |
|---|---------|--------|
| 输入 | 整段 prompt | 每步通常 1 个新 token |
| Q/K/V 规模 | 序列维 ≈ `L_prompt` | Q 常 1 行；K/V 用 cache 全长 |
| 常见瓶颈 | 算力（大 matmul） | 带宽（读 KV + 读权重） |
| 总延迟 | 一步重 | 步数多，**总时间 decode 常更长** |

### 面试口述

「Prefill 是大矩阵乘、吃算力；decode 每步薄但要反复读权重和历史 KV，吃带宽。」

---

## 2. KV Cache：机制、形状、与两阶段衔接

### 2.1 为什么需要 KV Cache

若 decode 每步 **重算** 从 1 到 t 的 attention，复杂度约 **O(t²)** 每层，生成 n 个 token 会 **O(n³)** 量级，不可用。

**做法**：Prefill 时把各层各头的 **K、V 写入 cache**；Decode 每步 **只算新 token 的 Q、K、V**，**历史 K/V 从 cache 读**，只 **追加** 新 K/V。

### 2.2 为什么只存 K、V，不存 Q

Decode 第 `t` 步：**当前 Q** 与 **位置 1…t 的 K** 做点积得权重，再对 **同长度 V** 加权求和。  
**过去的 K、V** 后续步还要反复读；**过去位置的 Q** 不再当未来步的 query → **不必存 Q**。

### 2.3 与 Prefill / Decode 的衔接

1. **Prefill 结束**：cache 里已有 prompt 各层各头的 K、V。  
2. **Decode 每步**：新 token 算 Q、K、V；**K、V 追加到 cache 末尾**；**Q 只用于本步输出**。

### 2.4 显存形状（粗算）

每层大致两份：

```
K, V 各: [batch, num_kv_heads, seq_len, head_dim]
```

乘 **层数**；`seq_len` 随 decode **线性增长**。MQA/GQA/MLA 通过 **少头数或压缩 KV** 省显存（见 **`06`**）。  
训练时 activation 的 L² 与 **FlashAttention** 见 **`04`**；本篇只管 **推理 cache**。

### 面试口述

「KV Cache 让 decode 不重算历史；只存 K、V 因为未来步只需要历史 K/V 被当前 Q 查询。」

---

## 3. vLLM：多用户 serving 在优化什么

vLLM **不改 Transformer 公式**，解决 **线上多请求、长短不一** 时的 **KV 布局与 batch 调度**。

**静态 padding**：短序列也按 max len 算 → 算力、显存浪费。  
**连续大块预留 KV**：长度不一 → **碎片、利用率低**。

### 3.1 PagedAttention

借鉴 OS **分页**：逻辑上的 KV 序列映射到 **不连续物理块**，按需申请/回收，提高 **同一时刻能塞下的请求数**。

### 3.2 Continuous batching

某条生成 **EOS** 后立刻腾槽，新请求 **插入当前 decode 时间步**，提高 **吞吐（tokens/s）**。

**batch 与吞吐**：batch 增大先 **摊薄 kernel 固定开销**；过大则带宽/OOM，吞吐反降。

### 面试口述

「vLLM = KV 分页 + 连续批处理提吞吐；不是改模型。」

---

## 4. 与 FlashAttention 的分工（别混一层）

| 层级 | 代表 | 解决什么 |
|------|------|----------|
| **算子** | FlashAttention、FlashDecoding | 单次 attention **少写 HBM**、分块 + online softmax（见 **`04` §9**） |
| **服务** | vLLM | **多请求 KV 怎么存、batch 怎么拼** |

底层 serving 仍常调用 **FlashAttention kernel**；二者 **叠用**。

---

## 5. 工程进阶：chunked prefill、投机 decode

**Chunked prefill**：极长 prompt 时 **分段 prefill**，避免单次激活峰值顶满显存；代价是更多 kernel 次数。

**Speculative decode**：小 **草稿模型** 先猜多 token，大模型 **并行验证** 前缀；接受则一步产出多 token，在带宽受限的 decode 下提高 **有效 tokens/s**；失败回退单步。

---

## 本篇小结

- **Prefill** = 整段 prompt 前向；**Decode** = 逐步生成，常 **带宽瓶颈**。  
- **KV Cache** = prefill 写入、decode 追加；**只存 K、V**。  
- **vLLM** = KV 分页 + continuous batching。  
- **Flash** = 算子层；**vLLM** = 服务层。
