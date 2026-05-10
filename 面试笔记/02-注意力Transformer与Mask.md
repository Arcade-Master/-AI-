# 02｜注意力、Transformer、Mask 与 KV Cache

对应：**第一批**（多头注意力、QKV）；**第二批**（Attention 公式与代码、mask、√dk、softmax、QKV 为何三分、自注意 vs 交叉注意、O(n²)、KV Cache、与 seq2seq 对比）。

---

## 1. Attention 在算什么：查询–检索–读出

把每个 token 的隐藏向量想成「我想从句子里取什么信息」：

- **Q（Query）**：当前位置「问什么问题」。
- **K（Key）**：每个位置提供的「可被匹配的索引」。
- **V（Value）**：匹配成功后「实际读出的内容」。

**Scaled Dot-Product Attention**（单头，矩阵形式）：

```
scores = Q @ K^T / sqrt(d_k)
weights = softmax(scores, dim=-1)   # 对「被 attend 的序列维」归一化
output = weights @ V
```

**输出形状**：若 Q 为 `(batch, Lq, d)`，K/V 为 `(batch, Lk, d)`，则 weights 为 `(batch, Lq, Lk)`，output 为 `(batch, Lq, d)`。即 **序列长度保持为 Lq，最后一维保持为 d**（多头再拼接/投影）。

---

## 2. 为什么要除以 sqrt(d_k)？

点积 Q·K 的每个元素是 **d_k 个乘积之和**。若 Q、K 分量方差大致 O(1)，则点积方差随 d_k **线性增长**，数值幅度约 **O(sqrt(d_k))**。不缩放时，softmax 输入过大 → **近似 one-hot** → 梯度接近 0（饱和），反传困难。

**本质**：让 softmax 输入的尺度与维度无关，使训练深度稳定。这是「方差控制」而不是玄学常数。

**其它 scale 技巧**（知道即可）：部分工作用可学习 temperature；稀疏/线性注意力会换核函数；FlashAttention 是 **IO 与实现** 层面优化，数学目标可与标准 attention 一致。

---

## 3. Softmax 在 attention 里的角色：能不能换？

- **作用**：把任意实数分数变成 **非负、和为 1** 的凸组合系数，从而 output 是 V 的行的 **凸包** 内插——可微、可解释「分配多少注意力」。
- **换别的激活**：若不用归一化（只用 relu），权重和不为 1，语义不再是「概率式混合」；若用 sigmoid 对每个 key 独立归一，则 **不再互斥**，表达的是多标签式关注，与标准 Transformer 设计不同，需要配套架构（如部分稀疏工作）。

面试答法：**标准注意力需要「对 key 维归一化的混合权重」**；softmax 是最简可微选择。

**数值稳定**：对 scores 每行减该行 max，再 exp / sum；与交叉熵一样，**稳定 softmax + log** 是工程常识。

---

## 4. Q、K、V 为什么要三个矩阵，而不是一个？

单矩阵只能做 **同一线性变换后再点积自身**，表达能力等价于一种对称二次型，**角色纠缠**：「问什么」和「供什么索引」被迫共享同一套方向。

拆成 W_Q、W_K、W_V 后：

- Q 与 K 可学习 **不同的子空间** 做匹配（相似度在「匹配空间」里算）；
- V 可学习 **读出什么信息**（与「匹配分数」解耦）。

**Cross-Attention**：Q 来自 decoder 当前步，K/V 来自 encoder 记忆；**Self-Attention**：Q/K/V 同源序列，分工仍是解耦「匹配」与「取值」。

---

## 5. 多头注意力（MHA）

把 d_model 切成 h 份，每份 d_k = d_model / h，各头并行算 attention，再 concat 经 W_O 融合。

**直觉**：不同头可捕捉不同关系（句法、指代、局部 n-gram 等），**子空间分工**；单头过大维度的点积反而难优化。

与 **MQA/GQA** 的对比见 `06-DeepSeek-R1-MLA与MoE.md` 中的 MLA 小节（推理侧 KV 共享与缓存）。

---

## 6. 训练为什么需要 mask？

**Causal mask（下三角）**：自回归语言模型在位置 t 不能看到 t 之后 token，否则训练时 **信息泄漏**，推理时对不上。

**Padding mask**：batch 内序列长短不一，padding 位置不应参与 attention 聚合，也不应影响 softmax 分母；做法是对非法位置 scores 加 **大负数**（或乘 0 再归一，注意实现一致性）。

**推理时**：自回归 decode 步通常仍用 causal 结构（或等价缓存机制）；**padding mask** 在单条生成可能无；批推理拼接序列时仍可能需要。

**训练 vs 推理序列长度**：训练常 **pack 或 pad 到固定长度**（或变长 + mask）；推理 prompt 长度与生成长度 **一般不等于** 训练时的固定 max（受 max_position / 显存 / 服务配置限制）。

---

## 7. Prefill 与 Decode（与 mask、KV 强相关）

- **Prefill**：一次性处理 prompt，可高度并行，算的是「整段上下文」的 Q、K、V；主要算力密集。
- **Decode**：逐步生成，每步只多一个新 token；历史 token 的 K、V **可缓存**，每步只算新 token 的 Q 及对应新 K、V 并追加缓存。

**为何 decode 难吃满 GPU**：每步 batch 小、序列短，kernel  launch 与访存占比高，**memory-bound**；详见 `08-推理部署Prefill与vLLM.md`。

---

## 8. KV Cache：是什么、算什么、为何只存 K/V

**为何需要**：decode 时新 token 的 attention 需要对所有历史位置算分数；历史位置的 K、V **不变**，重复算是浪费。

**为何只存 K/V**：每步的 Q 只对应当前新 token；历史 Q 不会再被用到。缓存的是 **每层、每头** 的 K、V tensor。

**粗算显存**（BF16 一字节 2 bytes 仅作数量级；实际还有布局、对齐）：

```
每层 KV ≈ 2 × batch × seq_len × num_kv_heads × head_dim × bytes_per_elem
总 KV ≈ 层数 × 上式
```

`num_kv_heads` 在 MQA/GQA 中小于 `num_heads`，故省缓存（见 06 篇 MLA）。

---

## 9. Transformer vs 传统 seq2seq（RNN/LSTM）

**并行**：Transformer 在序列维上可同时算所有位置的 self-attention（配合 mask）；RNN 每步依赖上一步隐状态，**时间步串行**，训练难并行。

**长距离**：RNN 梯度路径长易消失；Attention 任意两位置 **常数层数内** 可达（仍受层数与表示能力限制，但不是「一步步传」）。

**缺点**：O(n²) 算力与显存、对超长上下文不友好；数据少时不如归纳偏置强的模型稳；推理 decode 侧带宽瓶颈。

---

## 10. Attention 的时间复杂度与空间复杂度（面试 explicit 版）

记号：序列长度 **L**，模型宽度 **d_model**，头数 **h**，单头维度 **d_k ≈ d_model / h**，batch 设为 B（分析时常先取 B=1）。

### 时间复杂度（单步、单层 self-attention）

- **Q/K/V 线性投影**：各约 **O(L · d_model²)**（三个矩阵乘），合计 **O(L · d_model²)**。
- **注意力分数** `S = Q K^T / sqrt(d_k)`：结果是 **(L×L)**，每个元素是 **d_k** 次乘加 → **O(L² · d_k)**；乘 h 个头，若 **h·d_k ~ d_model**，整体 **O(L² · d_model)**。
- **softmax 与加权 V**：对 S 的 **O(L²)** + `attn @ V` 的 **O(L² · d_k)** → 仍 **O(L² · d_model)** 量级。

**结论**：相对长度 L，**主导项常是 O(L² · d_model)**（多头合计后）；相对宽度，还有 **O(L · d_model²)** 的投影项。面试说「**对序列长度平方**」通常指 **注意力矩阵** 这一块。

### 空间复杂度（峰值显存直觉）

- **物化完整注意力矩阵**（每头）：**O(L²)** 浮点数；FlashAttention 等 **分块融合** 可把峰值降到近似 **O(L)** 级别的额外缓存（仍要存 KV，见下）。
- **KV Cache（推理 decode）**：每层约 **O(L · d_model)**（与 head 布局有关，常记 **2·L·d** 量级理解 K+V）。
- **激活（训练）**：与实现是否 checkpoint 有关；**自注意力激活** 常带 **L²** 项，是长序列训练 OOM 主因之一。

### 与「O(n²) 在哪里」的关系

主要瓶颈仍是 **scores = Q K^T** 及后续对 **(L, L)** 结构的依赖；序列加倍，算力与（未优化时的）显存大致按 **平方** 增长。FlashAttention 等降低 **常数与 IO、峰值**，**渐近阶** 除非换线性/稀疏注意力，否则对 L 仍是二次为主。

**长上下文困难**：算力、显存、位置编码外推、注意力分散（难聚焦）等叠加。

---

## 11. 极简 NumPy 风格 attention（单头）

```python
import numpy as np

def attention(Q, K, V, causal=False, eps=1e-9):
    # Q,K,V: (L, d)
    d = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d)
    if causal:
        L = scores.shape[0]
        mask = np.triu(np.ones((L, L), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True) + eps
    return w @ V, w
```

---

## 小结

- Attention = **可学习的软检索**；√dk、softmax、三分 QKV 都有明确 **稳定性或可表达性** 动机。
- Mask 连接 **训练–推理一致性** 与 **批处理 padding**。
- KV Cache 把 decode 从「重复算历史」变成「只算新 token」，是推理工程的核心之一。
