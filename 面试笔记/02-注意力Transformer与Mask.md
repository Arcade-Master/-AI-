# 02｜注意力、Transformer、Mask 与 KV Cache

**建议阅读顺序（本篇内部）**：**Q/K/V 含义** → **注意力公式与输出形状** → **为什么要除以 sqrt(d_k)** → **softmax 干什么** → **为何拆三个矩阵** → **多头** → **时间/空间复杂度**（建立「为何长文贵」）→ **mask**（为何能/不能看某些位置）→ **Prefill 与 Decode** → **KV Cache** → **和 RNN 类模型对比** → 文末代码。

术语：下文 **compute-bound** 写作「**算力受限**」（算子大、GPU 算得满）；**memory-bound / bandwidth-bound** 写作「**显存带宽受限**」（算子小、卡在读写显存）。

---

## 1. Q、K、V 各自回答什么问题

- **Q（Query）**：当前位置「我想查什么」。  
- **K（Key）**：每个位置「我被匹配时的索引签名」。  
- **V（Value）**：匹配上了之后「真正读出来的内容向量」。

同一套隐藏状态，通过三个线性层变成三种角色，才能同时学「问法」「索引」「取值」，否则表达能力会绑死在一起（见第 5 节）。

---

## 2. Scaled Dot-Product Attention 公式与形状

```
scores = Q @ K^T / sqrt(d_k)
weights = softmax(scores, 在「被 attend 的那一维」上做)
output = weights @ V
```

常用矩阵写法（单头）：

```
Q = X W_Q,   K = X W_K,   V = X W_V
Attn(X) = softmax( QK^T / sqrt(d_k) ) V
```

若 Q 形状 `(batch, Lq, d)`，K/V `(batch, Lk, d)`，则 **weights** `(batch, Lq, Lk)`，**output** `(batch, Lq, d)`：**输出序列长仍是 Lq**，最后一维仍是 d（多头时再拼接/投影）。

### 面试怎么答

「注意力是对 K 那一维做 softmax 加权求和 V；输出长度跟 Q 走。」

---

## 3. 为什么要除以 sqrt(d_k)

点积里每个标量是 **d_k 个乘积相加**。若 Q、K 各分量方差大致 O(1)，点积方差随 d_k **线性涨**，幅度约 **O(sqrt(d_k))**。不缩放则 softmax 输入过大 → **权重几乎 one-hot** → 梯度消失，训练不稳。

**一句话**：缩放是在控制「点积的尺度」，让 softmax 不饱和。

---

## 4. Softmax 在这里的角色；能换吗

Softmax 把任意实数分数变成 **非负、和为 1** 的权重，使输出是 V 各行的 **凸组合**——可微、可解释「分配多少注意力」。

换成 **sigmoid 对每个 key 独立归一** 会变成「多标签」式关注，不再是标准互斥注意力；**不用归一化**则失去「概率混合」的语义。面试答：**标准注意力要对 key 维归一成权重分布**。

**数值**：每行先减该行 max 再 exp，与 CE 那边同理。

---

## 5. 为什么要三个矩阵 W_Q、W_K、W_V，而不是一个

单矩阵等价于「同一线性空间里的对称匹配」，**查询子空间** 与 **被匹配子空间** 被迫相同；拆开后可以让「匹配发生在子空间 A」「取值发生在子空间 B」，表达 **「谁像谁」与「读出什么」解耦**。

**Cross-Attention**：Q 来自 decoder 当前位置，K/V 来自 encoder；**Self-Attention**：三者同源，只是角色仍拆分。

### 面试怎么答

「一个矩阵把『匹配』和『取值』绑死了；拆三个才能学不同投影。」

---

## 6. 多头注意力（MHA）

把 `d_model` 切成 h 份，每头一份 `d_k`，各头并行做 attention，再 concat + `W_O`。

```
head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
MHA(X) = Concat(head_1, ..., head_h) W_O
```

**直觉**：不同头可分工（局部搭配、长距指代等），比单头超大维点积更好优化。与 **MQA/GQA/MLA**（为省 KV）的关系见 `06`。

---

## 7. 时间与空间复杂度（先懂这个，再读 04 显存）

记号：序列长 **L**，宽 **d_model**，头数 **h**，单头维 **d_k ≈ d_model/h**。

**时间（单层 self-attention 量级）**：  
- 三个线性投影：约 **O(L · d_model²)**。  
- `QK^T` 与后续加权：**O(L² · d_model)** 主导（相对 L）。  

**空间**：若把每头完整 **L×L** 分数矩阵 materialize 到显存（高带宽显存，口语仍常说「显存」），峰值多 **O(L²)**；**FlashAttention** 用分块融合减少 **HBM**（GPU 上容量大、相对慢的那层显存）往返，降低峰值。训练时 **激活** 常随 L、层数一起爆；推理 decode 要存 **KV Cache**（见第 10 节）。

**长文难**：算力 ∝ L²、显存/带宽压力、以及位置编码外推、注意力「摊太薄」等叠加。

### 面试怎么答

「瓶颈是 L×L 的注意力矩阵；Flash 主要省峰值和带宽，不是把理论复杂度 magically 变成 O(L)。」

---

## 8. 训练为什么需要 mask

**因果 mask（下三角）**：自回归训练位置 t **不能看见 t 之后**，否则 **标签泄漏**，推理对不上。

**Padding mask**：同一 batch 长度不同，pad 位置不能参与 softmax 分母与聚合；做法是给非法位置加大负数，或等价实现。

**推理**：单条生成常只有因果；**批推理**拼 batch 时仍可能要 pad mask。

**训练 vs 推理长度**：训练 max 长度、推理实际 prompt+生成长度 **可以不同**，受配置与显存限制。

---

## 9. Prefill 与 Decode（和 KV 强绑定）

- **Prefill**：一次性吃完整段 prompt，矩阵大，通常 **算力受限** 更明显。  
- **Decode**：每步只多一个 token，矩阵小，**读权重、读 KV、写回** 占比高，常 **显存带宽受限**，GPU 容易「吃不饱」。

---

## 10. KV Cache：为什么只存 K、V

Decode 时，历史 token 的表示在变的是整体 hidden，但 **已算过的 K、V 对应当前层可复用**；新 token 只新增自己的 Q、K、V，其中 **历史 K、V 不必重算**，故缓存 **每层每头的 K、V**。历史 Q 不会再被用到。

显存粗算仍见 `04` 与上文「每层 × 2 × …」。

---

## 11. Transformer 与 RNN/LSTM 类 seq2seq

**并行**：Transformer 在序列维可并行算（配 mask）；RNN **时间步依赖链**，训练难并行。

**长距**：RNN 梯度路径长；Attention 在固定层数内让任意位置 **直接连边**（仍受表示能力限制，但不是一步步传）。

**缺点**：L 大时 **L²** 贵；小数据时归纳偏置弱；decode 侧带宽瓶颈。

### 面试怎么答

「Transformer 用注意力换并行与长距；代价是二次方与显存；RNN 省显存但难并行。」

---

## 11.1 再看一眼 softmax：它在做的其实是「可微 bipartite 软匹配」

固定一行 `Q_i`（长度 `L_k` 的分数向量），`softmax` 输出 **非负、和为 1** 的权重，使 `output_i` 是 **`L_k` 行 V 的凸组合**。若把 **K 行** 看成「候选槽位」，**这就是带温度（由分数尺度隐含）的软选择**。温度隐式由 **`scores / sqrt(d_k)`** 与 **mask 大负数** 控制：**分数很尖** → 近似 one-hot（几乎只盯一个 key）；**分数平** → 信息混合、梯度更分散。

---

## 12. 单头注意力参考实现（NumPy）

```python
import numpy as np

def attention(Q, K, V, causal=False, eps=1e-9):
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

## 本篇小结

- 注意力 = **软检索**；√d_k = **控方差**；softmax = **在 key 维上归一成权重**。  
- mask = **训练与推理因果一致 + batch padding**。  
- KV Cache = **decode 不复算历史 K/V**。  
- 复杂度 = **理解长文与显存账单的前置**。  
- softmax 行 = **对 key 维的凸组合**；mask 与 **√d_k** 共同决定「多尖」。
