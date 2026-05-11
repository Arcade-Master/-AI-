# 02｜注意力、Transformer、Mask 与 KV Cache

**建议阅读顺序（本篇内部）**：**Q/K/V 含义** → **注意力公式与输出形状** → **为什么要除以 sqrt(d_k)** → **softmax 干什么** → **为何拆三个矩阵** → **多头（含 W_O 与 FFN 衔接）** → **时间/空间复杂度**（建立「为何长文贵」）→ **mask**（为何能/不能看某些位置）→ **Prefill 与 Decode**（提示阶段 vs 解码阶段）→ **KV Cache** → **和 RNN 类模型对比** → **单头 / 多头 NumPy 手撕（§12～§13）**。

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

**`W_O` 之后到哪**：concat 后维度已是 `d_model`，再经 **`W_O: d_model → d_model`** 与残差相加（再 Norm），**下一子层才是 FFN（两层 MLP）**；`W_O` 不是 MLP 里的一层。

**标准 Transformer block 里（自注意力 + FFN）**：

- **Self-Attention**：`x → MHA(x)`（内含各头与 **`W_O`**）→ 得到注意力子层输出。  
- **残差 + Norm**：Pre-LN 常为 `Norm(x) → attn → +x`；Post-LN 常为 `attn → +x → Norm`（依实现）。  
- **FFN**：`Linear(d→4d) → 激活 → Linear(4d→d)`，再残差 + Norm。

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

**场景**：自回归 LM **推理**时，先输入整段 **prompt**，再 **一个 token 一个 token** 生成续写。工程上把这两段算力形态拆开叫 **Prefill** 与 **Decode**（有的框架把第一次也算进 prefill，语义一致：**「先处理已知前缀」vs「每步只处理新 token」**）。

### Prefill（提示阶段 / 填充阶段）

- **在算什么**：对 prompt 里 **所有位置**（长度记为 `L_prompt`）做一次（或按块一次）**完整前向**：每层 self-attn 里 **Q、K、V 的序列维都是整段 prompt**，注意力矩阵一侧是 **`L_prompt` 量级**，矩阵乘 **大、并行度高**。  
- **KV Cache 发生什么**：**第一次**前向结束后，把 **每一层、每一头** 上 **整段 prompt 的 K、V** 按形状写入 cache（实现上常是「当前序列长度维」上的一段连续存储）。**Prefill 的本质**：在「还没有任何 cache」或「prompt 变长需要重算新增段」时，**把已知 token 的 K/V 算好并落盘到 GPU 显存**，供后面 decode **只增量更新**。  
- **瓶颈直觉**：`L_prompt` 大时，算子 **大矩阵、高 FLOPs**，往往更偏 **算力受限**（GPU SM 能吃得较满）；同时会 **一次性分配/写入** 较长 KV，显存占用 **阶上跳一截**。

### Decode（解码阶段 / 自回归步进）

- **在算什么**：第 `t` 步只新来一个 token（或 beam 里若干），self-attn 里 **当前步的 Q 往往只有 1 个 query 位置**（或极短），但要 **attend 到前面所有位置**（长度 `L_prefix` 随生成变长），于是 **主要算力变成：读整段历史 K/V、算一行注意力、写回新 K/V**。  
- **KV Cache 发生什么**：**只追加**本步新 token 在各层算出的 **K、V**；**历史 K/V 从 cache 读，不重算**。  
- **瓶颈直觉**：矩阵 **「薄」**（batch×头数可能仍大，但参与 GEMM 的 M/N 一侧很小），GPU **算不饱**，时间花在 **HBM 带宽**（反复读权重、读 KV、写 KV）上 → 常 **显存带宽受限**。

### 对照一句

| | Prefill | Decode |
|---|---------|--------|
| **输入** | 整段 prompt（已知） | 每步通常 1 个新 token |
| **Q/K/V 典型规模** | 序列维 ≈ `L_prompt` | Q 常 ≈1 行，K/V 用 cache 全长 |
| **KV Cache** | **写入**整段 prompt 的 K/V | **追加**新 token 的 K/V |
| **常见瓶颈** | 算力（大 matmul） | 带宽（读 KV + 读权重） |

---

## 10. KV Cache：为什么只存 K、V

**为何是 K、V**：Decode 第 `t` 步算 attention 时，**当前位置的 Q** 要和 **从位置 1…t 的 K** 做点积得权重，再对 **同长度的 V** 加权求和。**过去每个位置的 K、V** 在后续步里还要反复被读；**过去位置的 Q** 不会再参与未来步的「当前 query」，因此 **不必存 Q**。

**与 Prefill 的衔接**：Prefill 结束时，cache 里已有 **prompt 各层各头的 K、V**；Decode 每步只对 **新 token** 算 Q、K、V，其中 **新 K、V 拼到 cache 末尾**，**Q 只用于本步输出**，不写进长期 cache。

**形状直觉（与 `04` 一致）**：每层大致 **`[batch, n_kv_heads, seq_len, head_dim]`** 的 K 与 V 各一份，乘层数；`seq_len` 随 decode 增长，显存 **线性涨**。

---

## 11. Transformer 与 RNN/LSTM 类 seq2seq

**并行（训练侧）**：Transformer 在序列维上 **自注意力可并行**（配因果 mask）；RNN/LSTM **步 t 依赖 h_{t-1}**，时间维 **链式**，单条序列内 **难并行**，长序列训练常更慢。

**长距离依赖**：RNN 信息沿时间步 **逐格传递**，梯度路径长易衰减；Attention 在 **固定深度** 内让任意两位置 **经一层即可连边**（表示能力仍受层数与宽度限制，但不是「一步一步挪过去」这一种结构偏置）。

**推理侧对比**：Transformer decode 依赖 **KV Cache** 避免每步 O(t²) 重算历史；RNN 类 **隐状态 h_t 固定维**，每步更新 h，**不显式存 O(t) 的 per-position KV**，长文时 **单步显存常更省**，但 **单步难以像 Transformer 那样并行利用历史全序列结构**。

**缺点**：L 大时 self-attn **L²** 与 KV **O(L)** 仍贵；小数据时 Transformer **归纳偏置弱**；decode 阶段易出现 **带宽瓶颈**（见第 9 节）。

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

## 13. 多头注意力手撕（NumPy）

约定：`X` 形状 **`(B, L, d_model)`**，`n_heads` 整除 `d_model`，**`d_k = d_model // n_heads`**。`Wq, Wk, Wv` 均为 **`(d_model, d_model)`**（与「每头各一块小矩阵」等价，只是拼成大矩阵乘一次再切头）。`Wo` **`(d_model, d_model)`**。

```python
import numpy as np


def softmax_lastdim(x, eps=1e-9):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + eps)


def scaled_dot_product_attention(q, k, v, causal=False):
    """q, k, v: (B, H, Lq, dk), (B, H, Lk, dk), (B, H, Lk, dk) -> (B, H, Lq, dk)"""
    dk = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) / np.sqrt(dk)  # (B,H,Lq,Lk)
    if causal:
        Lq, Lk = scores.shape[-2], scores.shape[-1]
        assert Lq == Lk, "causal self-attn expects Lq == Lk"
        mask = np.triu(np.ones((Lq, Lk), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
    attn = softmax_lastdim(scores)
    return np.matmul(attn, v), attn


def multi_head_attention(x, wq, wk, wv, wo, n_heads, causal=False):
    """
    x: (B, L, d)
    wq, wk, wv, wo: (d, d), d == n_heads * d_k
    return: (B, L, d)
    """
    b, l, d = x.shape
    assert d % n_heads == 0
    dk = d // n_heads

    def proj(t, w):
        return np.matmul(t, w)  # (B, L, d)

    # (B, L, d) -> (B, L, H, dk) -> (B, H, L, dk)
    def split_heads(t):
        t = t.reshape(b, l, n_heads, dk)
        return np.transpose(t, (0, 2, 1, 3))

    q = split_heads(proj(x, wq))
    k = split_heads(proj(x, wk))
    v = split_heads(proj(x, wv))

    out, _ = scaled_dot_product_attention(q, k, v, causal=causal)  # (B,H,L,dk)
    # merge heads: (B,H,L,dk) -> (B,L,d)
    out = np.transpose(out, (0, 2, 1, 3)).reshape(b, l, d)
    return np.matmul(out, wo)
```

**要点**：`split_heads` / `transpose` 只是在换维，**注意力仍是在最后两维上做 `Lq×Lk` 与 `Lk×dk`**；多批、多头时 **最外两维 B、H 是广播维**。

---

## 本篇小结

- 注意力 = **软检索**；√d_k = **控方差**；softmax = **在 key 维上归一成权重**。  
- mask = **训练与推理因果一致 + batch padding**。  
- **Prefill** = 整段 prompt 前向 + **写入 KV**；**Decode** = 每步增量 + **读历史 KV、追加新 KV**；后者常 **带宽受限**。  
- KV Cache = **decode 不复算历史 K/V**；只存 **K、V**。  
- 复杂度 = **理解长文与显存账单的前置**。  
- softmax 行 = **对 key 维的凸组合**；mask 与 **√d_k** 共同决定「多尖」。

