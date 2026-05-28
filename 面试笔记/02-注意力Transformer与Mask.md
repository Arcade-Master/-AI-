# 02｜注意力、Transformer 结构、RoPE 与 Mask

**本篇只讲「模型结构里 attention 怎么算」**：Q/K/V、公式、mask、复杂度、与 RNN 对比、手撕。  
**推理侧** Prefill / Decode / KV Cache / vLLM → 统一见 **`08`**。  
**训练显存与 FlashAttention** → **`04`**。长窗外推、量化、评测 → **`07`**。手撕：[`手撕代码/AI手撕/02-手撕多头注意力.md`](../手撕代码/AI手撕/02-手撕多头注意力.md)。

术语：**算力受限** = compute-bound；**显存带宽受限** = memory-bandwidth-bound。

---

## 1. Q、K、V 各自回答什么问题

- **Q（Query）**：当前位置「我想查什么」。  
- **K（Key）**：每个位置「我被匹配时的索引签名」。  
- **V（Value）**：匹配上了之后「真正读出来的内容向量」。

同一套隐藏状态，通过三个线性层变成三种角色，才能同时学「问法」「索引」「取值」（见 §5）。

---

## 2. Scaled Dot-Product Attention 公式与形状

```
scores = Q @ K^T / sqrt(d_k)
weights = softmax(scores, 在 key 维上)
output = weights @ V
```

单头：`Q = X W_Q, K = X W_K, V = X W_V`，`Attn(X) = softmax(QK^T/√d_k) V`。

Q `(batch, Lq, d)`，K/V `(batch, Lk, d)` → weights `(batch, Lq, Lk)`，output `(batch, Lq, d)`。

---

## 3. 为什么要除以 sqrt(d_k)

点积方差随 d_k 线性涨，幅度约 O(√d_k)。不缩放则 softmax 饱和、梯度消失。缩放是 **控点积尺度**。

---

## 4. Softmax 在这里的角色

把分数变成 **非负、和为 1** 的权重，output 是 V 行的 **凸组合**。标准注意力要在 **key 维** 归一；每行减 max 再 exp。

---

## 5. 为什么要 W_Q、W_K、W_V 三个矩阵

单矩阵会把「匹配」和「取值」绑在同一子空间；拆开才能 **谁像谁** 与 **读出什么** 解耦。Cross-Attention：Q 来自 decoder，K/V 来自 encoder。

---

## 6. 多头注意力（MHA）

```
head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
MHA(X) = Concat(head_1, ..., head_h) W_O
```

**`W_O` 之后**：concat 后经 `W_O: d_model→d_model`，残差 + Norm，下一子层是 **FFN**（Linear→激活→Linear）。

与 **MQA/GQA/MLA**（省 KV）见 **`06`**。

---

## 7. 位置编码与 RoPE（Rotary Position Embedding）

### 7.1 为什么必须单独加「序」

Self-attention 对输入 token 的 **置换** 几乎不变：你把第 3 个词和第 7 个词对调，只要对应的 Q/K/V 行也对调，**同一套权重算出来的注意力结构一样**。模型因此 **默认不知道谁先谁后**。必须在进 attention 之前（或与之结合）注入 **位置信息**。常见三类：**绝对位置向量**（加在 embedding 上）、**相对位置偏置**（加在 score 上）、**RoPE**（旋进 Q/K，让点积自带相对距离）。

### 7.2 RoPE 在算什么

把隐藏维 **按相邻两维一对** `(x_{2i}, x_{2i+1})` 看成平面上的一个向量，对 **位置下标 m** 旋转一个角度 **m·θ_i**（第 i 对维度用角频率 **θ_i**）。旋转后的向量记为 **R_m x**（对整个 d 维向量是 **块对角** 的 2×2 旋转矩阵拼起来）。

对第 i 维对，标准写法（与 LLaMA 等实现一致）：

```
θ_i = base^(-2i/d)     # 常取 base=10000，d 为 head 维或参与 RoPE 的维数

x'_{2i}   = x_{2i}   cos(m·θ_i) - x_{2i+1} sin(m·θ_i)
x'_{2i+1} = x_{2i+1} cos(m·θ_i) + x_{2i}   sin(m·θ_i)
```

**R_m**：第 i 个 2×2 块为

```
R_m^(i) = [ cos(m·θ_i)  -sin(m·θ_i) ]
          [ sin(m·θ_i)   cos(m·θ_i) ]
```

全维 **R_m = diag(R_m^(0), R_m^(1), …, R_m^(d/2-1))**。  
**θ_i 越小 → 转得越慢 → 波长越长**，更抓 **远距离** 的相对关系；**θ_i 大** 的块对 **相邻 token** 更敏感。

对 Q、K 分别施加位置（实现里常先线性投影再旋，或先旋再投影，等价变体以代码为准）：

```
q_m = R_m (W_Q x_m)
k_n = R_n (W_K x_n)
```

**V 一般不做 RoPE**（历史 K/V 缓存的是已旋好的 K；推理细节见 **`08`**）。

### 7.3 为什么注意力分数只依赖相对距离 (n−m)

二维旋转满足：**R_m 是正交矩阵**，且 **R_m^T R_n = R_{n−m}**（转 m 步再转 n 步 = 净转 n−m 步）。

位置 m 的 query 与位置 n 的 key 做点积：

```
score(m, n) = (R_m q)^T (R_n k)
            = q^T R_m^T R_n k
            = q^T R_{n-m} k
```

右边 **只出现 (n−m)**，不出现单独的 m、n。这就是 RoPE 的核心：**相对位置进了 softmax 前的分数**，而不是简单给 embedding 加一个绝对位置向量。

数值例子（单维对、忽略 W）：设 `q = k = [1, 0]`，θ=1。则 `R_0 q = q`，`R_3 q` 为绕原点转 3 弧度后的向量；`q^T R_3 q` 与「相距 3 步」的相位一致，和「m=0,n=3」与「m=5,n=8」在 **相对距离 3** 时用的是 **同一块旋转 R_3**（在理想训练分布内）。

### 7.4 和「绝对位置 embedding」比一眼

| | 绝对位置（加在 x 上） | RoPE（旋在 Q/K 上） |
|---|----------------------|---------------------|
| 进模型的方式 | `x + p_m` | `q_m = R_m W_Q x_m` 等 |
| 分数里位置怎么出现 | 间接，经 W 与点积耦合 | 显式 **q^T R_{n-m} k** |
| 长文外推 | 依赖 p_m 是否见过 | 依赖 **n−m 与相位** 是否 OOD（见 **`07` §1～§2**） |

多模态里 patch 网格用 **2D RoPE**（行、列各一套角频率）见 **`11` §5**；文本 LLM 用本节 **1D RoPE**。

### 面试口述

「Attention 不辨序；RoPE 用块对角旋转把位置编进 Q/K，点积整理后只依赖 n−m。」

---

## 8. 时间与空间复杂度

记号：序列长 **L**，宽 **d_model**，头数 **h**。

- 线性投影：约 **O(L · d_model²)**  
- `QK^T` 与加权：**O(L² · d_model)** 主导  

若 **物化完整 L×L 分数矩阵**，激活峰值 **O(L²)**。训练侧用 **FlashAttention** 降 HBM 峰值（**`04` §9**）。  
**推理** 的 L 增长、KV 线性涨、Prefill/Decode 形态 → **`08`**。

---

## 9. 训练为什么需要 mask

**因果 mask**：位置 t 不能看 t 之后，防标签泄漏。  
**Padding mask**：pad 位置不参与 softmax。  
推理单条生成常只有因果；批推理拼 batch 仍可能要 pad mask。

---

## 10. Transformer 与 RNN/LSTM

**训练并行**：Transformer 序列维可并行（配因果 mask）；RNN 时间维链式，难并行。  
**长距离**：Attention 一层连任意两位置；RNN 逐步传递易衰减。  
**推理**：Transformer decode 依赖 **KV Cache**（**`08`**）；RNN 用固定维隐状态 h，不显式存 O(t) 的 per-position KV。

**代价**：L 大时 attention **L²**；小数据归纳偏置弱。

---

## 11. 手撕（NumPy）

形状：`X (B,L,d_model)`，`d_k = d_model // n_heads`。  
完整实现：[`手撕代码/AI手撕/02-手撕多头注意力.md`](../手撕代码/AI手撕/02-手撕多头注意力.md)

---

## 本篇小结

- 注意力 = 在 key 维 softmax 加权 V；√d_k 控尺度；三矩阵解耦匹配与取值。  
- **RoPE**：`q_m = R_m W_Q x_m`，`score = q^T R_{n-m} k`；**R_m** 为按维对的 2×2 旋转块。  
- mask = 因果 + padding。  
- 复杂度 L²；训练 Flash 见 **04**；推理 Prefill/Decode/KV/vLLM 见 **08**；长窗外推见 **07**。
