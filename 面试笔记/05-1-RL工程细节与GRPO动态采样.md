# 05-1｜RL 对齐工程细节：GAE、异步 Judge、off-policy、动态采样

**定位**：`05` 讲管线选型；本篇讲 **一条训练 step 里数据怎么流、loss 怎么算、常见实现坑**。

---

## 1. 公式链：从 REINFORCE 到 GRPO

### 1.1 REINFORCE 与 baseline

最大化 `J(θ) = E_{τ~π_θ}[R(τ)]`。策略梯度：

```
∇_θ J ≈ E[ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]
```

`G_t` 是从 t 起的回报，方差大。减与动作无关的 baseline `b(s_t)`（常用 `V(s_t)`）不改期望、降方差：

```
A_t = Q(s_t,a_t) - V(s_t)     （优势：比该状态平均好多少）
```

### 1.2 TD 误差与 GAE

**一步 TD 误差**（Value 的训练信号，也用于构造优势）：

```
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

**GAE(λ)**：多步 δ 的指数加权和，得到用于策略更新的 `A_t`：

```
A_t^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

`λ` 越大越接近 Monte-Carlo（低偏高方差），越小越接近单步 TD。

### 1.3 PPO clip

旧策略采样，用当前策略评估同一动作：

```
r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
L^CLIP = E_t[ min( r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t ) ]
```

常再加 **value loss** ` (V(s_t) - target_t)^2 `、**KL(π_θ ‖ π_ref)**、熵 bonus。

### 1.4 GRPO（无 critic）

对同一 prompt `x` 采 **G 条**完整回答 `y_i`，标量奖励 `R_i`：

```
μ = mean(R_i),  σ = std(R_i) + ε
A_i = (R_i - μ) / σ          （或 A_i = R_i - μ）
L_i = Σ_t log π_θ(y_{i,t} | x, y_{i,<t})     （整条序列 log-prob 和）
策略项：- mean_i( A_i · L_i )  - β·KL(π_θ ‖ π_ref)
```

**GSPO**：把 token 级 `r_t` 收到 **序列级** 几何平均比率 `s_i(θ)` 再 clip，降长链方差（见 `05` §4.1）。

---

## 2. RM 打分、稀疏 reward、Value 怎么训

### 2.1 RM 打的是整条序列，不是每一步

经典 RLHF 里 RM 输入 `(x, y)`，输出 **一个标量** `r(x,y)`。RL 阶段常见设定：

- 生成过程中 **`r_t = 0`（t < T）**
- **只在结束步**（或 EOS 后虚拟一步）放入 RM 分数：**`r_T = RM(x,y)`**

要对 **每一步** 打过程分，需要 **PRM（Process Reward Model）**，不是默认 RM。

### 2.2 没有 t+1 时：`V(s_{T+1}) = 0`

轨迹结束（`done=True`）后没有未来，定义 **`V(s_{T+1}) = 0`**。最后一步：

```
δ_T = r_T + γ·0 - V(s_T) = r_T - V(s_T)
```

前面各步 `r_t=0`，但 `δ_t` 里仍有 `γ V(s_{t+1})`，再经 GAE 把 **最终 reward 的信用** 往前传。

### 2.3 Value 的 label 与 loss

- **`V(s_t)`**：critic 网络输出，表示「从 t 继续生成，预期总回报」。
- **`target_t` / `G_t`**：从轨迹 **算出来的真实累积回报**（不是模型猜的）  
  - Monte-Carlo：`G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...`  
  - 或 GAE 里的 **λ-return**（更稳）
- **Value loss**：

```
L_V = (V(s_t) - G_t)^2
```

与策略 loss **分开算、同一 backward 或分优化器**。策略 loss 里乘的 **A 来自 GAE**，不是直接把 `G_t` 乘进 `log π`。

**数值例**（γ=1）：`r_1=0, r_2=0, r_3=1` → `G_1=G_2=1, G_3=0`。Value 要学：第 1、2 步预测接近 1，第 3 步接近 0。

---

## 3. Rollout buffer 存什么、更新时算不算梯度

### 3.1 不存梯度，只存样本

异步 Judge 场景下 buffer 里典型字段：

| 字段 | 含义 |
|------|------|
| prompt / response token ids | 条件与生成 |
| `log π_old` | rollout 时策略算的 logprob（按 token 或序列和） |
| `reward` | Judge 写回（可能延迟） |
| `group_id` | 同一 prompt 的 G 条归一组 |
| `π_old_step` / 时间戳 | 控制 staleness |

**不存** 整段计算图或梯度；显存主要是 token + 标量，远小于存梯度。

### 3.2 更新时：主流是重新前向

打分完成、从 buffer 取 batch 后：

1. 用 **当前** `π_θ` 对 `(x, y)` **再 forward 一次**，得到 `log π_current`。
2. 算 advantage（GRPO 组内 z-score 等）。
3. 算 loss → `backward` → `optimizer.step()`。

**为何不能只存 `log π_old` 就算 ratio**：  
`ratio = π_current / π_old` 的分子 **必须用当前参数** 才算；只存旧 logprob 没有 `π_current`，无法准确做 importance sampling（除非接受 bias、不校正 off-policy）。

### 3.3 异步带来的 off-policy

rollout 用 `π_old`，训练时可能已是 `π_current`。处理：

- **Re-forward + clip**（PPO/GRPO 常见）
- **staleness**：样本落后超过 K 个 optimizer step 就丢弃
- **轻度 off-policy**：小学习率、短队列，不做完整 IS，靠 group baseline 扛一点偏差

---

## 4. 异步 LLM Judge 打分

**不同步等 Judge**：rollout 后样本进队列，训练 loop 继续；Judge worker 异步写 `reward`。

```text
rollout(π) → buffer（无 reward 或占位）
Judge 队列 → 写回 reward
train：取 reward 已就绪的样本 → re-forward → loss
```

规则分（单测、格式）可同步；慢的是 **LLM Judge API**。

**缓存**：`hash(x,y,judge_version) → score`，避免重复调用。

---

## 5. 奖励归一化（跨 batch 量级漂移）

Judge 一批打 1～2 分、下一批 10～100 分时，**原始 R 不能直接当 advantage 尺度**。

| 方法 | 做法 |
|------|------|
| Group z-score | 同 prompt 的 G 条：`(R_i - μ_G)/(σ_G+ε)`，GRPO 默认思路 |
| Running whitening | 滑动窗口维护 reward 的 mean/std，新 R 先标准化 |
| Advantage clip | 如 clip 到 [-5, 5] |

有 **critic V** 时，V 学 baseline，可部分吸收绝对尺度漂移；**GRPO 无 V** 时更依赖组内归一化。

---

## 6. 动态采样（DAPO 类）：实现与 loss 平均

### 6.1 为什么要丢「无方差」的 group

若一组 reward 全相同，如 `[0.8, 0.8, 0.8, 0.8]`：

```
A_i = (R_i - μ) / σ  →  全 0
loss_i = -A_i · log π_i  →  全 0
```

**不 mask**：仍占 forward/backward，梯度≈0，浪费算力、增大梯度估计方差。  
**动态采样**：`std(R_i) < ε` 的 prompt **整组跳过**，不参与本 step 的 loss。

### 6.2 向量化实现（伪代码）

```python
rewards = ...                    # [B, G]
std_g = rewards.std(dim=1)       # [B]
valid = std_g > threshold        # [B] bool

if not valid.any():
    continue

r = rewards[valid]               # [B', G]
logp = model.forward(responses[valid])   # re-forward
mu, sig = r.mean(1, True), r.std(1, True) + 1e-8
adv = (r - mu) / sig
loss = -(adv.unsqueeze(-1) * logp).mean()   # 对有效样本 mean
loss.backward()
```

### 6.3 mask 后会不会把梯度「稀释」？

**会，若用错分母**：

```python
# 错：用原始 batch 大小 B 做平均，有效组只有 B'
loss = loss_sum / B              # 梯度被人为拉小

# 对：只对有效样本平均
loss = loss_per_valid.mean()
# 或 loss_sum / valid.sum()
```

不 mask 时，无效 group 的 loss≈0，**均值也被拉低**，但有效组的 per-sample 梯度方向仍对；问题是 **白算 forward**。动态采样 + **对 valid 做 mean** 既省算力又不压低有效梯度幅度。

### 6.4 不 mask 时数值上差多少？

8 个 group 里 3 个 advantage 全 0、5 个正常：  
`loss = (0+...+正常)/8`，总 loss 约为「只对 5 个 group 平均」的 **5/8**，梯度幅度偏小。mask 后用 `mean over 5` 恢复正确尺度。

---

## 7. Judge / RM 要不要微调

| 角色 | 微调 | 说明 |
|------|------|------|
| 在线 RL 的打分器 | 长期常训小 RM | 便宜、口径稳 |
| 仅造 DPO 偏好对 | 可不微调 | 强模型 API 批量产 `(y_w,y_l)` |
| 策略 π_θ | 不为打分而训 | 别用同一 checkpoint 自评 |

---

## 本篇小结

- **稀疏 reward**：前面 `r_t=0`，最后 `r_T=RM分`；结束态 **`V(s_{T+1})=0`**；GAE 把终局分往前传。  
- **Value**：`L_V=(V(s_t)-G_t)²`；策略 loss 用 **A^GAE**，不是直接用 G_t。  
- **Buffer 存样本+log π_old+reward**，不存梯度；更新时 **re-forward** 算 `π_current`。  
- **异步 Judge** → off-policy → re-forward / staleness / 轻度容忍。  
- **动态采样**：`std(R)<ε` 丢组；**loss 对 valid 做 mean**，勿用原始 B 做分母。

### 面试口述

「RM 一般整条序列一个分；Value 用回报 G_t 做 MSE；GRPO 用组内 z-score 当 A。Judge 异步则 buffer 存轨迹、训练时重算 logprob；无方差 group 过滤掉，loss 只对有效 group 平均，避免梯度被无效样本稀释。」
