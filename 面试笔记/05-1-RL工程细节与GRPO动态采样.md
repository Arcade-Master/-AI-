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

### 1.2.1 在 LLM 里：`s_t`、`r_t`、`δ_t`、`A_t` 各是什么（易混：PPO 里的 `r_t(θ)` 是比率，不是 reward）

把 **每生成一个 response token** 当成 RL 里的一步：

| 符号 | 在 LLM 里指什么 |
|------|----------------|
| **状态 `s_t`** | 已看到的前缀：`prompt + y_{<t}`（下一步要采样 `y_t` 之前） |
| **动作 `a_t`** | 本步采样的 token `y_t` |
| **即时奖励 `r_t`** | 本步环境给的标量。**RM 整条打分** 时：中间步常 **`r_t=0`**，**写完**（EOS）那步 **`r_T = RM(x,y)`** |
| **`V(s_t)`** | Critic 网络输出：**「从这个前缀继续写完，预期还能拿多少总回报」**（要和自己定义的 `r_t` 同一套折扣） |
| **`δ_t`** | **一步 TD 误差**：这一步的「实际 + 对未来的估计」比「原先预期」好多少 |
| **`A_t`** | **优势**，给 **PPO 策略梯度** 用：在 `s_t` 选 `a_t` 比平均水平好/差多少；由 **GAE** 把当前及以后的 `δ` 加权得到 |

**`δ_t` 公式（记住一句）**：

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
      └─本步真拿到的┘ └─下一步预期┘ └─原先从这里起的预期┘
```

- **`δ_t > 0`**：比 critic 原先想的 **更好**（例如后面突然值很高，或本步拿到了 RM 分）。  
- **`δ_t < 0`**：比预期 **更差**。  
- 轨迹结束：**`V(s_{T+1}) = 0`**（没有未来了）。

**`γ`（gamma）**：折扣未来。LLM 里常取 **1**（终局 RM 分不分早晚都重要）。`γ<1` 则更重视靠近当前的 reward。

**`λ`（lambda，GAE 专用）**：在 **「只听一步 TD（低方差）」** 和 **「听整条回报（低偏差）」** 之间折中。  
- **`λ=0`**：`A_t = δ_t`，只看一步 surprise。  
- **`λ→1`**：`A_t` 接近用 **整条 MC 回报减 `V(s_t)`**，更信终局 RM，**方差更大**。  
- 实现里常 **0.95～0.99**。

**`r_t` 和 `A_t` 在各位置长什么样（稀疏 RM，`r_1=…=r_{T-1}=0`，`r_T=R`）**：

| 位置 | `r_t` | 直觉上的 `A_t`（`V` 训练正常时） |
|------|-------|----------------------------------|
| 中间 token | **0** | 常为 **正**（终局有好分，GAE 把信用往前传）；越早的 token `A_t` 往往略小（若 `γ<1` 或 `λ<1`） |
| 最后一个 token | **`R`（RM 分）** | 由 **`δ_T = R - V(s_T)`** 主导；若 RM 很高且 `V` 低估，**`A_T` 很大** |

**小数值例**（3 个 response token，`γ=1`，`λ=1`，RM 只在结尾给 **`r_3=1`**，前面 **`r_1=r_2=0`**；critic 已理想：`V(s_1)=V(s_2)=1`，`V(s_3)=0`，`V(s_4)=0`）：

```
δ_3 = 1 + 0 - 0 = 1
δ_2 = 0 + 0 - 1 = -1    # 本步没奖，但「下一步状态」预期已是 0，比 s_2 处预期的 1 低 → TD 说这一步过渡「不如预期」
δ_1 = 0 + 1 - 1 = 0

A_3 = δ_3 = 1
A_2 = δ_2 + δ_3 = 0
A_1 = δ_1 + δ_2 + δ_3 = 0
```

说明：**`A_t` 不是把 RM 分平均分给每个 token**；是 **TD 残差链** 算出来的。上面理想 `V` 下中间步 `A_2` 可为 0，但 **`V` 未收敛时** 前面 token 的 `A_t` 会非 0，从而仍收到梯度。训练 critic 的 target 常取 **`G_t = r_t + r_{t+1} + …`**（该例 **`G_1=G_2=1, G_3=0`**），与 **`L_V = (V(s_t)-G_t)²`** 一致。

**和 PPO clip 里 `r_t(θ)` 区分**：策略 loss 里的 **`r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)`** 是 **importance 比率**，与奖励 **`r_t` 同名不同义**。

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

### 2.4 PPO vs GRPO：整条 RM 分怎么落到各个 token？优势按位置怎么区分？

**共同前提**：经典 RM / Judge 对 **整段回答** 打一个标量 `R`（或 `r_T`），**默认没有**「第 3 个 token 好、第 7 个 token 差」的 per-token 标签。要对每一步打过程分，需要 **PRM** 或规则 **dense reward**。

#### PPO：reward 稀疏，**优势按 token 不同**（靠 GAE + Value）

1. **奖励分配**：`r_t = 0`（`t` 在 response 内、未到 EOS），**只在轨迹结束** 给 `r_T = R`（或最后有效步）。
2. **回报 / 优势**：用 **Value `V(s_t)`** 和 **GAE** 从后往前传信用。前面 token 的 `A_t` **可以彼此不同**——早步的 `A_t` 反映「从这里继续写，离最终 RM 分还有多远」；`V` 学得好时，越靠近结尾、与好坏更相关的步，`A_t` 往往更大（符号取决于好于/差于 baseline）。
3. **策略 loss**：**每个 token 一步**，`L^CLIP` 里是 **`r_t(θ) · A_t`**（`r_t` 是该 token 的 importance ratio），**不同位置乘不同的 `A_t`**。
4. **直觉**：RM 只给终点分，**不手工**把 R 除以长度；**GAE 自动**把终局奖励折成逐步优势。若 `V` 很烂或 λ 不合适，会出现 **credit 传错**（全句一起背锅或只改最后一个 token）。

#### GRPO：reward 仍是整条一个 `R_i`，**同一条回答里各 token 共用同一个 `A_i`**

对同一 prompt 采 G 条完整回答 `y_i`，每条一个标量 `R_i`：

```
A_i = (R_i - μ_G) / (σ_G + ε)     # 一条轨迹一个数
```

**策略项（标准 GRPO）**：

```
L_policy ∝ - Σ_i  A_i · Σ_{t ∈ response_i}  log π_θ(y_{i,t} | x, y_{i,<t})
```

- **`A_i` 不随 t 变**：第 i 条回答里 **每一个 response token 乘的是同一个 `A_i`**。
- **位置之间不靠不同的 A 区分**；区分的是 **哪一条轨迹**（`i`）更好/更差。好轨迹整体 **抬高** 该条所有 token 的 log π 梯度，差轨迹整体 **压低**。
- **prompt 段** 通常 **不进** 策略梯度（mask 掉），只对 **模型自己生成的 token** 求和。
- **细粒度 credit**：默认 **没有**「第 5 个 token 该多分、第 20 个少分」；若某 token 对最终质量关键，只能靠 **它自己的 log π** 在求和里占权重——**整条一起奖/一起罚**。长回答里 **前后 token 梯度尺度相同**（同一 `A_i`），这是 GRPO **简单但粗** 的地方。

#### 对照（易混）

| | PPO + RM | GRPO |
|---|----------|------|
| RM 分落在哪 | 通常 **最后一步** `r_T` | 整条一个 **`R_i`**，不进 per-step `r_t` |
| 各 token 的优势 | **`A_t` 逐步不同**（GAE） | **同一条内 `A_i` 相同** |
| baseline | **Critic `V(s_t)`** | **同组 G 条的 `μ_G, σ_G`** |
| 想按 token 区分好坏 | 用 **PRM** 或改 reward 设计 | 用 **PRM / dense 规则**；或 **GSPO** 等改 ratio 粒度（优势仍常为序列级） |

**GSPO 补充一句**：它主要把 **importance ratio** 从 per-token 收到 **序列级** 降方差，**不是**默认给每个 token 不同的 `A_t`；优势仍多是 **组内 z-score 的一个标量 per 序列**。

### 2.5 Critic `V` 怎么训、PPO 里何时前向（不是无监督）

**`V` 是什么**：一个 **神经网络**（常与 policy **共享 backbone + 单独 value head**，或独立 critic）。输入 **状态 `s_t`**（实现里多为 **当前前缀的 hidden state**），输出标量 **`V_θ(s_t)`**。

**不是无监督**：有 **监督目标**，来自 **这条 rollout 上真实出现的 reward 序列** 算出的 **`G_t` 或 λ-return`**（只用环境/RM 给的 `r_t`，不用人逐 token 标注）。

```
L_V = mean_t ( V_θ(s_t) - G_t )²
```

`G_t` 例（`γ=1`，`r_1=r_2=0, r_3=1`）：`G_1=G_2=1, G_3=0`。训练 **`V(s_1)≈1, V(s_2)≈1, V(s_3)≈0`** —— 标签来自 **已生成完的轨迹**，不是凭空猜。

**`V` 在 `δ_t` / `A_t` 里的角色**：`δ_t = r_t + γ V(s_{t+1}) - V(s_t)` 里的 **`V` 是「baseline / 预期」**。  
- 算 **`A_t` 时** 常用 **rollout 当时** 的 `V_old(s_t)`（**detach，不反传进这次 advantage 计算**），避免 moving target 太乱。  
- **训 `V` 时** 用 **当前** `V_θ(s_t)` 去拟合 **`G_t`**，**正常反传** 更新 critic 参数。

**PPO 两阶段（和手算顺序一致）**：

```text
【Rollout，常 no_grad】
  π_old 生成 y → 得到 r_t（RM 多在 r_T）
  → 前向 critic 得 V_old(s_t)（可存 buffer）
  → 用 r_t、V_old 算 δ_t、A_t（GAE），A_t 存下来

【Train，要梯度】
  同一 (x,y) 再 forward：log π_current(a_t|s_t)、V_current(s_t)
  → L_policy = PPO_clip( ratio · A_t )   # A_t 多用 rollout 时算好的常数
  → L_value  = (V_current(s_t) - G_t)²
  → L = L_policy + c_v·L_value - c_ent·H  → backward → step
```

**是不是「全靠回传实时算」**：  
- **Rollout**：前向算 `V_old`、`log π_old`；**advantage 多在 rollout 后一次性算好**（用已知的 `r_t` + `V_old`），不是每个 token 边生成边反传 RM。  
- **Train**：对 buffer 里样本 **再前向** 得 `V_current`、`π_current`，**这时才反传** 更新 policy + critic。  
- RM/Judge 分通常 **rollout 末或异步写回**；`G_t` / `A_t` 在 **reward 齐备后** 才算。

**手算：V 如何进 δ 再进 A**（3 token，`r_1=r_2=0, r_3=1`，`γ=1`，`λ=1`，`V(s_4)=0`）

| | `V(s_1)` | `V(s_2)` | `V(s_3)` | `δ_3` | `δ_2` | `δ_1` | `A_3` | `A_2` | `A_1` |
|---|----------|----------|----------|-------|-------|-------|-------|-------|-------|
| critic **还没学好** | 0.2 | 0.2 | 0.2 | 0.8 | −0.2 | 0 | 0.8 | 0.6 | 0.6 |
| critic **已学好** | 1 | 1 | 0 | 1 | −1 | 0 | 1 | 0 | 0 |

算式：`δ_3=1-V(s_3)`，`δ_2=0+V(s_3)-V(s_2)`，`δ_1=0+V(s_2)-V(s_1)`；`A_t=δ_t+δ_{t+1}+…`（`λ=1`）。  
**同一串 `r_t`，V 不同 → δ、A 全变**；policy 梯度乘的 **`A_t` 依赖 critic 质量**。所以 PPO 要 **同时训 π 和 V**，不能只训 policy。

**易混**：训 `V` 的 label 是 **`G_t`（真回报）**；进 **`δ_t` 的是 `V` 的预测`**。`A_t` 给 policy 当 **常数系数**（常用 rollout 时算好的），value loss 另有一条梯度训 critic。

**固定的是什么、变的是什么**（`γ`、`λ`、整条 `r_t` 序列一旦定下）：

- **`G_t`（MC 回报）只由 `r_t, r_{t+1}, …` 和 `γ` 决定**，与当前 `V` 无关。稀疏 RM、只在结尾 `r_T=R`、`γ=1` 时：从 **`s_1` 到 `s_{T-1}`** 的 **`G_t` 往往都等于 `R`**（后面还能碰到那 1 分）；**最后一步** 的 `G_T` 取 **0 还是 `R`** 取决于实现是「在 `s_T` 上拿奖」还是「拿奖后终止、无未来」——同一套 `r_t` 下 **全网标签一致**，batch 里每条轨迹各算各的。
- **`λ` 不改变 `G_t^MC`**；`λ` 只改 **GAE 怎么把 `δ` 合成 `A_t`**。所以 **critic 的回归目标（若用 MC `G_t`）在 rollout 后就钉死**；变的是 **`V_θ` 的预测** 越来越贴近这些常数标签。
- **`A_t` 不固定**：同一串 `r_t`，**`V_old` 不同 → `δ_t` 不同 → `A_t` 不同**（见 §2.5 手算表）。Policy 用的是 **`A_t`**，不是直接用 **`G_t`**。
- 若 value target 用 **λ-return**（`R_t^λ`，与 GAE 配套），目标里会含 **`V_old(s_{t+1})`**，在 **这一轮更新内** 对 **该 batch 的 advantage 计算** 通常仍把 **`V_old` 当常数**；但 **跨训练步** `V_old` 变了，λ-return 目标也会变——**仍不是「无监督」**，锚仍是 **`r_t`**。

面试一句：**「终局 RM 定下后，各步 MC 回报 `G_t` 是常数标签；`V` 去学它。`δ`/`A` 里用 `V` 当 baseline，所以 advantage 会随 critic 变，但真回报标签不随 `V` 变。」**

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
