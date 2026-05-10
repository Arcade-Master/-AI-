# 05｜SFT、对齐（RLHF/DPO/GRPO）、PEFT、灾难性遗忘、RLAIF

对应：**第一批**（PEFT、RLAIF、LoRA）；**第二批**（SFT loss/mask、instruction mask、label shift、teacher forcing、RLHF/DPO/GRPO、灾难性遗忘、LoRA 深入）。

---

## 1. SFT（Supervised Fine-Tuning）在优化什么？

**目标**：给定 (instruction, response) 或对话片段，最大化 **人类书写部分** 的似然。

**Loss 对哪些 token 算？**

- 常见做法：只对 **assistant / response** 位置算交叉熵；**user / system / instruction** 位置 loss mask 为 0。
- **原因**：我们不想用「教模型复述用户问题」的梯度浪费容量；希望参数更新集中在 **生成行为** 上。

**Label shift（右移）**：自回归训练里输入是 `y_0..y_{T-1}`，目标是预测 `y_1..y_T`；实现上常体现为 logits 与 labels **错位一位**，与 `nn.CrossEntropyLoss(ignore_index=...)` 配合。答清「**预测下一个 token**」即可。

**Teacher forcing**：训练时用 **真值历史** 作为 decoder 输入，而不是用上一步模型自己的输出。好处是 **稳定、可并行**；暴露偏差是推理时可能分布偏移（exposure bias），对话模型里有时用 scheduled sampling 等缓解（大模型 SFT 仍以 teacher forcing 为主流）。

---

## 2. RLHF vs DPO vs GRPO（答「差别 + 何时用」）

**RLHF（典型 PPO + RM）**：

- 训练 **Reward Model** 学人类偏好；
- 用 **PPO** 最大化 reward，同时 KL 约束策略不要离 SFT 模型太远。

**复杂点**：PPO 需 value head、advantage、clip、多超参、训练不稳、工程链路长。

**DPO（Direct Preference Optimization）**：

- 用偏好对 **直接优化策略**，跳过显式 RL 循环；损失重写为分类式目标，**更稳定、实现简单**。
- 代价：对数据质量与参考模型绑定方式敏感；长链推理等场景有时仍会用 RL 类方法探索。

**GRPO（Group Relative Policy Optimization，DeepSeek 等推理模型路线中常见）**：

- 一组采样里 **用组内相对 reward 构造优势**，减少 critic/value network 依赖，适合 **可验证奖励**（数学、代码执行结果）的推理强化。
- 面试答：**更适配「规则/执行可打分」的推理任务**，降低传统 actor-critic 复杂度。

**Reward Model 是什么**：学一个标量函数 r(prompt, response)，在偏好数据上满足 `r(y_w) > r(y_l)`；RL 阶段用它作 **可微或可采样的信号**（实现上常是「策略生成 → RM 打分 → 策略梯度」）。

---

## 3. RLAIF（与第一批衔接）

用 **强模型** 代替人类标偏好，后续管线可与 RLHF 类似（RM + PPO）或走向宪法 AI 等 **规则化 AI 反馈**。关键是 **反馈质量与偏差**：AI 裁判会放大自身偏见，需要校验集与规则约束。

---

## 4. LoRA：为什么低秩有效、rank 怎么选、为什么省显存

**原理**：大矩阵微调更新 ΔW 往往 **近似低秩**（任务只在少数子方向偏离预训练）；显式约束 `ΔW = B A`（rank=r）是归纳偏置。

**rank**：小任务 r=4～8；通用 SFT 常 8～64；**越大容量越大但过拟合与显存上升**。没有银弹，靠 **dev 集 + 下游指标** 选。

**省显存原因**：**不更新、不存全量 W 的 optimizer 状态**（W 冻结）；只优化 B、A 的小参数量；反向时 **可合并 LoRA 到一次矩阵乘**（推理合并）进一步加速。

**与遗忘关系**：见下一节。

---

## 5. 灾难性遗忘（Catastrophic Forgetting）

**现象**：强 SFT 或持续学习新域后，**通用能力或旧任务**明显下降。

**为何 SFT 易诱发**：小数据强优化会把表示 **拉向窄分布**，覆盖预训练「宽知识」所需的流形。

**缓解**：

- 数据：**混合通用语料**、高质量多样化指令；
- 优化：较小 lr、早停、正则；
- 算法：**KL to reference**（RLHF 里常见）、**replay**、**LoRA/Adapter** 限制可塑自由度；
- 评测：不仅看新任务，也看 **通用 benchmark**。

**LoRA 为何能减轻**：**冻结主干**，更新 confined 在低秩子空间，**减少对基底表示的全局改写**；但不是万能，强数据+高 rank 仍会忘。

---

## 小结

- SFT 的考点一半是 **mask 与因果**，一半是 **训练-推理一致性**（label shift、teacher forcing）。
- RLHF / DPO / GRPO 区分 **优化范式与工程复杂度**，能举 **适用场景** 比背定义分高。
- 遗忘是 **表示漂移** 问题；缓解是 **数据 + 优化约束 + 参数效率** 的组合拳。

**更深一层的 PPO / GRPO 计算、Value、REINFORCE 方差与「REINFORCE++」口径**：见 **[12-RL进阶REINFORCE-PPO-GRPO与价值函数](./12-RL进阶REINFORCE-PPO-GRPO与价值函数.md)**。
