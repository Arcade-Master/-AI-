# 11｜对齐层、Q-Former、Vision Token、2D/1D RoPE、ViT 多阶段训练、PPL 与 mask 手撕

对应：**`面试答题.md` 新增**（第 8～17 行：对齐层与 Q-Former、2D RoPE、vision token、Attention 复杂度、ViT 训练与学习率、PPL+mask 手撕）。Attention 复杂度在 **`02` 第 10 节** 有并行表述，本篇侧重 **多模态与训练侧**。

---

## 1. 「对齐层」在对齐什么？

多模态里通常有两类「不对齐」：

1. **模态表示空间不同**：图像编码器（ViT/CLIP visual）输出在 **视觉嵌入空间**；LLM 词嵌入在 **离散 token 的语义空间**。
2. **序列接口不同**：图像是一串 patch token；LLM 期望 **与文本同维度的连续向量序列** 接在输入 embedding 之后。

**对齐层（alignment / connector / projector）** 做的事可以概括为：

- **维度对齐**：`d_vision → d_LLM`（线性、MLP、轻量 Transformer 等）。
- **语义对齐**（若可训练）：用 **图文配对数据** 让视觉 token 经映射后，与 **描述该图的文本** 在同一 LLM 输入空间里「可一起被注意力处理」——本质是 **可学习的跨模态接口**，常配合 **冻结或慢训** 大 LLM 以控成本。

---

## 2. Q-Former 是什么？

出自 **BLIP-2** 一类路线：在 **冻结的视觉编码器** 与 **冻结的 LLM** 之间插一层 **可学习的 Query Transformer**。

**机制要点**：

- 引入一组 **可学习的 query token**（数量远少于视觉 patch 数）。
- Query 通过 **cross-attention** 去读 **冻结 ViT 输出的视觉 token**（视觉侧 K/V，query 侧为 learnable queries）。
- 输出再经 **投影** 喂给 LLM。

**直觉**：视觉 patch 又碎又冗余；Q-Former 用 **瓶颈式查询** 把信息 **压缩成固定条数的「视觉摘要 token」**，降低 LLM 侧序列长度与噪声，同时 **只训练 Q-Former + 投影** 就能接上冻结大模型。

---

## 3. 还有哪些「对齐层」变体？（面试对比表）

| 路线 | 做法概要 | 特点 |
|------|----------|------|
| **Linear / MLP projector**（如 LLaVA 早期） | ViT patch 序列直接线性映射到 LLM 维 | 极简、快；表达能力弱于带交互的模块 |
| **Q-Former / Perceiver Resampler** | 可学习 query 从视觉 token **抽取**信息 | **定长**视觉 token，省长度；适合冻结大模型 |
| **C-Former / 额外轻量 Transformer** | 在视觉 token 上做几层 self-attn 再投影 | 增强 **模态内** 上下文再对齐 |
| **Adapter / LoRA 在 LLM 入口** | 不只映射视觉，还在 LLM **浅层**插适配器 | 更强但更易动到「语言能力」、调参复杂 |
| **统一 Embedding（原生多模态）** | 从训练起就共享 tokenizer/embedding | 非「后接」对齐，而是 **架构级** 融合 |

面试答法：先定义 **「瓶颈查询 + cross-attn」= Q-Former 系**；再说 **「线性投影」= 最简对齐**；最后提 **「原生多模态」是长期方向**。

---

## 4. Vision token 进入对齐层前后：常见流水线

以「ViT + 对齐 + LLM」为例，**从像素到 LLM 输入** 常经历：

1. **Patch embed**：图像切块 → 线性嵌入为 patch token 序列。
2. **2D/1D 位置编码**（若有）：让 patch 知道网格位置。
3. **（可选）额外 ViT block**：得到上下文化的视觉表示。
4. **对齐层**：  
   - Q-Former：**queries × 视觉 K/V** → 得到固定条数「对齐 token」；  
   - 或 MLP：**逐 token 或整段池化后广播**（视架构而定）。
5. **与文本 token 拼接**：`<image_start> + 对齐后视觉 tokens + <image_end> + 文本 tokens` 送入 LLM。
6. **训练时 mask**：图像区是否参与 **语言建模 loss** 依任务而定（常 **不算 CE** 在图像 token 上，只算文本生成段）。

**面试追问「做了哪些操作」**：按上面顺序 **点名模块 + 张量形状变化（L_vis、D）→（L_q、D_LLM）**。

---

## 5. 2D RoPE 是什么？和 1D RoPE 区别？

**1D RoPE（语言）**：位置索引 **一条链** `t = 0,1,…,L-1`；在复数/二维旋转意义下给 **每个 token 一个相位**，使 attention 内积 **只依赖相对距离 (m-n)**。

**2D RoPE（视觉常见）**：patch 有 **(row, col)** 二维网格坐标。做法可以是：

- **两个方向各一套旋转**（行相位 + 列相位），再 **组合** 到 Q/K 的不同维度对上；或
- 把 `(h,w)` 展平成序列仍用 1D RoPE（**退化为 1D**，但几何上不等价于真正的 2D 归纳偏置）。

**区别本质**：1D 假设 **全序**；2D 显式利用 **平面邻域结构**，对 **空间关系敏感** 的任务（检测、高分辨率布局）更自然。是否用 2D RoPE 取决于 **视觉编码器是否仍用类 Transformer 注意力** 以及实现成本。

**RoPE 为什么起作用（与 07 篇呼应）**：把「绝对位置」编码成 **旋转**，使 **点积自然出现相对位置依赖**；比可加绝对位置 embedding 更易 **外推讨论**（尽管外推仍有限）。

---

## 6. 改 ViT 后如何「多阶段训练」？数据量怎么说？

典型 **分阶段** 不是玄学，而是为了 **稳定与控遗忘**：

1. **阶段 A：冻结 ViT，只训对齐层（+ 可选 LoRA）**  
   - 让接口先学会 **把视觉语义送进 LLM**；避免一开始大动 ViT 破坏预训练视觉特征。  
   - 数据：**高质量图文对**（指令跟随、描述、QA），规模从 **百万级图文对** 到更大取决于目标；面试可答「**先小后大、先高质量再扩域**」。

2. **阶段 B：解冻 ViT 后几层或全 ViT，小学习率**  
   - 适应 **域偏移**（如工业图纸、医学影像）。  
   - 数据：加入 **域内** 数据；比阶段 A 更依赖 **标注或伪标签质量**。

3. **阶段 C（可选）：联合微调 LLM**  
   - 用 **极小 LR + 正则 / LoRA**，防止 **语言能力塌**。

**「用了多少数据」**：面试诚实答 **量级 + 来源 + 清洗比例**；若涉密，答 **「阶段 A xM 对，阶段 B 加 y% 域内」** 的方法论即可。

---

## 7. ViT、对齐层、LLM 的学习率一般怎么设？

**经验原则**（需用验证集调，不是死数）：

| 模块 | 相对 LR | 原因 |
|------|---------|------|
| **随机初始化对齐层** | **最高**（如 1e-3 ~ 1e-4 量级谈法） | 从头学映射，需要大步长起步 |
| **ViT（ImageNet 预训练）** | **中等偏小**（如对齐的 1/10～1/3） | 已有好特征，大 LR 易 **域崩溃** |
| **LLM（预训练强基座）** | **最小** 或 **先冻结** | 防 **灾难性遗忘**、输出分布漂移 |

常用技巧：**layer-wise decay**（越靠近输入 LR 越小）、**warmup + cosine decay**、**global batch 放大时线性放大 LR**（有上限）。

---

## 8. 手撕：PPL（困惑度）怎么算？mask 代码哪里最容易错？

### PPL 定义

语言模型对 token 序列 `x_1…x_T` 的（平均）负对数似然：

```
NLL = - (1/N) Σ log P(x_i | x_<i)   （只对「要预测的目标 token」求和）
PPL = exp(NLL)
```

**只对有效 token 平均**：padding、instruction（若配置为不训练）应 **不参与平均**，否则 PPL 被 **稀释或拉偏**。

### 正确实现骨架（PyTorch 风格）

```python
import torch
import torch.nn.functional as F

def perplexity(logits, labels, ignore_index=-100):
    """
    logits: (B, T, V)
    labels: (B, T)  已右移：labels[:, t] = 要预测的 token（与 logits[:, t] 对齐）
    """
    # 展平 token 维
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    mask = flat_labels != ignore_index
    if not mask.any():
        return float("nan")
    # reduction="none"：ignore_index 位置 loss 为 0，但平均必须用 mask.sum() 作分母
    nll = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=ignore_index)
    mean_nll = nll.sum() / mask.float().sum()
    return torch.exp(mean_nll).item()
```

### 常见 **mask bug**

1. **平均的分母错了**：用 `T` 或 `B*T` 而不是 **`mask.sum()`** → padding 稀释 NLL，**PPL 虚低**。  
2. **ignore_index 与 padding id 不一致**：label 里 pad 是 0，但 `ignore_index=-100` 且忘记把 pad 改成 -100。  
3. **label 未与 logits 对齐**：自回归应对齐 **预测位置**；常见是 **shift 错一位** 或 **多减一次 bos**。  
4. **对 instruction 不算 loss 却在算 PPL 时算了**：评测口径与训练不一致。  
5. **在 softmax 前先对整行含 pad 算 CE**：应对 **每个位置** 用正确 label；整 batch 一起 mean 前必须 mask。

---

## 小结

- **对齐层** = 维度和语义接口；**Q-Former** = query 瓶颈 + cross-attn 抽视觉摘要。  
- **2D RoPE** 显式编码网格；**1D** 编码序列序。  
- **多阶段训练** = 先训接口再微调视觉再动 LLM，核心是 **稳定性与防遗忘**。  
- **PPL** = `exp(平均 NLL)`，**mask 与分母**是手撕题高频扣分点。
