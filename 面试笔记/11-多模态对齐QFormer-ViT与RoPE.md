# 11｜多模态对齐：对齐层、Q-Former、ViT 训练、RoPE、PPL 手撕

**建议阅读顺序（本篇内部）**：先弄清 **「对齐」到底对齐哪两件事（维度 + 语义）** → **Q-Former 在接口里占什么位置** → **其它对齐变体对照** → **像素到 LLM 的张量流** → **1D/2D RoPE** → **多阶段训练与学习率** → 最后 **PPL 与 mask 手撕**（依赖你已会 **交叉熵与 mask**，见 `01`）。

---

## 1. 对齐层：两件事

1. **维度**：视觉塔输出维度 `d_vis` → LLM 嵌入维度 `d_llm`。  
2. **语义**：让「图像片段」经可训练映射后，与「描述它的文字」在 **同一注意力空间里可交互」。

常 **冻结大 LLM**，先只训对齐模块控成本。

### 面试怎么答

「对齐层 = 线性/小网络把视觉 token 投到 LLM 维，并用图文对学语义对齐。」

---

## 2. Q-Former 是什么

来自 BLIP-2 一类：**可学习 query token** 通过 **cross-attention** 去读 **冻结 ViT 的 patch 特征**，再 **投影** 给冻结 LLM。

**动机**：patch 太多太碎；用 **固定条数的 query** 做 **信息瓶颈**，降序列长度与噪声。

### 面试怎么答

「Q-Former = 用少量 query 从视觉 token 里抽摘要，再接 LLM。」

### 2.1 前向数据流（能按步说出来就过关）

1. **冻结 ViT** 输出 patch token 序列 `T_vis`，长度可达 **数百～上千**。  
2. **可学习 query** `Q`（条数 `N_q` 远小于 patch 数）作为「槽位」，与文本无关地初始化或通过自注意先交互。  
3. **`Q` cross-attend to `T_vis`**：`Q` 为 Query，`T_vis` 为 K/V，得到 **`N_q` 条视觉摘要 token**（信息瓶颈）。  
4. **线性/MLP 投影** 到 LLM 词嵌入维，与 **文本 token 序列拼接**，进 **冻结或 LoRA 的 LLM**。  
5. **训练目标**：常见是 **只对文本段算 CE**（图像段 loss=0），或图文联合任务另设；梯度主要更新 **Q-Former + 投影**，ViT/LLM 按阶段解冻策略走（见第 6 节）。

---

## 3. 变体对照（背表不如背轴）

| 类型 | 特点 |
|------|------|
| 线性/MLP 投影 | 最简单，表达弱 |
| Q-Former / Perceiver | **定长**摘要，适合冻大模 |
| 小 Transformer 再接 | 模态内多交互一层再对齐 |
| LLM 侧 LoRA/Adapter | 更强，更易动语言能力 |
| 原生多模态 | 训练期就统一 tokenizer |

---

## 4. Vision token 到 LLM：按数据流说（与 §2.1 衔接）

把 **一张图** 变成 LLM 能 attend 的 **一段「伪文本 token」**，标准路径如下（数字可与具体模型不同，**顺序** 通用）：

1. **切 patch**：图像 `H×W` → 网格 patch（如 14×14 像素一块），每 patch 展平后经 **线性层** 得到 **patch embedding**，序列长度 `N_patch = (H/p)×(W/p)`。  
2. **ViT 编码**（常含 cls 或不用）：多层 self-attention，输出 **`T_vis`**，形状 `[N_patch, d_vis]`。  
3. **对齐模块**（§1～§3）：  
   - **线性投影**：`d_vis → d_llm`，条数仍为 `N_patch`；或  
   - **Q-Former**：压成 **`N_q` 条** 摘要 token（`N_q ≪ N_patch`）。  
4. **与文本拼接**：文本侧 tokenizer 得 `[BOS] … 问题 …` 的嵌入；在 **指定位置** 插入视觉 token（常见 **前缀**：`[视觉 tokens][文本 tokens]`）。  
5. **进 LLM**：一次 forward；**因果 mask** 下文本只能看 **左侧**（含已插入的视觉 token）。  
6. **Loss**：图文对话任务里，**图像 token 位置通常 mask 掉 CE**（`labels=-100`），**只对答案文本段** 算 loss，避免模型学「复述像素」。

**推理时**：图只 encode 一次，视觉 token 可 **缓存**；多轮对话若图不变，不必每轮重跑 ViT（工程优化点）。

### 面试怎么答

「图→patch→ViT→对齐压维→和文本 embed 拼接→只对文本段算 CE。」

---

## 5. 1D RoPE vs 2D RoPE

### 5.1 为什么多模态要提 2D

文本 token 天然是 **一维序列**（第 1 个词、第 2 个词…），**1D RoPE** 给每个位置一个 **沿序列的相位**（详见 **`07` §2**）。  
图像 patch 铺在 **二维网格** 上，有 **`(行 i, 列 j)`**。若强行 **按行展平** 成 1D 再 RoPE，**「正下方相邻」和「很远但序列相邻」** 的相位关系会 **扭曲**，空间归纳变弱。

### 5.2 2D RoPE 在干什么

给 Q/K 的维度块 **分配两套旋转**：一套随 **行坐标** 变，一套随 **列坐标** 变（或 **交错拼接** 到 head 维里）。注意力分数里同时感知 **Δrow、Δcol**，更像 **二维相对位置**。  
用于 **原生 ViT、部分多模态 LLM**；纯文本 LLM 仍用 1D。

### 5.3 若偷懒用 1D 展平

实现简单，但 **几何关系弱**；小图、弱空间推理任务有时 **够用**。面试可说：**「2D RoPE 为 patch 网格保留行列相对结构；展平 1D 是近似。」**

### 面试怎么答

「文本 1D 位置；图像 patch 用 2D RoPE 或展平近似；RoPE 本质是让点积带相对位置（见 07）。」

---

## 6. ViT 多阶段训练 & 学习率

**阶段逻辑**：先 **冻 ViT 训对齐**（学接口）→ **小学习率开 ViT 后几层/全 ViT**（吃域偏移）→ **必要时极小 LR 动 LLM 或 LoRA**（防语言塌）。

**LR 量级关系**：随机初始化的对齐层 **最大**；预训练 ViT **次之**；强 LLM **最小或先冻**。

数据量面试：**诚实说量级+来源+清洗比例**；涉密就讲 **阶段与方法**。

### 面试怎么答

「先训接口再动视觉再动语言；LR 随『已有先验强弱』递减。」

---

## 7. PPL 手撕与 mask 常见错

**定义**：对参与训练的 token 求平均 **NLL**，再 `exp`。

**分母必须是 mask.sum()**：`ignore_index` 位置 loss 常为 0，但若你用 `B*T` 平均会把 **padding 当有效样本** → **PPL 假低**。

**其它坑**：label 与 logits **错位**；instruction 段训练不算却评测算；pad id 与 ignore_index 不一致。

```python
import torch, torch.nn.functional as F

def perplexity(logits, labels, ignore_index=-100):
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    mask = flat_labels != ignore_index
    if not mask.any():
        return float("nan")
    nll = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=ignore_index)
    return torch.exp(nll.sum() / mask.float().sum()).item()
```

### 面试怎么答

「PPL=exp(平均 NLL)；平均只对有效 token；分母用 mask 个数。」

---

## 本篇小结

- 对齐 = **维 + 语义**；Q-Former = **摘要接口**。  
- 多阶段 = **先稳再动**。  
- PPL = **别被 padding 骗了分母**。
