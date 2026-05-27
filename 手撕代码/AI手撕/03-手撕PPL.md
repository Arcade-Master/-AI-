# 03｜手撕 PPL（困惑度）

**标签**：语言模型 / 评测 / PyTorch  
**关联笔记**：[`面试笔记/07-长上下文位置编码与量化评测.md`](../../面试笔记/07-长上下文位置编码与量化评测.md)（PPL 含义与局限）、[`面试笔记/11-多模态对齐…`](../../面试笔记/11-多模态对齐QFormer-ViT与RoPE.md) §7（mask 常见错）  
**难度**：简单（公式短，坑在分母与 label 对齐）  
**状态**：可背

---

## 要手撕什么

| 输入 | 输出 |
|------|------|
| logits `(B, T, V)`、labels `(B, T)` | 标量 PPL |

**定义**：对 **参与训练/评测的有效 token** 求平均 **NLL（负对数似然）**，再取 `exp`：

```
PPL = exp( mean( -log p(y_t | context) ) )
```

PyTorch 里常用 `cross_entropy(..., reduction="none")` 逐 token 算 NLL，再对有效位求平均。

---

## 1. 分母必须是有效 token 数

`ignore_index`（常见 `-100`）位置的 loss 常为 0，但若用 **`B*T` 或 `labels.numel()`** 做分母，会把 **padding / 不算 loss 的 instruction 段** 也算进平均 → **PPL 假低**。

正确：`nll.sum() / mask.float().sum()`，其中 `mask = labels != ignore_index`。

---

## 2. 其它常见坑

- **label 与 logits 错位**：因果 LM 常 `labels = input_ids` 左移一位，或 `logits[..., :-1]` 对 `labels[..., 1:]`。
- **instruction 段**：训练时 `labels=-100` 不算 loss；评测时若没同样 mask，PPL 不可比。
- **pad id 与 ignore_index 不一致**：padding 位置必须进 mask。

---

## 参考实现（PyTorch）

```python
import torch
import torch.nn.functional as F


def perplexity(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    logits: (B, T, V)
    labels: (B, T)，无效位置为 ignore_index
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    mask = flat_labels != ignore_index
    if not mask.any():
        return float("nan")
    nll = F.cross_entropy(
        flat_logits, flat_labels, reduction="none", ignore_index=ignore_index
    )
    return torch.exp(nll.sum() / mask.float().sum()).item()


if __name__ == "__main__":
    B, T, V = 2, 4, 8
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[0, -1] = -100  # 模拟 padding / 不算 loss
    print("PPL:", perplexity(logits, labels))
```

---

## 易错点

- 分母用 `mask.sum()`，不是 `labels.numel()`。
- `cross_entropy` 的 `ignore_index` 与 `mask` 逻辑一致。
- 多模态 / SFT 场景：图像 token、system prompt 段是否在 labels 里标 `-100`，训练和评测要同一套规则。

---

## 面试怎么答（30 秒）

「PPL = exp(有效 token 上的平均 NLL)。实现上 flatten 后 `cross_entropy(reduction='none')`，分母只数 `labels != ignore_index` 的个数；别用 batch×seq 当分母，否则 padding 会把 PPL 拉假低。」
