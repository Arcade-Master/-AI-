# 21｜opc 子序列计数

**标签**：字符串 / 计数  
**来源**：截图题  
**难度**：简单  
**状态**：可背

---

## 题面

给定长 n 字符串 s，字符仅 `o`、`p`、`c`。  
统计三元组下标 `(i,j,k)`（1<=i<j<k<=n）满足：

- s[i]='o'，s[j]='p'；
- s[k] 为 'c' **或** 'o'（即子序列 "opc" 或 "opo"）。

输出个数 mod 1e9+7。

**输入**：多组 t；每组 n 和 s。  
**约束**：t<=2e5，n 总和 <=4e5。

---

## 思路

### 要算什么

保序选三个位置 i<j<k，形状像 **opc** 或 **opo** 的子序列个数。

### 分几步

从左扫 s，维护「前面有多少个 o」「前面有多少个 op」，遇到能当第三位的字符就把前面的 op 全接上。

### 状态（白话）

| 变量 | 含义 |
|------|------|
| `cnt_o` | 当前位置**之前**（含当前若作第一字符）可当地一个 o 的个数 |
| `cnt_op` | 当前位置之前，子序列 **"op"** 的个数（选一个 o 再选一个 p） |
| `ans` | 已完成的 **opc / opo** 三元组个数 |

### 手推（s = "opo"）

| 位置 | 字符 | 操作 | cnt_o | cnt_op | ans |
|------|------|------|-------|--------|-----|
| 1 | o | 先 ans+=0；再 cnt_o=1 | 1 | 0 | 0 |
| 2 | p | cnt_op += cnt_o → 1 | 1 | 1 | 0 |
| 3 | o | ans += cnt_op → 1（opo）；cnt_o=2 | 2 | 1 | **1** |

答案 1：唯一三元组 (1,2,3) 组成 "opo"。

若第三位是 `c`：只 `ans += cnt_op`，不增加 cnt_o。

### 转移（对应代码顺序）

| 字符 | 操作 |
|------|------|
| 'o' | **先** `ans += cnt_op`（作 opo 的第三个 o），**再** `cnt_o += 1` |
| 'p' | `cnt_op += cnt_o`（每个旧 o 与当前 p 组成 op） |
| 'c' | `ans += cnt_op`（opc） |

**复杂度**：O(n)，O(1) 空间。

**易错点**：

- 第三个是 c 或 o，但 **o 的处理顺序** 不能反（否则漏 opo 或重复计）；
- `i<j<k` 由严格下标保证；
- 全程 mod 1e9+7。

---

## 参考实现

```python
MOD = 10**9 + 7


def count_opc(s: str) -> int:
    cnt_o = cnt_op = ans = 0
    for ch in s:
        if ch == "o":
            ans = (ans + cnt_op) % MOD      # 第三位 o：接前面的 op
            cnt_o = (cnt_o + 1) % MOD       # 再当作后面 op 的起点
        elif ch == "p":
            cnt_op = (cnt_op + cnt_o) % MOD  # 每个 o 与当前 p 配成 op
        else:  # 'c'
            ans = (ans + cnt_op) % MOD
    return ans


def main() -> None:
    t = int(input())
    for _ in range(t):
        n = int(input())
        s = input().strip()
        print(count_opc(s))


if __name__ == "__main__":
    main()
```

---

## 变体 / 追问

- 固定模式 "abc"：同一框架，中间字符扩展前缀，末字符加 ans。

---

## 面试怎么答（30 秒）

「扫一遍：p 时 cnt_op+=cnt_o；c 或 o（第三位）时 ans+=cnt_op；o 还要先加 ans 再 cnt_o+=1。」
