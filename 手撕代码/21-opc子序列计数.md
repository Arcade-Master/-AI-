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

**一句话**：扫一遍，维护 `o` 个数和 `op` 对数；遇到 `c` 或 `o` 就接上所有 `op`。

### 状态

- `cnt_o`：当前前缀里 'o' 的个数；
- `cnt_op`：当前前缀里 "op" 子序列个数。

### 转移

| 字符 | 操作 |
|------|------|
| 'o'  | 先 `ans += cnt_op`（作 opo 的第三个 o），再 `cnt_o += 1` |
| 'p'  | `cnt_op += cnt_o` |
| 'c'  | `ans += cnt_op` |

每一步 mod 1e9+7。

**复杂度**：时间 O(n)，空间 O(1)。

**易错点**：

- 第三个字符是 'c' **或** 'o'：'c' 只加 ans；'o' 要**先** `ans += cnt_op`（作 opo 第三位），**再** `cnt_o += 1`（作后续 op 起点）；
- `i<j<k` 由下标严格递增保证，同一位置的 'o' 不会既当第三字符又当第一字符；
- 累加随时取模。

---

## 参考实现

```python
MOD = 10**9 + 7


def count_opc(s: str) -> int:
    cnt_o = cnt_op = ans = 0
    for ch in s:
        if ch == "o":
            ans = (ans + cnt_op) % MOD      # 作 opo 的第三个 o
            cnt_o = (cnt_o + 1) % MOD       # 作后续 op 的起点
        elif ch == "p":
            cnt_op = (cnt_op + cnt_o) % MOD
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

- 固定模式 "abc" 三字符子序列计数：同一框架，中间字符累加前缀，末字符加 ans。
- LC 不同路径（网格）：DP 计数，思路类似「前缀状态 + 当前字符扩展」。

---

## 面试怎么答（30 秒）

「线性扫：p 时把 cnt_o 并进 cnt_op；c 或 o 当第三位时 ans 加 cnt_op。全程 mod。」
