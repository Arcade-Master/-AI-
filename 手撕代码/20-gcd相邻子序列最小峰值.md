# 20｜gcd 相邻子序列最小峰值

**标签**：二分 / DP / 数论  
**来源**：截图题  
**难度**：困难  
**状态**：可背

---

## 题面

给定长 n 数组 a，选长 k 的子序列（下标严格递增），要求**相邻两项 gcd > 1**。  
在所有合法子序列中，让子序列**最大值尽量小**；输出该最小可能的最大值，不存在输出 -1。

**约束**：2<=n<=2e5，1<=k<=n，1<=a_i<=1e6。  
**输入**：一行 n,k，一行 a（单组，无 T）。

**样例**：n=5,k=3,a=[2,4,3,9,6] → **6**（子序列 [2,4,6]）。

---

## 思路

### 要算什么

选 k 个**保序**下标，相邻 gcd>1，让子序列里**最大的那个数尽量小**。

### 分几步

1. **二分**答案 X：只许用 `<= X` 的数，看能不能拼出长 k 的链；
2. **check(X)**：按原序扫，用质因子做「最长 gcd 链」DP。

### check(X)：`best[p]` 是什么

按**原数组顺序**扫（不能排序，子序列要保留下标递增）：

`best[p]` = 到当前位置为止，**以质因子 p 结尾**的、只使用 `<=X` 的数、的最长链长度。

### 手推（X=6，能否凑 k=3）

数组 `[2, 4, 3, 9, 6]`，只用 `<=6` 的：2,4,3,6（9 跳过）。

| 扫到 | 质因子 | 接在谁后面 | 链长 dp |
|------|--------|------------|---------|
| 2 | {2} | 无 | 1，best[2]=1 |
| 4 | {2} | best[2]+1 | 2，best[2]=2 |
| 3 | {3} | 无 | 1 |
| 6 | {2,3} | best[2]+1 或 best[3]+1 | max=3 |

dp=3 >= k → **X=6 可行**。而 X=5 时 6 不能用，最长链 <3 → 不可行。二分得答案 **6**。

### 转移（对照代码）

对当前值 v（且 v<=X）：

1. 分解 v 得**不同**质因子集合 fs（4 只算因子 2 一次，避免同数接两次算两段）；
2. `dp = 1 + max(best[p])`（p 在 fs 里；没有则 dp=1）；
3. 若 `dp >= k` → 可行；
4. 用 dp 更新每个 `best[p]`。

**为何 gcd>1 等价于共享质因子**：gcd(a,b)>1 当且仅当 a、b 有公共质因子，链可以按「最后一个数的某个因子 p」接力。

### 二分

在 `[1, max(a)]` 上二分最小可行 X；全体都不可行则 -1。

**复杂度**：单次 check O(n * sqrt(A))，二分 O(log max(a))。

**易错点**：

- 必须按**原下标顺序**扫；
- 质因子**去重**；
- 不存在输出 -1。

---

## 参考实现

```python
def factors(x: int) -> list[int]:
    fs = []
    d = 2
    while d * d <= x:
        if x % d == 0:
            fs.append(d)
            while x % d == 0:
                x //= d
        d += 1 if d == 2 else 2
    if x > 1:
        fs.append(x)
    return fs


def can(n: int, k: int, a: list[int], limit: int) -> bool:
    best: dict[int, int] = {}  # best[p]：以质因子 p 结尾的最长链
    for v in a:
        if v > limit:
            continue
        dp = 1
        fs = factors(v)
        for p in fs:
            dp = max(dp, best.get(p, 0) + 1)  # 接在曾以 p 结尾的链后面
        if dp >= k:
            return True
        for p in fs:
            best[p] = max(best.get(p, 0), dp)
    return False


def min_max_value(n: int, k: int, a: list[int]) -> int:
    lo, hi = 1, max(a)
    if not can(n, k, a, hi):
        return -1
    while lo < hi:
        mid = (lo + hi) // 2
        if can(n, k, a, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> None:
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    print(min_max_value(n, k, a))


if __name__ == "__main__":
    main()
```

---

## 变体 / 追问

- 只求最长 gcd 链：同一套 best[p]，取 max dp。
- k=2：存在相邻（保序）共用因子即可。

---

## 面试怎么答（30 秒）

「二分最大值 X。check 按序扫，best[p] 记以因子 p 结尾的最长链，共享因子就能接。dp>=k 则可行。」
