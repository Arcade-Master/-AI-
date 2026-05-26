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

**一句话**：二分答案 X，只保留 `a[i]<=X` 的元素，看能否拼出长 k 的 gcd 链。

### 判定 check(X)

过滤后从左扫，维护 `best[p]` = 以质因子 p 关联、当前最长链长。

对 `a[i] <= X`：

1. 分解 `a[i]` 得质因子集合（去重）；
2. `dp = 1 + max(best[p])`（没有则 dp=1）；
3. 用 `dp` 更新每个 `best[p]`；
4. 若 `dp >= k` 则可行。

相邻 gcd>1 等价于两项**共享至少一个质因子**，所以按质因子接力即可。

### 手推（样例）

`[2,4,6]` 相邻 gcd 均 >1，最大值 6；`[3,9,6]` 也合法但最大值 9 更大 → 答案 6。

### 二分

X 在 `[1, max(a)]` 上二分最小可行 X；若最大 X 仍不可行则 -1。

**复杂度**：单次 check O(n * sqrt(A))，二分 O(log max(a))，总约 O(n log A log max)。

**易错点**：

- 子序列必须保留下标顺序，所以按原序扫、不能排序数组；
- 质因子要去重（如 4 和 2 共享因子 2，不能算两次转移）；
- 不存在合法子序列时输出 -1，不是 0。

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
        d += 1 if d == 2 else 2  # 2 后只试奇数
    if x > 1:
        fs.append(x)
    return fs


def can(n: int, k: int, a: list[int], limit: int) -> bool:
    best: dict[int, int] = {}
    for v in a:
        if v > limit:
            continue
        dp = 1
        fs = factors(v)
        for p in fs:
            dp = max(dp, best.get(p, 0) + 1)
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

- 求最长 gcd 链（不限制最大值）：同一套 `best[p]` DP，取 max dp。
- k=2：只需存在相邻可用对，可更简单。

---

## 面试怎么答（30 秒）

「二分最大值 X。check 时按序扫，用质因子做 DP 接力：共享因子就能接在前项后面。dp>=k 即可。」
