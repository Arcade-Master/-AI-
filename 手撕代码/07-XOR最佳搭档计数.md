# 07｜XOR 最佳搭档计数

**标签**：位运算 / 数位 DP / 困难  
**来源**：代码题截图 `已处理/07-XOR最佳搭档-题面.png`（阿里 2026）  
**状态**：可背

---

## 题面

给定 x 与 a 的区间 [la, ra]、b 的区间 [lb, rb]。

**最佳搭档**：在所有 (a,b) 合法对中，使 `x XOR a XOR b` 达到**最大值**的那些对。

求最佳搭档的对数。

**输入**：多测 T；每组 x, la ra, lb rb。范围 < 2^31。  
**样例**：x=0, a∈[1,2], b∈[0,2] → 最大异或 3，有 (1,2)(2,1) 共 2 对。

---

## 思路

**全文脉络**：

```
1. 从高到低试每一位，贪心求最大异或值 M（数位 DP 判「能否达到」）
2. 再数位 DP 数有多少 (a,b) 满足 x^a^b 的每一位都等于 M
```

### 1. 贪心求 M

从高位 bit=30 到 0：若存在合法 (a,b) 使 `x^a^b` 在已固定位之上与 `M|(1<<b)` 一致，且当前位能为 1，则 `M |= 1<<b`。

「是否存在」用 DP：`feasible(pos, 四维 tight, trial, min_pos)`，只约束 pos >= min_pos 的位必须等于 trial 的该位。

### 2. 计数

M 固定后，同样从高到低 DP，要求每一位 `((x>>pos)&1) ^ ab ^ bb == (M>>pos)&1`，累加方案数。

### 3. 区间 tight 写法（易错）

不要用 `range(lo, hi+1)` 当 la 位 > ra 位时为空；应对 ab∈{0,1} 逐个判断：

```python
if tla and ab < ((la >> pos) & 1): continue
if tra and ab > ((ra >> pos) & 1): continue
```

**复杂度**：O(log V) 位 * 16 状态，每组 O(31*16) 常数小。

---

## 参考实现

```python
from functools import lru_cache


def solve_xor(x: int, la: int, ra: int, lb: int, rb: int) -> int:
    @lru_cache(None)
    def feasible(pos, tla, tra, tlb, trb, trial, min_pos):
        if pos < 0:
            return True
        for ab in (0, 1):
            if tla and ab < ((la >> pos) & 1):
                continue
            if tra and ab > ((ra >> pos) & 1):
                continue
            ntla = tla and ab == ((la >> pos) & 1)
            ntra = tra and ab == ((ra >> pos) & 1)
            for bb in (0, 1):
                if tlb and bb < ((lb >> pos) & 1):
                    continue
                if trb and bb > ((rb >> pos) & 1):
                    continue
                ntlb = tlb and bb == ((lb >> pos) & 1)
                ntrb = trb and bb == ((rb >> pos) & 1)
                r = ((x >> pos) & 1) ^ ab ^ bb
                if pos >= min_pos and r != ((trial >> pos) & 1):
                    continue
                if feasible(pos - 1, ntla, ntra, ntlb, ntrb, trial, min_pos):
                    return True
        return False

    M = 0
    for b in range(30, -1, -1):
        if feasible(30, True, True, True, True, M | (1 << b), b):
            M |= 1 << b

    @lru_cache(None)
    def count(pos, tla, tra, tlb, trb):
        if pos < 0:
            return 1
        res = 0
        for ab in (0, 1):
            if tla and ab < ((la >> pos) & 1):
                continue
            if tra and ab > ((ra >> pos) & 1):
                continue
            ntla = tla and ab == ((la >> pos) & 1)
            ntra = tra and ab == ((ra >> pos) & 1)
            for bb in (0, 1):
                if tlb and bb < ((lb >> pos) & 1):
                    continue
                if trb and bb > ((rb >> pos) & 1):
                    continue
                ntlb = tlb and bb == ((lb >> pos) & 1)
                ntrb = trb and bb == ((rb >> pos) & 1)
                if (((x >> pos) & 1) ^ ab ^ bb) != ((M >> pos) & 1):
                    continue
                res += count(pos - 1, ntla, ntra, ntlb, ntrb)
        return res

    return count(30, True, True, True, True)


def main() -> None:
    t = int(input())
    for _ in range(t):
        x = int(input())
        la, ra = map(int, input().split())
        lb, rb = map(int, input().split())
        print(solve_xor(x, la, ra, lb, rb))


if __name__ == "__main__":
    main()
```

---

## 面试怎么答（30 秒）

「先按位贪心求最大 x^a^b，再用四维 tight 的数位 DP 计数达到最大值的 (a,b) 对数。区间判断用 0/1 枚举+tight 剪枝，别用错误的 lo-hi 区间。」
