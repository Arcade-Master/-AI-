# 07｜XOR 最佳搭档计数

**标签**：位运算 / 数位 DP / 困难  
**来源**：阿里 2026  
**状态**：可背

---

## 题面

x，a∈[la,ra]，b∈[lb,rb]。求有多少对 (a,b) 使 `x XOR a XOR b` 等于**所有合法对里的最大值**。

样例：x=0, a∈[1,2], b∈[0,2] → 最大 3，(1,2)(2,1) 共 2 对。

---

## 算法（两步）

1. **M 事先不知道**，从高位到低位试：这一位能不能是 1？能就 `M |= 1<<b`。
2. M 定死后，数位 DP **数**有多少 (a,b) 使 `x^a^b == M`。

样例：试 bit1 能要 1 → M=2；试 bit0 能要 1 → M=3；再数等于 3 的对数 → 2。

---

## 参考实现

（x、la、ra、lb、rb 都是**普通十进制整数**；DP 里用 `>> pos` 取其二进制第 pos 位。）

```python
from functools import lru_cache


def solve_xor(x: int, la: int, ra: int, lb: int, rb: int) -> int:
    # x, la, ra, lb, rb：题面给的十进制整数，例如 la=1, ra=2 表示 a 只能取 1 或 2
    TOP = 30  # 值 < 2^31，最高位下标 30

    def bit_at(num: int, pos: int) -> int:
        # 十进制 num 写成二进制后，第 pos 位是 0 还是 1（pos=0 是最低位）
        return (num >> pos) & 1

    def try_digit(d: int, lo: int, hi: int, pin_lo: bool, pin_hi: bool, pos: int):
        """
        正在构造一个十进制数，当前填它的第 pos 位二进制 d（0 或 1）。
        lo, hi：十进制下界、上界（如 a 的 la, ra）。
        pin_lo：True = 已填的高位和 lo 完全一样，这一位不能比 lo 该位更小。
        pin_hi：True = 已填的高位和 hi 完全一样，这一位不能比 hi 该位更大。
        返回 (能否选 d, 选完后的 pin_lo, 选完后的 pin_hi)。
        """
        if pin_lo and d < bit_at(lo, pos):
            return False, pin_lo, pin_hi
        if pin_hi and d > bit_at(hi, pos):
            return False, pin_lo, pin_hi
        # 这一位比 lo 大 → 后面不可能再小于 lo，pin_lo 解除
        new_pin_lo = pin_lo and (d == bit_at(lo, pos))
        # 这一位比 hi 小 → 后面不可能再大于 hi，pin_hi 解除
        new_pin_hi = pin_hi and (d == bit_at(hi, pos))
        return True, new_pin_lo, new_pin_hi

    @lru_cache(None)  # 记忆化：相同 (pos, 四个 pin, need, from_pos) 只算一次
    def exists(pos, pin_lo_a, pin_hi_a, pin_lo_b, pin_hi_b, need, from_pos):
        """
        从第 pos 位往下填 a、b，是否存在合法对使：
        x^a^b 在 bit[from_pos..TOP] 上与十进制 need 的二进制一致。
        低于 from_pos 的位暂不约束（贪心还没试到）。
        """
        if pos < 0:
            return True
        for da in (0, 1):  # a 在当前位的二进制
            ok, pla, pha = try_digit(da, la, ra, pin_lo_a, pin_hi_a, pos)
            if not ok:
                continue
            for db in (0, 1):  # b 在当前位的二进制
                ok, plb, phb = try_digit(db, lb, rb, pin_lo_b, pin_hi_b, pos)
                if not ok:
                    continue
                xor_bit = bit_at(x, pos) ^ da ^ db  # x^a^b 在这一位
                if pos >= from_pos and xor_bit != bit_at(need, pos):
                    continue
                if exists(pos - 1, pla, pha, plb, phb, need, from_pos):
                    return True
        return False

    # ① 求最大异或值 M（十进制），事先不知道，一位位试
    M = 0
    for b in range(TOP, -1, -1):
        trial = M | (1 << b)  # 试探：第 b 位能否为 1
        if exists(TOP, True, True, True, True, trial, b):
            M = trial  # 可以 → 把 M 的第 b 位钉成 1

    @lru_cache(None)
    def count(pos, pin_lo_a, pin_hi_a, pin_lo_b, pin_hi_b):
        """M 已固定，数有多少对 (a,b) 使十进制 x^a^b == M（每一位都要对上）"""
        if pos < 0:
            return 1
        res = 0
        for da in (0, 1):
            ok, pla, pha = try_digit(da, la, ra, pin_lo_a, pin_hi_a, pos)
            if not ok:
                continue
            for db in (0, 1):
                ok, plb, phb = try_digit(db, lb, rb, pin_lo_b, pin_hi_b, pos)
                if not ok:
                    continue
                if bit_at(x, pos) ^ da ^ db != bit_at(M, pos):
                    continue
                res += count(pos - 1, pla, pha, plb, phb)
        return res

    return count(TOP, True, True, True, True)


def main() -> None:
    t = int(input())
    for _ in range(t):
        x = int(input())
        la, ra = map(int, input().split())  # 十进制区间 [la, ra]
        lb, rb = map(int, input().split())  # 十进制区间 [lb, rb]
        print(solve_xor(x, la, ra, lb, rb))


if __name__ == "__main__":
    main()
```

---

## 面试 20 秒

「M 高位到低位贪心试 1；每位用 exists 判有没有 (a,b)。M 定后 count 数位 DP 数对数。pin_lo/pin_hi 表示前缀是否贴区间边界，贴就要满足该位不能越界。」
