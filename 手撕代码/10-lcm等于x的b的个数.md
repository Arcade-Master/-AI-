# 10｜lcm(a,b)=x 的 b 的个数

**标签**：数论 / 质因数分解  
**来源**：代码题截图 `已处理/10-lcm等于x-题面.png`  
**状态**：可背

---

## 题面

给定正整数 x、a，求正整数 b 的个数，使得 lcm(a, b) = x。

**输入**：多测 T；每组 x, a。1 <= x,a <= 1e9。

---

## 思路

### 1. 必要条件

lcm(a,b)=x 则 a|x 且 b|x。若 x % a != 0，答案 0。

### 2. 质因子分解

设 x = prod p^e，a = prod p^f（f <= e）。

对每个质数 p：

- lcm 在该位指数 = max(f, g) = e，其中 g 是 b 在该位的指数；
- 若 f < e：b 在该位必须是 e（1 种）；
- 若 f == e：b 在该位可以是 0..e（e+1 种）。

答案 = 各质数方案数之积。

### 3. 实现

分解 x（sqrt(x)），同时数 a 中该质因子指数。

**复杂度**：O(sqrt(x)) 每组。

---

## 参考实现

```python
def count_b(x: int, a: int) -> int:
    if x % a:
        return 0
    ans = 1
    t = x
    p = 2
    while p * p <= t:
        if t % p == 0:
            e = 0
            while t % p == 0:
                e += 1
                t //= p
            f = 0
            aa = a
            while aa % p == 0:
                f += 1
                aa //= p
            if f < e:
                pass
            else:
                ans *= e + 1
        p += 1 if p == 2 else 2
    if t > 1:
        e = 1
        f = 0
        aa = a
        while aa % t == 0:
            f += 1
            aa //= t
        if f == e:
            ans *= e + 1
    return ans


def main() -> None:
    t = int(input())
    for _ in range(t):
        x, a = map(int, input().split())
        print(count_b(x, a))


if __name__ == "__main__":
    main()
```

---

## 面试怎么答（30 秒）

「先判 x%a==0。分解 x，每位质因子：a 指数小于 e 则 b 该位只能 e；等于 e 则 b 该位 0..e 共 e+1 种。乘起来。」
