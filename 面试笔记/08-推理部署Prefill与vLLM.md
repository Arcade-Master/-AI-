# 08｜Prefill/Decode、vLLM、Continuous Batching、PagedAttention

对应：**面试题2** 第 26、36～37 节与 FlashAttention（04 篇已讲原理，此处侧重 **服务侧**）。

---

## 1. Prefill vs Decode：算力与带宽

**Prefill（提示阶段）**

- 一次性处理整段 prompt，**矩阵大、并行度高**，通常 **计算密集（compute-bound）** 占比高。

**Decode（逐 token 生成）**

- 每步序列长度 +1，但 **有效并行矩阵乘变小**；大量时间花在 **读权重、读 KV Cache、写回**，常 **内存带宽 bound**。
- **难吃满 GPU**：小 kernel、低 occupancy、同步多。

**谁更耗算力**：通常 **prefill FLOPs 高**（整段 self-attention）；**decode 步数多且带宽受限**，总延迟里 decode 常占大头。

---

## 2. vLLM：解决什么问题？

**核心痛点**：服务多用户时，batch 内 **序列长短不一**，传统静态 padding **浪费算力与显存**；KV Cache **非连续分配** 导致碎片与预留过大。

**PagedAttention（思想）**

- 借鉴 OS **虚拟内存 + 物理页**：逻辑 KV 序列映射到 **非连续物理块**；
- 按需分配块，减少 **预留与碎片**，提高 **batch 可容纳量**。

**Continuous batching（动态批）**

- 某条序列 **EOS 提前结束** 即腾出 slot，**新请求可插入** 正在运行的 step，无需等整批同长；
- 提高 **吞吐（tokens/s）** 与 **GPU 利用率**。

**为何吞吐随 batch 增加（在一定范围内）**

- 更大 batch **摊薄 kernel launch 与权重读取** 的固定开销；
- 直到 **带宽或显存** 饱和，再继续增大反而 **延迟上升** 或 OOM。

---

## 3. 与 FlashAttention 的关系

vLLM 等系统常 **融合 FlashAttention kernel** 做 attention 本体加速，PagedAttention 解决 **KV 存储与调度**；两者层级不同，可叠加。

---

## 小结

- Prefill **算**，Decode **读**——延迟分析常用这套框架。
- vLLM ≈ **Paged KV + continuous batching + 高效 CUDA kernel** 的系统工程组合。
