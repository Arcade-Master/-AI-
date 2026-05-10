# 14｜推理框架、asyncio、进程线程、强制 JSON

**建议阅读顺序（本篇内部）**：先 **列举推理框架各自解决什么** → **SGLang 与 vLLM 的侧重点（别背谁绝对快）** → **asyncio 解决的是哪类并发** → **PDF 解析用多进程还是多线程** → **强制 JSON 的分层手段**。

与 **`08`** 分工：`08` 讲 **Prefill/Decode 与 vLLM 核心思想**；本篇补 **框架对比、Python 工程、JSON**。

---

## 1. 常见推理框架（口述名单即可）

**vLLM**：PagedAttention + continuous batching，**吞吐**友好。  
**SGLang**：强调 **前缀复用（RadixAttention）**、**结构化生成/调度**，多轮共享 system 时 **延迟**常有好故事。  
另有 **TensorRT-LLM、TGI、llama.cpp** 等按场景选用。

### 面试怎么答

「vLLM 偏服务吞吐；SGLang 偏复杂控制流与前缀缓存；都可叠 Flash kernel。」

---

## 2. SGLang vs vLLM：别写成拉踩小作文

**可陈述的设计差异**：SGLang 的 **RadixAttention / radix cache** 把 **不同请求的共享前缀** 组织成 **树（trie）** 节点：同一 system prompt、同一文档前缀在 **逻辑路径相同** 时 **KV 物理块可复用**，减少 **重复 prefill** 与 **显存复制**。vLLM 的 **PagedAttention** 解决的是 **分页式 KV 布局 + continuous batching**，前缀复用也有演进，但叙事上 SGLang 更强调 **radix 树调度**。

**动态 batch**：多请求 **decode 步** 若可对齐到同一 wave，**合并 GEMM** 提高吞吐；**分支/结构化生成** 多时调度器复杂度高。

### 面试怎么答

「我看 workload：长前缀多轮对话会多看 SGLang；纯吞吐服务也会用 vLLM。」

---

## 3. asyncio：什么时候真有用

**API 网关**等 **I/O 等待** 重的地方，`asyncio` 用 **单线程事件循环** 挂起大量在途 HTTP，**比一线程一请求省资源**。  
**本地单卡推理**：async 只解决 **排队**，**算力仍是一块 GPU 顺序算**；要高吞吐要 **多实例或多卡**。

### 面试怎么答

「asyncio 解决高并发 I/O；GPU 吞吐要靠 batch 引擎和多副本。」

---

## 4. 大规模 PDF：多进程还是多线程

**CPU 重的解析/OCR**：**多进程** 躲 **GIL**；注意内存与句柄。  
**I/O 重**：**多线程或 async**。  
混合：**async 收任务 + 进程池算 CPU 段**。

---

## 5. 强制 JSON：从硬到软四层

1. **约束解码 / grammar**：每步 logits **屏蔽非法 token**（`response_format`、`outlines` 等）。  
2. **JSON Schema 模式**：字段类型枚举约束。  
3. **Prompt**：只输出 JSON、不要 markdown 围栏。  
4. **后处理**：括号栈截取 + `json.loads` + **Pydantic 校验** + **把 schema 错误喂回模型修复一轮**。

### 面试怎么答

「能约束解码就不要只靠正则；正则留给老模型兜底。」

---

## 本篇小结

- 框架对比 = **看 workload**。  
- asyncio = **I/O 并发** ≠ **GPU 变快**。  
- JSON = **grammar > schema > prompt > extract**。
