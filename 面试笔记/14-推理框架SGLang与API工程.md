# 14｜推理框架（vLLM / SGLang 等）、asyncio、多进程 vs 多线程、JSON 可靠输出

对应：**`面试答题.md` 新增**（第 14～17 行）；与 **`08`**（vLLM、PagedAttention）互补，本篇补 **SGLang、异步、并发模型、JSON**。

---

## 1. 你了解哪些大模型推理框架？

**常见名字**（面试枚举即可）：

- **vLLM**：PagedAttention、continuous batching，**吞吐**与 **多租户服务** 强。  
- **SGLang**：**结构化生成、RadixAttention、前端语言（SGL）** 描述 **多分支/多轮共享前缀** 的图，强调 **延迟与调度**。  
- **TensorRT-LLM / FasterTransformer**：NVIDIA 生态 **极致 kernel**；部署成本高。  
- **TGI（Text Generation Inference）**：HuggingFace 栈常用。  
- **llama.cpp / ollama**：CPU/端侧与 **本地** 场景。

选型：**延迟敏感 + 复杂控制流** 多看 SGLang；**纯服务吞吐 + 生态** 多看 vLLM（非绝对）。

---

## 2. SGLang 相对 vLLM 的 PagedAttention：推理延迟上可能有哪些优势？

**说明**：两者都在快速迭代，以下答 **设计动机级**，避免「绝对谁快」。

**SGLang 常见卖点（与延迟相关）**：

1. **RadixAttention（前缀树缓存 KV）**：多请求 **共享相同 system prompt / 多轮历史前缀** 时，**KV 复用** 更系统，减少 **重复 prefill**。  
2. **结构化生成与批调度**：把 **多个候选分支、工具调用、约束解码** 表达成图，**scheduler** 可在 **token 粒度** 做更细 batching，减少 **空等**。  
3. **与 PagedAttention 关系**：SGLang 也可 **内部用分页式 KV**；对比对象常是「**仅 vLLM 默认路径**」在 **重前缀 / 多轮对话** 下的 **prefill 重复**。

**vLLM 侧**：PagedAttention 解决 **显存碎片与 batching**；**前缀缓存** 后续版本也在加强。

**面试金句**：**延迟** 不仅看 kernel，还看 **前缀复用率** 与 **调度**；**多轮 RAG / Agent** 场景 **前缀长**，Radix/前缀缓存收益大。

---

## 3. 调用大模型 API 为什么要用 asyncio？

**原因**：API 调用是 **I/O 等待型**；同步代码 **阻塞线程**，并发高时 **线程数爆炸** 或吞吐差。

**asyncio 优势**：

- **单线程事件循环** + **非阻塞 HTTP**，可同时挂起 **成百上千** 在途请求；  
- **限流（semaphore）+ 重试 + 熔断** 易组合；  
- 与 **FastAPI / aiohttp** 栈一致。

**注意**：**GPU 本地推理** 若单进程单卡，async 只解决 **请求排队**，真正算力仍 **串行**；要 **高吞吐** 需 **多 worker / 多实例 + 推理引擎 batch**。

---

## 4. 大规模 PDF 解析：多线程还是多进程？

**经验法则**：

- **CPU 密集**（解析、OCR、布局模型在 CPU 跑）：**多进程** 绕过 **GIL**，用 **进程池**；注意 **内存复制** 与 **文件句柄** 上限。  
- **I/O 密集**（读 S3、写盘）：**多线程** 或 **async I/O** 即可。  
- **混合**：**async 收任务 + 进程池执行 CPU 段** 是常见架构。

**Python 特供**：纯 Python 解析且 **重 CPU** → 优先 **多进程**。

---

## 5. 如何确保 Agent 返回 **标准 JSON**？有多余说明文字怎么提取？

**分层策略**（从内到外）：

1. **约束解码**：**JSON mode / grammar-guided decoding**（outlines、guidance、厂商 API 的 `response_format`），从根上 **消灭非法 JSON**。  
2. **Prompt**：「**仅输出 JSON，无 markdown 围栏**」+ **示例 schema**。  
3. **后处理提取**：  
   - 先找 **第一个 `{` 与最后一个匹配的 `}`**（栈计数）；  
   - 或正则提取 **```json ... ```** 块；  
   - 再用 **`json.loads`**；失败则 **重试**（带错误信息反馈给模型 **repair**）。  
4. **校验**：**Pydantic / JSON Schema** 校验字段类型与必填项；失败走 **修复轮** 或 **降级空对象**。

**工程建议**：**能约束解码就不要只靠正则**；正则用于 **兼容老模型**。

---

## 6. 「强制」大模型生成 JSON：实现上怎么做？

从 **强约束 → 弱约束** 排列，面试按层答即可。

### 6.1 约束解码（最硬）

- **OpenAI 兼容 API**：`response_format={"type":"json_schema", ...}` 等（以厂商文档为准），服务端用 **grammar mask** 在 **每步解码** 禁止非法 token。  
- **本地推理**：`outlines`、`lm-format-enforcer`、`guidance`、部分引擎内置 **JSON mode / EBNF**。  
- **原理**：在 logits 上 **置 -inf** 给非法续写，保证 **可 parse**；代价是 **实现依赖引擎**、有时 **吞吐略降**。

### 6.2 结构化输出模式（中硬）

- 要求输出 **单行 JSON**、禁止 markdown；配合 **stop sequences** 截断尾部废话。

### 6.3 仅 Prompt + 后处理（最软）

- 栈扫描截取 `{...}` + `json.loads` + **Pydantic 校验** + **repair 重试**；适合 **无约束解码 API**。

### 6.4 校验与降级

- **JSON Schema / Pydantic**：字段类型、枚举、必填；失败则 **自动重试**（把 schema 错误贴回模型）或 **返回业务错误码**。

---

## 小结

- **推理框架**：按 **吞吐 vs 延迟 vs 前缀复用** 选型。  
- **asyncio**：解决 **高并发 I/O**；本地 GPU 仍要 **batch 与多实例**。  
- **PDF**：**CPU 多进程 + I/O 异步** 混合常见。  
- **JSON**：**grammar / json_schema 约束解码 > 提示词 > 提取+校验+修复**；第 6 节为「强制」实现的展开。
