# 09｜Agent 工程、Harness 与 Prompt / 幻觉

本篇整合两条线：**LangChain → LangGraph → Harness Engineering**（Agent 框架与生产运行时），以及 **Prompt / 数据 / 幻觉**（产品向接口与系统缓解）。与 **`13` RAG**、**`15` Agent 架构**、**`05` 对齐** 互补阅读。

**建议阅读顺序**：第一部分（框架演化）→ 第二部分（Harness 与生产问题）→ 第三部分（为什么是 Graph）→ 第四部分（Prompt 与幻觉）。

**演化路线**：

```text
LangChain（LLM 应用组件库 / Chain 流水线）
  → LangGraph（有状态图 / Agent Runtime）
  → Harness Engineering（长期稳定：State、调度、校验、预算、恢复）
```

| 本篇侧重 | 可对照 |
|----------|--------|
| 框架、Runtime、Harness | `15` Agent、Tool、上下文 |
| Prompt、数据、幻觉 | `13` RAG、`05` DPO/RLAIF |
| 约束解码 | `14`（若有） |

---

## 第一部分：LangChain 与 LangGraph

### 1.1 先分清「框架」各自解决什么层级

很多人第一次接触 Agent 相关框架时，会把 LangChain、LangGraph、AutoGen、CrewAI 都笼统叫「AI 框架」。实际上它们处在不同抽象层：

- **LangChain**：LLM **应用开发工具库**——把 Prompt、Retriever、Memory、Tool、OutputParser 等拆成模块，用 **Chain** 串成流水线。
- **LangGraph**：**Agent Runtime / 状态机框架**——把运行过程建模成 **有状态的图**（State → Node → 路由 → 下一 Node），支持 checkpoint、人工介入、多 Agent 编排。
- **AutoGen / CrewAI**（对照）：更偏 **多 Agent 对话编排** 与 **角色分工**；和 LangGraph 的 **显式 State 图 + 可恢复执行** 不是同一层，实践中常 **组合**（例如 LangGraph 做总调度，子图里跑 AutoGen 小组）。

LangGraph 可以理解为：LangChain 团队发现「纯 Chain 抽象盖不住复杂 Agent」之后，新做的一层 **运行时系统**。整条线反映的是 **Agent Engineering** 从「拼 Prompt」走向「做 Runtime」。

---

### 1.2 LangChain 是什么、为什么会火

早期（约 2022–2023）LLM 应用多是 `prompt → llm → output`。很快大家需要 **memory、retrieval、tools、agents、output parser、chains**，LangChain 于是把 LLM 应用拆成可复用模块，例如 `PromptTemplate / LLM / Retriever / Memory / Tool / Agent / OutputParser`，再用 Chain 连接——故名 LangChain。本质是 **「LLM 应用组件库」**。

它火起来是因为当时直接调 API 很乱，例如裸调：

```python
response = openai.chat.completions.create(...)
```

你要自己处理 prompt、memory、retrieval、parsing、retry、tool calling。LangChain 提供统一抽象，有点像 AI 时代的 PyTorch Lightning、requests 或 sklearn pipeline。模块拆分示意：

```python
PromptTemplate, LLM, Retriever, Memory, Tool, Agent, OutputParser
# 再通过 Chain 连接
```

**核心抽象是 Chain**，典型 RAG 流水线：

```text
User Question → Retriever → Prompt → LLM → Parser
```

这叫 **pipeline abstraction**。问题在于：**真实 Agent 往往不是线性的**——会 loop、retry、branch、checkpoint、interrupt、human-in-the-loop、rollback、并行。LangChain 早期抽象仍偏 **线性流水线**，复杂 Agent 一写就容易在业务代码里堆满：

```python
if ...
while ...
retry ...
agent_executor ...
```

最后变成 spaghetti。大家后来吐槽 **「LangChain abstraction leak」**：框架抽象 **盖不住** 真实控制流复杂度——该分支、该 checkpoint、该 HITL 的逻辑仍要你自己写，只是套了一层 Chain 的皮。

---

### 1.3 LangGraph 为什么出现、核心思想

LangGraph 代表 LangChain 侧的一个认知升级：**Agent 本质不是 chain，而是 stateful graph**。

核心公式是：

```text
State → Node 执行 → State 更新 → 路由 → 下一 Node
```

因此 LangGraph 更像 **workflow engine / 分布式编排 / 有限状态机运行时**，而不是「又一个 prompt 库」。它主要解决 **长生命周期 Agent 的控制问题**。例如 coding agent：

```text
User request → Planner → Tool call → Code generation → Run tests
  → Failure → Reflection → Retry → Human approval → Deploy
```

这里有 **循环、条件跳转、错误恢复、状态持久化、checkpoint、并发**——这是 **graph**，不是固定 A→B→C 的 chain。

---

### 1.4 State：LangGraph 里最重要的一块

以前「上下文」往往等于 **一坨 prompt 文本**；LangGraph 把 **state 结构化**，例如：

```python
class AgentState:
    messages
    current_goal
    completed_steps
    retrieved_docs
    tool_results
    failures
```

每个 node 形如 `def planner(state): ... return updated_state`，语义上接近 Redux、Workflow Runtime 或 Actor 里的 **显式状态**，而不是把所有历史塞进自然语言让模型「自己记」。

**要点**：这里的 State 是 **runtime-level state**，不是「又多了一段 token」。LLM 在 node 里做 **局部推理**；**记忆与进度** 由 runtime 维护。例如 planner 只读 `goal` 与 `completed_steps`，不必吞掉全部 tool 原始输出——这就是 **harness** 思想的起点。

更细的 AgentState 字段示例（coding agent）：

```python
class AgentState:
    user_goal: str
    current_plan: list
    completed_steps: list
    retrieved_files: list
    edited_files: list
    test_results: dict
    failure_count: int
    current_subtask: str
    memory_summary: str
```

---

### 1.5 LangGraph 与 ReAct 的关系（别混层级）

**ReAct** 是 **推理策略**（thought → action → observation 循环）。**LangGraph** 是 **运行时编排框架**。完全可以在 LangGraph 的某个 node 里跑 ReAct：

```text
Graph Runtime → Planner Node → ReAct Executor Node → Verifier Node
```

LangGraph 比 ReAct **高一个层级**：ReAct 管「模型怎么想一步」；LangGraph 管「系统怎么跑很多步、怎么分叉、怎么恢复」。

---

### 1.6 为什么严肃 Agent 都在 graph 化

复杂任务天然是 graph。Browser agent 举例：

```text
Open page
    ↓
Need login?
   /   \
 yes    no
  ↓      ↓
login  continue
```

打开页面后 **要不要登录** 就分叉；登录流与继续浏览是不同路径——不是 linear chain。

LangGraph 的真正价值不是「更复杂的 LangChain」，而是把 Agent Engineering 从 **prompt orchestration** 推进到 **runtime systems engineering**。工业界 Agent 也越来越像 Airflow、Temporal、Ray、Prefect、Dagster 这类 **工作流系统**，只是 node 里换成 LLM / tool / human / verifier。

---

### 1.7 LangGraph 的工业级能力（对照记忆）

1. **Durable execution**：中断后可从 checkpoint 继续，长任务不必从头跑。  
2. **Checkpointing**：可 rollback；long-horizon agent 中途失败否则很难救。  
3. **Human-in-the-loop**：例如 Agent 说「要删库」→ 等人批准 → 再执行 destructive tool。  
4. **Multi-agent orchestration**：多 Agent 分工、并行子图、汇总 verifier。  
5. **Streaming state updates**：状态增量推给 UI/日志，便于观测。

| 类比 | LangChain | LangGraph |
|------|-----------|-----------|
| Web | React components | Node.js runtime |
| OS | 函数库 | 操作系统（调度+状态） |
| ML | sklearn pipeline | distributed runtime |
| Backend | utility library | workflow orchestrator |

---

### 1.8 为什么很多人学 LangChain 学歪了

很多教程只教 `prompt | llm`，但 production Agent 的难点根本不是 prompt chaining，而是：**状态控制、failure recovery、tool 可靠性、memory、observability、orchestration、long-horizon execution**——这些正是 LangGraph / Harness 更关心的。

---

## 第二部分：Harness Engineering 与 LangGraph 落地

### 2.1 三层分工：ReAct / LangGraph / Harness

- **ReAct**：模型 **怎么思考**（一步 thought-action-observation）。  
- **LangGraph**：系统 **怎么运行**（图、状态、路由、checkpoint）。  
- **Harness engineering**：怎么让整套系统 **长期稳定、不失控**（上下文构造、校验、预算、恢复）。

---

### 2.2 最原始的 Agent 写法（ReAct loop）

做一个 coding agent，用户说「帮我修 repo 的 bug」，最原始实现往往是：

```python
while True:
    thought = llm(...)
    if thought.tool == "search":
        result = search(...)
    elif thought.tool == "edit":
        edit(...)
    elif thought.tool == "finish":
        break
```

一开始觉得「能跑」；上生产后很快暴露问题。下面六类是 **真实 production** 里最常见、也最值得在面试/设计里讲清楚的。

---

### 2.3 生产环境六大典型问题

**（1）上下文爆炸**  
跑 30 步后 prompt 里堆满 `step1…step30…`，带来 token 爆炸、注意力退化、模型忘重点、幻觉上升——往往是第一大坑。

**（2）状态混乱**  
Agent 已经找到 bug 文件、改过代码、跑过测试，但信息全散落在自然语言里（`I have edited app.py…`）。下一轮模型可能「忘」了，因为 **LLM 没有真正的 state，只有 token continuation**。

**（3）失败无法恢复**  
第 27 步 tool timeout，整个 run 直接崩——没有 checkpoint，无法从第 12 步状态恢复。

**（4）无限循环**  
反复 `search search search…`，ReAct 经典死循环，没有图层面的终止条件与预算。

**（5）tool observation 污染 context**  
例如网页 tool 返回：

```html
<div>5000行垃圾HTML</div>
```

若原样塞进 context，模型立刻被噪声淹没；Harness 里应由 **Context Constructor** 做裁剪、结构化或摘要后再进窗。

**（6）planner 与 executor 相互污染**  
Planner 写下「1 read 2 fix 3 test」，executor 跑完后 context 全是 execution noise，下次 replan 看不清全局。

---

### 2.4 LangGraph 怎么治：显式 State 与状态隔离

**本质**：把 **隐式藏在 prompt 里的 token 状态**，变成 **显式结构化 state**。

以前状态藏在：`We already searched app.py…`  
现在单独存字段，runtime 负责读写；LLM **不负责长期记忆**，只负责 **当前 node 的局部推理**。例如：

```python
def planner(state):
    plan = llm(goal=state.user_goal, completed=state.completed_steps)
    return {"current_plan": plan}
```

**状态隔离（Context Specialization）** 是 production 里极关键的一招：不同 node 只看 **最小必要上下文**。

- **Planner Node**：`goal, completed_steps, current_progress`  
- **Code Editor Node**：`target_file, bug_description, relevant_code`  
- **Verifier Node**：`generated_patch, unit_test_results`  

这样能显著降低 attention noise、幻觉与 drift。

**显式 execution graph**（对比 `while True` 的 ReAct）：

```text
START → Planner → Executor → Verifier
                    ├ pass → END
                    └ fail → Reflection → Replanner → Executor
```

---

### 2.5 Failure Recovery

以前一步错、全盘崩。Graph runtime 可 **retry node、rollback state、branch、checkpoint restore**。例如 `checkpoint(step=12)`，第 17 步挂了恢复到 12，不必从 0 重跑。

---

### 2.6 Harness 是什么：控制 graph runtime 的七大块

Harness = **控制整个 graph runtime 的系统**，常见包括：

1. **State Manager**：working memory、摘要、tool 输出、plan、metadata。  
2. **Scheduler**：决定下一步跑哪个 node，例如 `if verification_failed: goto("reflection")`。  
3. **Context Constructor**：**给不同 node 构造最小必要上下文**——很多 production Agent 的核心竞争力在这里。  
4. **Verifier System**：`run_unit_test()`、`check_schema()`、`check_grounding()`——**不信 LLM 自说自话**，runtime 验证。  
5. **Memory Compression**：`summary = summarize(old_trajectory)`，否则长任务必 context 爆炸。  
6. **Retry Policy**：`if tool_timeout: retry(3)`。  
7. **Budget Controller**：token、latency、API cost，否则线上会烧钱。

---

### 2.7 案例：Devin 类任务为什么是 graph

任务：修复 CI failed tests。真实路径往往是：

```text
read github issue → inspect repo → find failing test → run locally
→ identify dependency issue → edit code → rerun tests → still failing
→ rollback → search docs → retry patch
```

这不是 linear chain，而是 **graph search**。

---

### 2.8 「State 比 Prompt 重要」与 frontier 结构

Agent engineering 的一大转向：以前容易以为 **Prompt = intelligence**；现在更清楚 **State management = reliability**。LLM 已经够强，production 瓶颈常在：**状态污染、memory drift、trajectory 爆炸、recovery failure**——不是「prompt 不够 fancy」。

很多 frontier 系统结构类似：

```text
User Goal → Task Decomposer → Execution Graph Runtime
  ├── Tool / Browser / Coding / Retrieval Agent
  └── Verifier Agent
        ↓
  State Store → Memory Compression → Recovery System
```

越来越像 **分布式系统 / OS / 工作流引擎**，而不是单轮 chatbot。

---

### 2.9 自己做 Agent 时最容易犯的错

初学者常死磕 prompt wording、few-shot、CoT trick；production 真正折磨人的往往是：

```text
第 42 步为什么突然忘记任务？
为什么无限循环？
为什么 planner 开始胡说？
为什么 tool 输出污染 context？
为什么 retry 后越来越偏？
```

这些都是 **runtime / state / harness** 问题，不是换几句 prompt 能根治的。

---

## 第三部分：为什么 Agent 任务天然是 Graph

### 3.1 常见误解

看到「graph」不要以为只是「流程图画复杂一点」。**核心**是：执行过程中 **未来路径不是预先固定的**，而是 **根据当前状态与 observation 动态分叉**——这才叫 graph。

---

### 3.2 Linear chain 长什么样

路径固定：`A → B → C → D`，没有分支、回退、条件跳转。典型 RAG：`用户输入 → RAG → LLM → 输出`，执行路径事先定死。

---

### 3.3 真实 coding agent：一步 error 就分叉

任务「修复 failing test」，纸面上像 `read test → fix code → run tests → done`。真实可能是：读完 test 发现 `ModuleNotFoundError`，然后：

```text
                    ┌─ dependency 没装 → install → rerun
error ──────────────┼─ import path 错 → 改 PYTHONPATH → rerun
                    └─ 测试本身坏了 → inspect test → rewrite test
```

这不是 `A→B→C`，而是 **graph**。

**动态决策**与搜索同构：当前状态 → 有哪些动作 → 执行 → 新状态 → 继续。已接近 RL / 树搜索。

---

### 3.4 Browser agent：订票

「帮我订机票」→ 打开网站 → **要登录？** 是则走 login flow，否则 search flights → 若无直飞再分叉：换日期 / 换机场 / 换航司。每一步都是 **动态 branching**，即 graph traversal。

---

### 3.5 本质：State Space

任意时刻 Agent 有一个 state，例如：

```python
State = {
  current_page, login_status, selected_flight,
  known_errors, completed_actions
}
```

不同 action 把系统带到不同 state：`state_1 --action_a--> state_2`，`state_1 --action_b--> state_3`。  
**节点 = state，边 = action**——这就是图。

Frontier agent 越来越像 **search system**：在巨大状态空间里找成功路径，与 game AI、AlphaGo、robotics planning、path planning 同族。

---

### 3.6 为什么 ReAct 相对弱、强系统开始 tree search

ReAct 只有 **一条 trajectory**（thought → action → observation → …），难以 **branch、比较备选、rollback、并行搜多条路径**；早期决策错了后面全歪。

强系统开始：**多个 candidate action → 并行执行 → verifier 打分 → 保留最优 trajectory**，很像 Monte Carlo Tree Search。LangGraph 适合表达：

```text
planner → executor → verifier
  ├ success → finish
  ├ retry → executor
  └ replan → planner
```

重点：**不是「画张图」**，而是 **runtime 真的在走不同路径**。

**再强调**：Graph 不是因为结构画得复杂，而是因为 **未来路径依赖运行时状态**。路径若提前固定，再复杂也是 chain；若运行时决定，就是 graph。现实任务有 uncertainty、partial observability、动态环境、failure recovery、replanning——**不存在固定 execution path**，Agent 必须边观察、边搜索、边修正。高级 Agent 本质是 **在状态空间中的受控搜索系统**。

---

## 第四部分：Prompt、数据、幻觉（产品向）

本部分偏 **不改权重** 条件下的接口设计与数据治理；与第一～三部分的 **Runtime / RAG 系统** 互补：**幻觉最终要靠系统治，不能只靠 prompt 措辞**。

---

### 4.1 Prompt 不是咒语，是「接口设计」

在不改模型权重的前提下，你用 **格式、顺序、示例、禁止项** 去 **缩小模型合法输出空间**；它与 **温度、top-p、后处理** 是一套工程。

**怎么算达标**：任务成功率、格式约束通过率、换说法仍稳（鲁棒）。**怎么评估**：小离线集 + 规则校验；线上影子流量；**prompt 版本与模型版本一起发版**。

**面试口述**：「Prompt 是条件分布的工程设计；要版本化、要离线+线上双评。」

---

### 4.2 Few-shot、CoT、角色提示

- **Few-shot**：用例子 **锚定输出形态**；坏例子会被学走。  
- **CoT**：要求 **中间步骤**，相当于换更深的 effective 计算；仍可能 **过程胡编**。  
- **角色**：语气与立场软约束；与安全策略冲突时要 **系统层兜底**。

**面试口述**：「few-shot 定格式；CoT 买深度；角色定风格；事实错要靠 RAG/工具。」

---

### 4.3 数据清洗与 instruction 构造

**清洗**：去重、去隐私、语言检测、低质过滤、保留文档结构（标题层级）。**脏数据**：规则优先 + 小模型质检 + 人审难例；**记录版本号** 可复现。

**instruction 数据**：覆盖 **任务类型 × 难度 × 语言**；与线上一致的 **system / tools schema**；合成数据要 **人审或交叉审**。

---

### 4.4 幻觉：定义、发现、缓解

**定义**：读起来像那么回事，但 **无依据或与事实不符**。

**发现**：检索对照、可执行检查、多采样一致性、线上点踩。

**缓解（系统向，不单靠 prompt）**：

- **RAG + 引用链**：生成前注入 chunk；生成后 **span–chunk 重叠**、**引用 id 是否存在**；失败则拒答或重检索（见 `13`）。  
- **约束解码**：必须 JSON / 必须 citation 字段时用 **grammar**（见 `14`）。  
- **偏好对齐**：事实性/有据性维度做人标或 RLAIF，走 DPO/RLHF（见 `05`）——标注维度是「是否胡编」，loss 仍是成对或 RM+PPO。  
- **工具闭环**：算术、查库、实时状态 **必须走执行器**；reward 可对执行结果二值化，与 GRPO 友好。

**面试口述**：「幻觉要系统：证据链 + 后验校验 + 约束解码；对齐里把有据当偏好维度训。」

---

### 4.5 「实际项目」口述模板

- **清洗**：按来源说规则 + 抽样表 + 版本脚本。  
- **脏数据**：丢弃阈值 vs 修正策略，避免分布塌缩。  
- **instruction**：意图矩阵 + 拒答/澄清负例 + schema 对齐。  
- **幻觉**：离线引用检查 + 线上触发检索/拒答。

---

## 全篇小结

- **LangChain**：组件 + Chain 流水线；复杂 Agent 易 abstraction leak。  
- **LangGraph**：State + Node + 路由；checkpoint、HITL、多 Agent；比 ReAct 高一层。  
- **Harness**：按 node 构造上下文、Verifier、压缩、重试、预算、恢复——production 难在 runtime。  
- **任务天然是 Graph**：路径依赖运行时 observation 动态分叉。  
- **Prompt / 数据 / 幻觉**：Prompt 是接口设计；幻觉要 RAG + 校验 + 工具 + 对齐。
