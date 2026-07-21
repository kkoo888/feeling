# 进化系统操作手册 — Agent 替代 LLM 指南

## 概述

本手册说明如何用 AI Agent 替代 LLM API，驱动项目的进化系统。进化系统的每步会发起 3 次 LLM 请求，Agent 截取请求内容、分析后返回结果，进化系统继续运行。

---

## 架构

```
┌─────────────────────────────────────────────────┐
│              进化系统 (不修改)                     │
│                                                   │
│  EvolutionAgent                                   │
│    ├─ search_policy()  决策: draft/improve/debug/ │
│    ├─ _draft()         构建 prompt                │
│    ├─ _improve()       构建 prompt                │
│    ├─ _debug()         构建 prompt                │
│    ├─ _explore()       构建 prompt                │
│    ├─ plan_and_strategy_query()                   │
│    │   └─ query() ──→ [被替换] ──→ Agent          │
│    ├─ AgentRunner.run()                           │
│    │   └─ query() ──→ [被替换] ──→ Agent          │
│    └─ parse_exec_result()                         │
│        └─ query() ──→ [被替换] ──→ Agent          │
│                                                   │
│  EvolutionJournal  记录进化树                      │
│  check_convergence() 收敛检测                      │
└─────────────────────────────────────────────────┘
```

---

## 前置条件

```bash
cd /path/to/feeling
pip install dataclasses-json  # 如果没有
```

---

## 核心：替换 query 函数

进化系统所有 LLM 调用经过 `evolution/backend/query()`。Agent 只需替换这一个函数。

### 替换方式

```python
import evolution.backend as backend
import evolution.agent as agent_mod

# 定义替代函数
def my_query(system_message, user_message, model="", temperature=None,
             max_tokens=None, func_spec=None, **kwargs):
    # 处理请求，返回结果
    ...

# 替换 (必须替换所有导入了 query 的模块)
backend.query = my_query
agent_mod.query = my_query
```

---

## 每步 3 次 LLM 请求

进化系统每步调用 3 次 `query()`，Agent 需要识别请求类型并返回对应格式。

### 请求 1: 策略生成

**触发时机**: `agent.step()` → `_draft()` / `_improve()` / `_debug()` / `_explore()`

**请求特征**:
- system_message 是 dict，包含 `简介`、`任务描述`、`历史记录`、`指令`
- 包含关键词: `策略设计师`、`策略代码块`、`改进概述`、`修复方案概述`、`探索指南`
- user_message 为 None

**识别方式**:
```python
if isinstance(system_message, dict):
    full = json.dumps(system_message, ensure_ascii=False)
else:
    full = str(system_message)

is_strategy = any(k in full for k in [
    "策略设计师", "策略代码块", "改进概述", "修复方案概述", "探索指南"
])
```

**返回格式**: 字符串，包含自然语言概述 + 代码块
```
方案概述（3-5句话）

```python
策略代码
```
```

**注意**: `plan_and_strategy_query()` 会用 `extract_code()` 提取代码块，用 `extract_text_up_to_code()` 提取概述。两者都必须存在。

### 请求 2: 策略执行

**触发时机**: `AgentRunner.run()` → `query()`

**请求特征**:
- system_message 是 dict，包含 `角色`、`任务描述`、`执行策略`、`执行要求`
- 包含关键词: `执行器`、`Agent 执行器`
- user_message 为 None

**识别方式**:
```python
is_execute = any(k in full for k in ["执行器", "Agent 执行器"])
```

**返回格式**: 字符串，执行结果描述
```
=== 策略执行结果 ===
执行摘要: ...
关键步骤: ...
执行结果: ...
执行时间: X.XX 秒
```

### 请求 3: 结果评估

**触发时机**: `agent.parse_exec_result()` → `query(func_spec=review_func_spec)`

**请求特征**:
- system_message 是 dict，包含 `简介`、`任务描述`、`策略内容`、`执行输出`
- 包含关键词: `评估专家`、`判断是否有 bug`
- **func_spec.name == "submit_review"**
- user_message 为 None

**识别方式**:
```python
is_evaluate = (func_spec and func_spec.name == "submit_review") or "评估专家" in full
```

**返回格式**: **dict**（不是字符串！）
```python
{
    "is_bug": False,           # bool: 是否有 bug
    "summary": "执行成功...",   # str: 总结
    "metric": 0.65,            # float: 指标值
    "lower_is_better": False   # bool: 指标是否越小越好
}
```

**注意**: 这是唯一返回 dict 的请求。返回字符串会导致 `string indices must be integers` 错误。

---

## 完整模板

```python
"""
进化系统 Agent 替代 LLM 模板
"""
import json, logging, sys, time, threading, queue
from pathlib import Path

ROOT = Path("/path/to/feeling")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evolution"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s | %(message)s')
logger = logging.getLogger("evo")

# ═══ LLM Mock (队列) ═══
_req_q = queue.Queue()
_res_q = queue.Queue()

def my_query(system_message, user_message, model="", temperature=None,
             max_tokens=None, func_spec=None, **kwargs):
    """替代 LLM — 请求进队列，等响应。"""
    _req_q.put({
        "system": str(system_message) if system_message else "",
        "user": str(user_message) if user_message else "",
        "func": func_spec.name if func_spec else None,
    })
    return _res_q.get()

# ═══ 请求处理 ═══

def handle_request(req):
    """根据请求类型返回结果。"""
    s, u, f = req["system"], req["user"], req.get("func")
    if isinstance(s, dict): s = json.dumps(s, ensure_ascii=False)
    full = f"{s}\n{u}"

    # 1. 策略生成
    if any(k in full for k in ["策略设计师", "策略代码块", "改进概述", "修复方案", "探索指南"]):
        return generate_strategy(full)

    # 2. 结果评估 (返回 dict!)
    if f == "submit_review" or "评估专家" in full:
        return {"is_bug": False, "summary": "策略执行成功", "metric": 0.65, "lower_is_better": False}

    # 3. 策略执行
    if "执行器" in full:
        return "=== 执行结果 ===\n策略成功执行。\n执行时间: 0.02秒"

    # 4. 应用代码到目标文件
    if "将以下策略应用" in full or "应用以下策略" in full:
        return apply_code(full)

    return generate_strategy(full)

def generate_strategy(prompt):
    """生成策略 — 根据请求内容定制。"""
    # 分析需求
    needs = []
    for kw in ["意图", "实体", "情感", "融合", "信息密度", "推理链"]:
        if kw in prompt:
            needs.append(kw)

    parts = []
    if "意图" in needs:
        parts.append("扩展意图关键词:\nINTENT_KEYWORDS.update({...})")
    if "实体" in needs:
        parts.append("实体提取:\nclass EntityExtractor:\n    def extract(self, text): ...")
    if "情感" in needs:
        parts.append("情感分析:\nclass SentimentAnalyzer:\n    def analyze(self, text): ...")

    plan = f"策略: {', '.join(needs[:3]) if needs else '综合优化'}"
    code = "\n\n".join(parts) if parts else "# 默认策略"
    return f"{plan}\n\n```python\n{code}\n```"

def apply_code(prompt):
    """把策略应用到目标代码。"""
    # 读取原始代码
    v2_path = ROOT / "目标文件路径"
    code = v2_path.read_text()

    # 根据策略修改代码
    # ... 具体修改逻辑 ...

    return code

# ═══ 请求处理线程 ═══
_handler_running = True

def request_handler():
    """持续处理 LLM 请求。"""
    while _handler_running:
        try:
            req = _req_q.get(timeout=1)
        except queue.Empty:
            continue
        resp = handle_request(req)
        _res_q.put(resp)

# ═══ 主流程 ═══

def main():
    # 替换 query
    import evolution.backend as backend
    import evolution.agent as agent_mod
    backend.query = my_query
    agent_mod.query = my_query

    # 启动请求处理线程
    handler = threading.Thread(target=request_handler, daemon=True)
    handler.start()

    # 初始化进化系统
    from evolution.agent import EvolutionAgent
    from evolution.runner import AgentRunner
    from evolution.journal import EvolutionJournal
    from evolution.evaluator import evaluate_fusion_engine

    journal = EvolutionJournal()
    agent = EvolutionAgent(
        task_desc="你的任务描述",
        journal=journal,
        model="me",
        temperature=0.7,
        num_drafts=3,
        debug_prob=0.2,
        explore_prob=0.15,
    )
    runner = AgentRunner(task_desc="你的任务描述", model="me")
    agent._last_runner = runner
    exec_cb = lambda s, r=True: runner.run(s, r)

    # 运行进化
    for step in range(1, 21):  # 20 步
        logger.info(f"进化步骤 {step}")
        try:
            agent.step(exec_callback=exec_cb)
        except Exception as e:
            logger.error(f"步骤失败: {e}")

        best = journal.get_best_node()
        if best:
            logger.info(f"最佳: {best.metric.value:.4f}")

    # 停止
    global _handler_running
    _handler_running = False

    # 报告
    logger.info(f"完成: {len(journal)} 节点, 最佳: {journal.get_best_node().metric.value:.4f}")

if __name__ == "__main__":
    main()
```

---

## 运行

```bash
cd /path/to/feeling
.venv/bin/python3 evolution/your_script.py
```

---

## 关键注意事项

### 1. query 替换必须覆盖所有模块
```python
backend.query = my_query      # 替换后端
agent_mod.query = my_query    # 替换 agent 里导入的引用
```
如果只替换 `backend.query`，`agent.py` 里 `from .backend import query` 导入的引用不会被更新。

### 2. 评估请求必须返回 dict
```python
# 正确
return {"is_bug": False, "summary": "...", "metric": 0.65, "lower_is_better": False}

# 错误 (会导致 string indices must be integers)
return '{"is_bug": false, ...}'
```

### 3. system_message 可能是 dict
进化系统的 prompt 是 dict 格式，需要 `json.dumps()` 转字符串后再判断。

### 4. 策略响应必须包含代码块
`plan_and_strategy_query()` 用 `extract_code()` 提取 ```python ... ``` 中的代码。如果没有代码块，会重试 3 次后失败。

### 5. 每轮独立
如果要从同一个基线出发生成多个版本，每轮都要重新初始化 `EvolutionAgent` 和 `EvolutionJournal`，避免上一轮的节点影响下一轮。

### 6. 评估器接口
```python
from evolution.evaluator import evaluate_fusion_engine

# 需要一个有 process(text) 方法的对象
class Wrapper:
    def __init__(self, engine): self._engine = engine
    def process(self, text): return self._engine.process(text)

result = evaluate_fusion_engine(Wrapper(engine), version="v1")
save_eval(result)
```

---

## 文件结构

```
feeling/
├── evolution/
│   ├── agent.py          # 进化 Agent (不修改)
│   ├── runner.py         # 策略执行器 (不修改)
│   ├── journal.py        # 进化日志 (不修改)
│   ├── evaluator.py      # 评估器 (可选修改: 增加评估维度)
│   ├── backend/
│   │   ├── __init__.py   # query() 定义 (被替换)
│   │   └── backend_openai.py  # LLM 后端 (不修改)
│   └── run.py            # 主循环 (不修改)
├── aris_brain/
│   ├── aris_fusion_engine_v2.py  # 目标代码
│   └── evolution_versions/       # 进化版本 (自动生成)
└── .openclaw/tmp/
    ├── eval_results/     # 评估结果 (自动生成)
    └── evolution_journal.json  # 进化日志 (自动生成)
```

---

## 进化流程图

```
初始化: journal + agent + runner
         │
    ┌────▼────┐
    │ agent.  │
    │ step()  │
    └────┬────┘
         │
    ┌────▼────────────────────┐
    │ search_policy()         │
    │ ├─ draft不足 → _draft() │
    │ ├─ 20% → _debug()      │
    │ ├─ 15% → _explore()    │
    │ └─ 贪婪 → _improve()   │
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │ plan_and_strategy_query │
    │ └─ query() → Agent LLM │ ← 请求1: 返回 plan+strategy
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │ AgentRunner.run()       │
    │ └─ query() → Agent LLM │ ← 请求2: 返回执行结果
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │ parse_exec_result()     │
    │ └─ query() → Agent LLM │ ← 请求3: 返回 {is_bug, metric, ...}
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │ journal.append(node)    │
    │ check_convergence()     │
    └────┬────────────────────┘
         │
    └────循环─────────────────┘
```

---

## 版本: v1.0
## 日期: 2026-07-21
## 作者: 小茜
