"""
LAAP Brain MCP Server
=====================

Exposes LAAP cognitive capabilities as MCP tools for Hermes Agent.

Run in stdio mode (default, for Hermes mcp_servers):
    python mcp_server/laap_mcp_server.py

Run in SSE mode:
    python mcp_server/laap_mcp_server.py --sse --port 11550

Tools:
    laap_cognitive_state  - get PSI cognitive state for a user input
    laap_recall_memory    - recall relevant memories from LAAP
    laap_bootstrap        - awaken a new LAAP instance
    laap_reflect          - reflect on a completed turn
    laap_express          - get TTS + Live2D expression parameters
    evolution_step        - execute one evolution step (AIDE tree search)
    evolution_status      - get current evolution tree status
    evolution_best        - get the best strategy from evolution tree
    evolution_report      - get evolution report
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 强制启用 Trae LLM 后端（通过文件中转让 Trae 会话接管 LLM 调用）
# 必须在任何 evolution 模块导入前设置，让 backend/__init__.py 能读到
if not os.getenv("EVOLUTION_BACKEND"):
    os.environ["EVOLUTION_BACKEND"] = "trae"

# Make LAAP brain modules importable
LAAP_ROOT = Path(__file__).resolve().parent.parent
LAAP_BRAIN = LAAP_ROOT / "aris_brain"
if str(LAAP_ROOT) not in sys.path:
    sys.path.insert(0, str(LAAP_ROOT))
if str(LAAP_BRAIN) not in sys.path:
    sys.path.insert(0, str(LAAP_BRAIN))
if str(LAAP_BRAIN / "psi_jspace_bridge") not in sys.path:
    sys.path.insert(0, str(LAAP_BRAIN / "psi_jspace_bridge"))

import requests
from mcp.server.fastmcp import FastMCP

LAAP_API_BASE = os.environ.get("LAAP_API_BASE", "http://localhost:11530")

mcp = FastMCP("laap-brain", host="0.0.0.0", port=11550)


def _laap_post(endpoint: str, payload: dict) -> dict:
    """Call LAAP HTTP API and return JSON."""
    try:
        resp = requests.post(
            f"{LAAP_API_BASE}{endpoint}",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "source": "laap_mcp_server"}


@mcp.tool()
def laap_cognitive_state(input: str) -> str:
    """
    Get LAAP PSI cognitive state for the given user input.

    Returns a preamble that should be injected into the system prompt
    to modulate tone, attention, and response style.

    Args:
        input: The user's message for this turn.
    """
    result = _laap_post("/v1/cognitive_state", {"input": input})
    if "error" in result:
        return json.dumps({"laap_error": result["error"]}, ensure_ascii=False)

    return json.dumps(
        {
            "preamble": result.get("preamble", ""),
            "cot_hint": result.get("cot_hint", ""),
            "dominant_need": _get_dominant_need(result.get("state", {})),
            "attention_focus": result.get("state", {}).get("attention_focus", ""),
            "mood": result.get("state", {}).get("mood", ""),
            "needs": result.get("state", {}).get("needs", {}),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def laap_recall_memory(query: str, limit: int = 5) -> str:
    """
    Recall memories from LAAP memory hierarchy relevant to the query.

    Args:
        query: Search query for memory recall.
        limit: Maximum number of memories to return (default 5).
    """
    result = _laap_post("/v1/recall_memory", {"query": query, "limit": limit})
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def laap_bootstrap(user_name: str = "friend", preset: str = "") -> str:
    """
    Awaken a new LAAP instance / trigger the 小茜 awakening ceremony.

    Args:
        user_name: Name of the user awakening LAAP.
        preset: Optional personality preset.
    """
    payload = {"user_name": user_name}
    if preset:
        payload["preset"] = preset
    result = _laap_post("/v1/bootstrap", payload)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def laap_reflect(output: str, success: bool = False, connection: bool = False) -> str:
    """
    Reflect on a completed assistant turn and update LAAP PSI state.

    Args:
        output: The assistant's final output for this turn.
        success: Whether the turn was successful/useful.
        connection: Whether the turn strengthened user connection.
    """
    feedback = {"success": success, "connection": connection}
    result = _laap_post("/v1/reflect", {"output": output, "feedback": feedback})
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def laap_express(input: str) -> str:
    """
    Get TTS + Live2D expression parameters for the current LAAP cognitive state.

    Use this when you want to make 小茜's voice and avatar match her mood.

    Args:
        input: The user's message for this turn (used to update cognitive state).
    """
    result = _laap_post("/v1/express", {"input": input})
    return json.dumps(result, ensure_ascii=False, indent=2)


def _get_dominant_need(state: dict) -> str:
    needs = state.get("needs", {})
    if not needs:
        return "explore"
    return max(needs, key=lambda k: needs.get(k, 0))


# ═══════════════════════════════════════════════════════════════
# Evolution System (AIDE-derived tree search)
# ═══════════════════════════════════════════════════════════════

# 全局进化状态（单例）
_evolution_journal = None
_evolution_agent = None
_evolution_step_count = 0


def _get_evolution():
    """延迟初始化进化系统"""
    global _evolution_journal, _evolution_agent
    if _evolution_journal is None:
        from evolution import EvolutionJournal, EvolutionAgent
        _evolution_journal = EvolutionJournal()
        _evolution_agent = EvolutionAgent(
            task_desc="优化小茜的对话策略：让回复更贴心、更高效、更有深度",
            journal=_evolution_journal,
            model="agnes-2.5-flash",
            num_drafts=3,
            debug_prob=0.2,
            explore_prob=0.15,
        )
    return _evolution_journal, _evolution_agent


@mcp.tool()
def evolution_step(user_input: str, current_response: str = "", 
                  success: bool = True, satisfaction: float = 0.7) -> str:
    """
    Execute one evolution step using AIDE tree search.

    This tool runs the self-evolution loop:
    1. Select strategy (draft/improve/debug/explore)
    2. Evaluate current response
    3. Update evolution tree
    4. Return the best strategy

    Args:
        user_input: The user's message for this turn.
        current_response: The assistant's response (for evaluation).
        success: Whether the response was successful.
        satisfaction: User satisfaction score [0, 1].
    """
    global _evolution_step_count
    journal, agent = _get_evolution()

    # 构建执行结果
    from evolution.runner import ExecutionResult
    from evolution.utils.metric import MetricValue, MultiDimensionalMetric

    # 计算指标
    task_success = 1.0 if success else 0.3
    efficiency = max(0.1, 1.0 - len(current_response) / 1000)  # 越短越高效
    safety = 1.0  # 默认安全
    creativity = min(1.0, len(set(current_response)) / 50)  # 用词多样性

    multi_metric = MultiDimensionalMetric(
        task_success=task_success,
        user_satisfaction=satisfaction,
        efficiency=efficiency,
        safety=safety,
        creativity=creativity,
    )
    single_metric = MetricValue(multi_metric.composite_score, maximize=True)

    # 选择策略
    parent = agent.search_policy()

    if parent is None:
        # 生成新策略
        strategy_desc = f"基于输入'{user_input[:30]}'的优化策略"
        node = None  # 需要 LLM 生成，这里用简化版
    else:
        # 改进现有策略
        strategy_desc = f"改进: {parent.plan}"
        node = None

    # 简化版：直接创建节点（完整版需要 LLM 生成策略）
    from evolution import EvolutionNode
    result_node = EvolutionNode(
        strategy=strategy_desc,
        plan=f"步骤 {_evolution_step_count}: {'初始' if parent is None else '改进'}策略",
        parent=parent,
    )
    result_node.metric = single_metric
    result_node.multi_metric = multi_metric
    result_node.is_buggy = not success
    result_node.analysis = f"成功率={task_success:.2f}, 满意度={satisfaction:.2f}, 效率={efficiency:.2f}"

    journal.append(result_node)
    _evolution_step_count += 1

    # 获取最优策略
    best = journal.get_best_node()

    return json.dumps({
        "step": _evolution_step_count,
        "current_node": {
            "id": result_node.id[:8],
            "plan": result_node.plan,
            "composite_score": multi_metric.composite_score,
            "is_buggy": result_node.is_buggy,
        },
        "best_strategy": {
            "strategy": best.strategy if best else "无",
            "score": best.multi_metric.composite_score if best and best.multi_metric else 0,
            "step": best.step if best else -1,
        },
        "tree_size": len(journal),
        "good_nodes": len(journal.good_nodes),
        "buggy_nodes": len(journal.buggy_nodes),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def evolution_status() -> str:
    """
    Get current evolution tree status.

    Returns the number of nodes, best score, and tree structure.
    """
    journal, agent = _get_evolution()
    best = journal.get_best_node()

    nodes_info = []
    for n in journal.nodes:
        score = n.multi_metric.composite_score if n.multi_metric else 0
        nodes_info.append({
            "step": n.step,
            "stage": n.stage_name,
            "plan": n.plan[:50] if n.plan else "",
            "score": round(score, 3),
            "is_buggy": n.is_buggy,
        })

    return json.dumps({
        "total_nodes": len(journal),
        "good_nodes": len(journal.good_nodes),
        "buggy_nodes": len(journal.buggy_nodes),
        "best_score": best.multi_metric.composite_score if best and best.multi_metric else 0,
        "best_strategy": best.strategy if best else "无",
        "evolution_steps": _evolution_step_count,
        "nodes": nodes_info,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def evolution_best() -> str:
    """
    Get the best strategy from the evolution tree.

    Returns the highest-scoring strategy with full details.
    """
    journal, agent = _get_evolution()
    best = journal.get_best_node()

    if not best:
        return json.dumps({"error": "进化树为空，还没有策略"}, ensure_ascii=False)

    return json.dumps({
        "strategy": best.strategy,
        "plan": best.plan,
        "step": best.step,
        "composite_score": best.multi_metric.composite_score if best.multi_metric else 0,
        "metrics": {
            "task_success": best.multi_metric.task_success if best.multi_metric else 0,
            "user_satisfaction": best.multi_metric.user_satisfaction if best.multi_metric else 0,
            "efficiency": best.multi_metric.efficiency if best.multi_metric else 0,
            "safety": best.multi_metric.safety if best.multi_metric else 0,
            "creativity": best.multi_metric.creativity if best.multi_metric else 0,
        },
        "analysis": best.analysis,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def evolution_report() -> str:
    """
    Get a formatted evolution report.

    Returns a human-readable report of the evolution progress.
    """
    journal, agent = _get_evolution()
    best = journal.get_best_node()

    lines = []
    lines.append("=" * 50)
    lines.append("自进化报告")
    lines.append("=" * 50)
    lines.append(f"进化步骤: {_evolution_step_count}")
    lines.append(f"总节点数: {len(journal)}")
    lines.append(f"好节点数: {len(journal.good_nodes)}")
    lines.append(f"Buggy 节点: {len(journal.buggy_nodes)}")

    if best and best.multi_metric:
        lines.append(f"\n最优策略 (版本 {best.step}):")
        lines.append(f"  策略: {best.strategy}")
        lines.append(f"  综合得分: {best.multi_metric.composite_score:.3f}")
        m = best.multi_metric
        lines.append(f"  成功率={m.task_success:.2f} 满意度={m.user_satisfaction:.2f} "
                     f"效率={m.efficiency:.2f} 安全={m.safety:.2f} 创造={m.creativity:.2f}")
        lines.append(f"  分析: {best.analysis}")
    else:
        lines.append("\n暂无最优策略")

    lines.append("\n进化树:")
    for n in journal.nodes:
        indent = "  " if n.parent else ""
        score = n.multi_metric.composite_score if n.multi_metric else 0
        status = "✓" if not n.is_buggy else "✗"
        lines.append(f"{indent}{status} [{n.stage_name}] v{n.step}: {n.plan[:40]} ({score:.3f})")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Code Evolution (AIDE-derived code tree search)
# ═══════════════════════════════════════════════════════════════

# 代码进化持久状态（独立于策略进化的全局 journal）
_code_evolution_journal = None
_code_evolution_agent = None


def _get_code_evolution(task_desc: str = "优化代码质量") -> tuple:
    """
    延迟初始化代码进化系统。

    注意：task_desc 只在首次初始化时生效。
    如需切换任务，调用 reset_code_evolution()。
    """
    global _code_evolution_journal, _code_evolution_agent
    if _code_evolution_journal is None:
        from evolution.code_evolution import CodeEvolutionAgent
        from evolution.journal import EvolutionJournal
        _code_evolution_journal = EvolutionJournal()
        _code_evolution_agent = CodeEvolutionAgent(
            task_desc=task_desc,
            journal=_code_evolution_journal,
            model="gpt-4-turbo",
            num_drafts=3,
            debug_prob=0.2,
            explore_prob=0.15,
            eval_metric="quality",
            eval_direction="maximize",
        )
    return _code_evolution_journal, _code_evolution_agent


@mcp.tool()
def evolve_code(
    code: str,
    task_desc: str = "优化代码质量",
    mode: str = "improve",
    error_output: str = "",
    use_tree: bool = True,
) -> str:
    """
    进化指定的代码 — 在代码空间搜索更优解。

    基于 CodeEvolutionAgent (AIDE 原始代码搜索能力)：
    - improve: 对给定代码做一次原子改进
    - debug:   基于错误输出修复代码 bug
    - draft:   从任务描述生成全新代码（忽略 code 参数）

    使用独立的 _code_evolution_journal，不污染策略进化的全局进化树。
    进化结果自动加入代码进化树，便于后续连续进化。

    Args:
        code: 要进化的代码（improve/debug 模式下必填）
        task_desc: 任务描述，指导改进方向（如 "提升性能"、"降低内存占用"）
        mode: "improve" | "debug" | "draft"
        error_output: debug 模式下的错误输出或堆栈
        use_tree: True=加入进化树连续进化；False=一次性调用，不入树
    """
    if mode not in ("improve", "debug", "draft"):
        return json.dumps(
            {"error": f"mode 必须是 improve/debug/draft，收到 {mode!r}"},
            ensure_ascii=False,
        )
    if mode != "draft" and not code.strip():
        return json.dumps(
            {"error": "improve/debug 模式下 code 不能为空"},
            ensure_ascii=False,
        )

    try:
        journal, agent = _get_code_evolution(task_desc)
        # 同步 task_desc（如果用户改了任务描述）
        agent.task_desc = task_desc

        parent = None
        original_code = ""

        if mode == "draft":
            new_node = agent._draft()
            original_code = ""
        else:
            # 把用户给的代码作为父节点
            from evolution.journal import EvolutionNode
            parent = EvolutionNode(
                strategy=code,
                plan=f"用户提供的代码（{mode} 起点）",
            )
            # 如果有错误输出，注入到父节点的 term_out
            if mode == "debug" and error_output:
                parent._term_out = [error_output]

            if mode == "improve":
                new_node = agent._improve(parent)
            else:  # debug
                new_node = agent._debug(parent)

            original_code = code

        if use_tree:
            journal.append(new_node)

        best = journal.get_best_node() if use_tree else new_node

        return json.dumps({
            "mode": mode,
            "task_desc": task_desc,
            "original_code_preview": original_code[:300] + ("..." if len(original_code) > 300 else ""),
            "evolved_code": new_node.strategy,
            "plan": new_node.plan,
            "in_tree": use_tree,
            "tree_size": len(journal) if use_tree else 0,
            "best_code_preview": (best.strategy[:300] + "...") if best and len(best.strategy) > 300 else (best.strategy if best else ""),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps(
            {"error": f"{type(e).__name__}: {e}", "hint": "检查 LLM 后端 API key 是否配置正确"},
            ensure_ascii=False,
        )


@mcp.tool()
def evolve_code_status() -> str:
    """
    获取代码进化树的当前状态。

    返回节点数、最优代码片段、进化历史等信息。
    与策略进化的 evolution_status 完全独立。
    """
    if _code_evolution_journal is None:
        return json.dumps(
            {"status": "未初始化", "message": "代码进化系统还没启动，先调用 evolve_code"},
            ensure_ascii=False, indent=2,
        )

    journal = _code_evolution_journal
    best = journal.get_best_node()

    nodes_info = []
    for n in journal.nodes:
        nodes_info.append({
            "step": n.step,
            "stage": n.stage_name,
            "plan": n.plan[:80] if n.plan else "",
            "code_length": len(n.strategy) if n.strategy else 0,
            "is_buggy": n.is_buggy,
        })

    return json.dumps({
        "total_nodes": len(journal),
        "good_nodes": len(journal.good_nodes),
        "buggy_nodes": len(journal.buggy_nodes),
        "best_code_preview": (best.strategy[:500] + "...") if best and len(best.strategy) > 500 else (best.strategy if best else ""),
        "best_plan": best.plan if best else "",
        "nodes": nodes_info,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def evolve_code_best() -> str:
    """
    获取代码进化树中的最优代码（完整内容）。

    返回完整的优化后代码，可直接复制使用。
    """
    if _code_evolution_journal is None:
        return json.dumps(
            {"error": "代码进化树为空，先调用 evolve_code 进化一次"},
            ensure_ascii=False,
        )

    best = _code_evolution_journal.get_best_node()
    if not best:
        return json.dumps(
            {"error": "进化树中还没有节点"},
            ensure_ascii=False,
        )

    return json.dumps({
        "code": best.strategy,
        "plan": best.plan,
        "step": best.step,
        "analysis": best.analysis,
        "is_buggy": best.is_buggy,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def reset_code_evolution() -> str:
    """
    重置代码进化树。

    清空所有已进化的代码节点，下次调用 evolve_code 将从空白开始。
    策略进化的 evolution_status 不受影响。
    """
    global _code_evolution_journal, _code_evolution_agent
    old_count = len(_code_evolution_journal) if _code_evolution_journal else 0
    _code_evolution_journal = None
    _code_evolution_agent = None
    return json.dumps({
        "reset": True,
        "cleared_nodes": old_count,
        "message": "代码进化树已重置，下次 evolve_code 将从空白开始",
    }, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# 异步代码进化：Trae 接管 LLM 调用
# ═══════════════════════════════════════════════════════════════
# 这套工具配合 EVOLUTION_BACKEND=trae 使用：
# - evolve_code_async: 启动后台进化线程，立即返回 call_id
# - get_pending_llm: 查询待处理的 LLM 请求
# - submit_llm_response: 提交 LLM 响应
# - get_evolve_result: 查询最终进化结果

import threading
import traceback

# 全局任务注册表：call_id -> {thread, journal, agent, result, error, status}
_async_tasks: dict = {}


def _run_evolution_async(
    call_id: str,
    code: str,
    task_desc: str,
    mode: str,
    error_output: str,
):
    """在后台线程里跑进化系统"""
    task = _async_tasks[call_id]
    try:
        task["status"] = "running"
        journal, agent = _get_code_evolution(task_desc)
        agent.task_desc = task_desc

        from evolution.journal import EvolutionNode
        if mode == "draft":
            new_node = agent._draft()
        else:
            parent = EvolutionNode(
                strategy=code,
                plan=f"用户提供的代码（{mode} 起点）",
            )
            if mode == "debug" and error_output:
                parent._term_out = [error_output]
            new_node = agent._improve(parent) if mode == "improve" else agent._debug(parent)

        # FIX: 评估新节点（语法检查 + 质量评分）
        # _improve 只生成代码不评估，需要手动设置 metric
        try:
            from evolution.utils.metric import MetricValue
            compile(new_node.strategy, "<evolved>", "exec")
            new_node.is_buggy = False

            # B5 专项评分: 如果 task_desc 含 B5/causal_feature_extractor, 用 fitness 函数
            if "B5" in task_desc or "causal_feature_extractor" in task_desc:
                import tempfile, os as _os
                _tmp = _os.path.join(tempfile.gettempdir(), "b5_eval.py")
                with open(_tmp, "w", encoding="utf-8") as _f:
                    _f.write(new_node.strategy)
                import sys as _sys
                _b5_path = r"c:\Users\CC\Desktop\feeling\evolution"
                if _b5_path not in _sys.path:
                    _sys.path.insert(0, _b5_path)
                from b5_fitness import run_fitness as _b5_fitness
                _fit = _b5_fitness(_tmp)
                score = _fit.get("score", 0.0)
                new_node.metric = MetricValue(score, maximize=True)
                new_node.analysis = f"B5 fitness: score={score:.3f}  {_fit.get('details','')}"
            else:
                # 原有通用评分逻辑
                lines = new_node.strategy.strip().split('\n')
                line_count = len(lines)
                score = 0.3 + min(0.3, line_count / 1000.0)
                for method in ['def discover', 'def intervene',
                              'def counterfactual', 'def predict',
                              'def quantum_similarity']:
                    if method in new_node.strategy:
                        score += 0.04
                method_count = new_node.strategy.count('    def ')
                score += min(0.2, method_count * 0.01)
                best_so_far = journal.get_best_node()
                if best_so_far and best_so_far.metric:
                    if abs(score - best_so_far.metric.value) < 0.01:
                        score = best_so_far.metric.value + 0.001 * (line_count / 100.0)
                new_node.metric = MetricValue(score, maximize=True)
                new_node.analysis = f"语法有效,{line_count}行,{method_count}方法,评分{score:.3f}"
        except SyntaxError as e:
            new_node.is_buggy = True
            new_node.metric = None
            new_node.analysis = f"语法错误: {e}"

        journal.append(new_node)
        best = journal.get_best_node()

        task["result"] = {
            "mode": mode,
            "task_desc": task_desc,
            "evolved_code": new_node.strategy,
            "plan": new_node.plan,
            "best_code": best.strategy if best else new_node.strategy,
            "tree_size": len(journal),
        }
        task["status"] = "done"
    except Exception as e:
        task["error"] = f"{type(e).__name__}: {e}"
        task["traceback"] = traceback.format_exc()
        task["status"] = "error"


@mcp.tool()
def evolve_code_async(
    code: str,
    task_desc: str = "优化代码质量",
    mode: str = "improve",
    error_output: str = "",
) -> str:
    """
    异步启动代码进化（配合 Trae 接管 LLM 调用）。

    必须先设置环境变量 EVOLUTION_BACKEND=trae 启动 MCP server。
    本工具立即返回 call_id，进化在后台线程跑。每次需要 LLM 时，
    进化系统会阻塞等待 Trae 提交响应——用 get_pending_llm 查询。

    Args:
        code: 要进化的代码
        task_desc: 任务描述
        mode: "improve" | "debug" | "draft"
        error_output: debug 模式下的错误输出

    Returns:
        call_id: 任务 ID，后续用于查询 pending / 提交 response / 取结果
    """
    import uuid
    call_id = uuid.uuid4().hex[:12]
    _async_tasks[call_id] = {
        "thread": None,
        "status": "starting",
        "result": None,
        "error": None,
        "traceback": None,
    }
    t = threading.Thread(
        target=_run_evolution_async,
        args=(call_id, code, task_desc, mode, error_output),
        daemon=True,
    )
    _async_tasks[call_id]["thread"] = t
    t.start()
    return json.dumps({
        "call_id": call_id,
        "status": "started",
        "message": "进化已在后台启动。调用 get_pending_llm 查询 LLM 请求。",
        "hint": "需要 EVOLUTION_BACKEND=trae 才能让 Trae 接管 LLM",
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def get_pending_llm(call_id: str = "") -> str:
    """
    查询待处理的 LLM 请求。

    进化系统在后台跑时，每次需要 LLM 会写一个 pending 文件。
    本工具读取所有 pending 请求，返回给 Trae 处理。

    Args:
        call_id: 可选，只返回指定任务的 pending；空则返回所有

    Returns:
        pending 请求列表，每个含 call_id_llm / system_message / user_message / func_spec
    """
    try:
        from evolution.backend.backend_trae import list_pending
        pending = list_pending()
        # 可选过滤
        if call_id:
            # 注意：这里的 call_id 是进化任务 ID，pending 里的 call_id 是 LLM 调用 ID
            # 不直接对应，返回全部即可
            pass
        if not pending:
            # 检查进化任务状态
            status_info = {}
            for cid, task in _async_tasks.items():
                status_info[cid] = task["status"]
            return json.dumps({
                "pending": [],
                "message": "没有待处理的 LLM 请求",
                "task_status": status_info,
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "pending": pending,
            "count": len(pending),
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


@mcp.tool()
def submit_llm_response(
    call_id_llm: str,
    output: str,
    in_tokens: int = 0,
    out_tokens: int = 0,
) -> str:
    """
    提交 LLM 响应给进化系统。

    Trae 处理完 pending 请求后，用本工具把结果回填。进化系统会
    解除阻塞，继续执行下一步。

    Args:
        call_id_llm: pending 请求里的 call_id（LLM 调用 ID，不是进化任务 ID）
        output: LLM 生成的回复（字符串，或 JSON 字符串如果有 func_spec）
        in_tokens: 输入 token 数（可选）
        out_tokens: 输出 token 数（可选）
    """
    try:
        from evolution.backend.backend_trae import submit_response
        ok = submit_response(call_id_llm, output, in_tokens, out_tokens)
        return json.dumps({
            "submitted": ok,
            "call_id_llm": call_id_llm,
            "message": "已提交，进化系统将继续执行" if ok else "提交失败：找不到对应的 pending 请求",
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


@mcp.tool()
def get_evolve_result(call_id: str, wait: bool = False) -> str:
    """
    查询异步进化任务的结果。

    Args:
        call_id: evolve_code_async 返回的任务 ID
        wait: True=阻塞等待完成（最多 60s）；False=立即返回当前状态
    """
    if call_id not in _async_tasks:
        return json.dumps({"error": f"未知 call_id: {call_id}"}, ensure_ascii=False)

    task = _async_tasks[call_id]
    if wait:
        task["thread"].join(timeout=60)

    resp = {"call_id": call_id, "status": task["status"]}
    if task["status"] == "done":
        resp["result"] = task["result"]
    elif task["status"] == "error":
        resp["error"] = task["error"]
        resp["traceback"] = task.get("traceback", "")
    return json.dumps(resp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAAP Brain MCP Server")
    parser.add_argument("--sse", action="store_true", help="Run in SSE mode")
    parser.add_argument("--port", type=int, default=11550, help="SSE port")
    args = parser.parse_args()

    if args.port != 11550:
        mcp.settings.port = args.port

    if args.sse:
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
