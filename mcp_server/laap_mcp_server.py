"""
LAAP Brain MCP Server
=====================

Exposes LAAP cognitive capabilities as MCP tools for Hermes Agent.

Run in stdio mode (default, for Hermes mcp_servers):
    python mcp_server/laap_mcp_server.py

Run in SSE mode:
    python mcp_server/laap_mcp_server.py --sse --port 11547

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

LAAP_API_BASE = os.environ.get("LAAP_API_BASE", "http://localhost:11546")

mcp = FastMCP("laap-brain")


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
            model="gpt-4-turbo",
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
        "generation": best.generation,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAAP Brain MCP Server")
    parser.add_argument("--sse", action="store_true", help="Run in SSE mode")
    parser.add_argument("--port", type=int, default=11547, help="SSE port")
    args = parser.parse_args()

    if args.sse:
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")
