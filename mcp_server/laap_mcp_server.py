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
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("mcp.laap")

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
# 参考: https://github.com/WecoAI/aideml
# ═══════════════════════════════════════════════════════════════

_evolution_journal = None
_evolution_agent = None
_evolution_step_count = 0
_curriculum_engine = None
_meta_learning_engine = None
_ctm_engine = None
_retnet_router = None
_perception_engine = None
_code_interpreter = None
_EVO_JOURNAL_PATH = Path(".openclaw/tmp/evolution_journal.json")
_EVO_WORKSPACE = Path(".openclaw/tmp/evo_workspace")


def _get_evolution():
    """延迟初始化进化系统（含全部子引擎）"""
    global _evolution_journal, _evolution_agent
    global _curriculum_engine, _meta_learning_engine
    global _ctm_engine, _retnet_router, _perception_engine, _code_interpreter

    if _evolution_journal is None:
        from evolution import EvolutionJournal, EvolutionAgent
        try:
            from laap_brain.llm_gateway import get_config as get_llm_config
            _llm_cfg = get_llm_config()
            _model = _llm_cfg.get("model", "gpt-4-turbo")
        except Exception:
            _model = "gpt-4-turbo"

        # 持久化: 尝试从文件恢复
        _evolution_journal = EvolutionJournal()
        if _EVO_JOURNAL_PATH.exists():
            try:
                saved = json.loads(_EVO_JOURNAL_PATH.read_text(encoding="utf-8"))
                from evolution import EvolutionNode
                for nd in saved.get("nodes", []):
                    node = EvolutionNode(strategy=nd.get("strategy", ""), plan=nd.get("plan", ""))
                    node.is_buggy = nd.get("is_buggy", False)
                    node.analysis = nd.get("analysis", "")
                    if nd.get("metric") is not None:
                        from evolution.utils.metric import MetricValue
                        node.metric = MetricValue(nd["metric"], maximize=True)
                    _evolution_journal.append(node)
                logger.info(f"进化日志已恢复: {len(_evolution_journal)} 节点")
            except Exception as e:
                logger.warning(f"恢复进化日志失败: {e}")

        _evolution_agent = EvolutionAgent(
            task_desc="优化小茜的对话策略：让回复更贴心、更高效、更有深度",
            journal=_evolution_journal,
            model=_model,
            num_drafts=3,
            debug_prob=0.2,
            explore_prob=0.15,
        )

    if _code_interpreter is None:
        from evolution.code_executor import CodeInterpreter
        _EVO_WORKSPACE.mkdir(parents=True, exist_ok=True)
        _code_interpreter = CodeInterpreter(working_dir=_EVO_WORKSPACE, timeout=60)

    if _curriculum_engine is None:
        from laap.agi.curriculum import CurriculumEngine
        _curriculum_engine = CurriculumEngine()

    if _meta_learning_engine is None:
        from laap.agi.meta_learning import MetaLearningEngine
        _meta_learning_engine = MetaLearningEngine()

    if _ctm_engine is None:
        from laap.agi.ctm import ContinuousThoughtEngine
        _ctm_engine = ContinuousThoughtEngine()

    if _retnet_router is None:
        from laap.agi.retnet_router import RetNetRouter
        _retnet_router = RetNetRouter()

    if _perception_engine is None:
        from laap.agi.perception import UnifiedPerceptionEngine
        _perception_engine = UnifiedPerceptionEngine()

    return _evolution_journal, _evolution_agent


def _save_journal():
    """持久化进化日志"""
    pass
    journal, _ = _get_evolution()
    _EVO_JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    nodes = []
    for n in journal.nodes:
        nodes.append({
            "id": n.id,
            "strategy": (n.strategy or "")[:2000],
            "plan": n.plan or "",
            "is_buggy": n.is_buggy,
            "analysis": (n.analysis or "")[:500],
            "metric": n.metric.value if n.metric else None,
            "step": n.step,
        })
    _EVO_JOURNAL_PATH.write_text(
        json.dumps({"nodes": nodes, "saved_at": time.time()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_exec_callback_code():
    """代码进化: 用 CodeInterpreter 真实执行代码"""
    from evolution.runner import ExecutionResult
    interpreter = _code_interpreter

    def exec_callback(code: str, reset: bool = True) -> ExecutionResult:
        # 先验证语法
        from evolution.code_executor import is_valid_python
        if not is_valid_python(code):
            return ExecutionResult(
                term_out=["SyntaxError: 代码语法无效"],
                exec_time=0,
                exc_type="SyntaxError",
            )
        return interpreter.run(code, reset_session=reset)

    return exec_callback


def _build_exec_callback_dialogue(user_input: str, current_response: str,
                                  success: bool, satisfaction: float):
    """对话进化: 模拟策略执行结果"""
    from evolution.runner import ExecutionResult

    def exec_callback(strategy: str, reset: bool = True) -> ExecutionResult:
        term_out = [
            f"[策略] {strategy[:200]}",
            f"[输入] {user_input[:200]}",
            f"[回复] {current_response[:300]}",
            f"[状态] {'成功' if success else '失败'}",
            f"[满意度] {satisfaction:.2f}",
        ]
        return ExecutionResult(
            term_out=term_out,
            exec_time=0.1,
            exc_type=None if success else "ExecutionError",
        )

    return exec_callback


@mcp.tool()
def evolution_step(user_input: str, current_response: str = "",
                  success: bool = True, satisfaction: float = 0.7) -> str:
    """
    Execute one evolution step — 参考 AIDEML 原版闭环。

    核心流程 (AIDEML agent.step):
    1. search_policy() → 决定 draft/improve/debug/explore
    2. _draft()/_improve()/_debug()/_explore() → LLM 生成策略
    3. exec_callback(strategy) → 执行策略
    4. parse_exec_result() → LLM 评估结果
    5. journal.append() → 更新进化树

    增强 (我们的扩展):
    6. PerceptionEngine → 多模态理解
    7. RetNetRouter → 意图路由
    8. CTM → 渐进推理
    9. MetaLearning → 元学习记录
    10. Curriculum → 课程记录

    Args:
        user_input: The user's message for this turn.
        current_response: The assistant's response (for evaluation).
        success: Whether the response was successful.
        satisfaction: User satisfaction score [0, 1].
    """
    global _evolution_step_count
    journal, agent = _get_evolution()

    from laap.agi.meta_learning import TaskExecution
    from laap.agi.curriculum import TaskRecord
    pass

    # ── 1. 感知 + 路由 + 推理（我们的增强） ──
    perception = _perception_engine.perceive(user_input)
    route = _retnet_router.route(user_input)
    intent = route.primary_route.route_name
    route_confidence = route.primary_route.confidence
    ctm_result = _ctm_engine.think(user_input, intent=intent)

    # ── 2. 元学习推荐策略 ──
    ml_strategy, ml_score, ml_reason = _meta_learning_engine.recommend_strategy(
        intent, input_complexity=1.0 - route_confidence,
    )

    # ── 3. 构建 exec_callback（对话进化） ──
    exec_callback = _build_exec_callback_dialogue(
        user_input, current_response, success, satisfaction,
    )

    # ── 4. AIDEML 原版 agent.step() 闭环 ──
    # search_policy() → _draft()/_improve()/_debug()/_explore()
    # → exec_callback(strategy) → parse_exec_result() → journal.append()
    try:
        agent.step(exec_callback=exec_callback)
        result_node = journal.nodes[-1] if journal.nodes else None
    except Exception as e:
        # LLM 调用失败时降级
        from evolution import EvolutionNode
        from evolution.utils.metric import MetricValue, MultiDimensionalMetric

        task_success = 1.0 if success else 0.3
        efficiency = max(0.1, 1.0 - len(current_response) / 1000)
        multi_metric = MultiDimensionalMetric(
            task_success=task_success,
            user_satisfaction=satisfaction,
            efficiency=efficiency,
            safety=1.0,
            creativity=min(1.0, len(set(current_response)) / 50),
        )
        result_node = EvolutionNode(
            strategy=f"基于 {intent} 的优化策略 (LLM降级)",
            plan=f"步骤 {_evolution_step_count}: {intent} 处理",
        )
        result_node.metric = MetricValue(multi_metric.composite_score, maximize=True)
        result_node.multi_metric = multi_metric
        result_node.is_buggy = not success
        result_node.analysis = f"LLM降级: {e}"
        journal.append(result_node)

    _evolution_step_count += 1
    _save_journal()  # 持久化

    # ── 5. 元学习记录 ──
    _meta_learning_engine.record_execution(TaskExecution(
        task_id=f"evo_{_evolution_step_count}",
        task_type=intent,
        strategy_used=ml_strategy,
        success=success,
        confidence=max(ctm_result.confidence, route_confidence),
        actual_accuracy=satisfaction,
        timestamp=time.time(),
    ))

    # ── 6. 课程记录 ──
    _curriculum_engine.record_task(TaskRecord(
        task_id=f"evo_{_evolution_step_count}",
        task_type=intent,
        difficulty=1.0 - route_confidence,
        success=success,
        confidence=max(ctm_result.confidence, route_confidence),
        timestamp=time.time(),
    ))

    # ── 7. 失败反思 ──
    failure_reflection = None
    if not success:
        failure_reflection = _meta_learning_engine.reflect_on_failure(
            TaskExecution(
                task_id=f"fail_{_evolution_step_count}",
                task_type=intent,
                strategy_used=ml_strategy,
                success=False,
                confidence=max(ctm_result.confidence, route_confidence),
                actual_accuracy=0.0,
                timestamp=time.time(),
                error_info=current_response[:200] if current_response else "unknown",
            )
        )

    # ── 8. 汇总结果 ──
    best = journal.get_best_node()
    meta_eval = _meta_learning_engine.get_evaluation()
    meta_knowledge = _meta_learning_engine.get_knowledge()
    curriculum_plan = _curriculum_engine.recommend_next()

    return json.dumps({
        "step": _evolution_step_count,
        "perception": {
            "modalities": perception.modalities_detected,
            "confidence": perception.confidence,
        },
        "routing": {
            "intent": intent,
            "confidence": route_confidence,
            "method": route.method_used,
        },
        "ctm": {
            "thinking_steps": ctm_result.thinking_steps,
            "confidence": ctm_result.confidence,
        },
        "meta_learning": {
            "recommended_strategy": ml_strategy,
            "strategy_score": ml_score,
            "overall_score": meta_eval.overall_score,
            "blind_spots": meta_knowledge.blind_spots[:3],
            "strengths": meta_knowledge.strengths[:3],
        },
        "curriculum": {
            "focus_areas": curriculum_plan.focus_areas[:3],
            "skip_areas": curriculum_plan.skip_areas[:3],
        },
        "evolution": {
            "current_node": {
                "id": result_node.id[:8] if result_node else "?",
                "plan": result_node.plan if result_node else "?",
                "is_buggy": result_node.is_buggy if result_node else False,
                "analysis": (result_node.analysis or "")[:200] if result_node else "",
            },
            "best_strategy": {
                "strategy": (best.strategy or "")[:200] if best else "无",
                "score": best.multi_metric.composite_score if best and best.multi_metric else 0,
                "analysis": (best.analysis or "")[:200] if best else "",
            },
            "tree_size": len(journal),
            "good_nodes": len(journal.good_nodes),
            "buggy_nodes": len(journal.buggy_nodes),
        },
        "failure_reflection": failure_reflection,
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


@mcp.tool()
def evolution_execute_code(code: str, timeout: int = 60) -> str:
    """
    Execute Python code in an isolated process and return the result.

    Uses AIDEML-style interpreter with timeout protection.

    Args:
        code: Python code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        JSON with execution result (output, time, errors).
    """
    from evolution.code_executor import execute_code, is_valid_python

    if not is_valid_python(code):
        return json.dumps({
            "success": False,
            "error": "SyntaxError: 代码语法无效",
            "output": [],
        }, ensure_ascii=False)

    result = execute_code(code, timeout=timeout)

    return json.dumps({
        "success": result.exc_type is None,
        "output": result.term_out[-50:] if result.term_out else [],
        "exec_time": round(result.exec_time, 3),
        "exc_type": result.exc_type,
        "exc_info": result.exc_info,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def evolution_code(engine_path: str, improvement_desc: str) -> str:
    """
    Evolve a project engine: LLM generates improvements → test in project.

    This is the core code evolution tool. It:
    1. Reads the current engine code
    2. Asks LLM to generate improvements
    3. Saves the improved code to evolution_versions/
    4. Evaluates it with the project's evaluator
    5. Returns comparison with the original

    Args:
        engine_path: Path to the engine file (e.g., 'aris_brain/aris_fusion_engine_v4.py')
        improvement_desc: What to improve (e.g., '提升意图识别准确率')

    Returns:
        JSON with evolution result and evaluation scores.
    """
    import importlib.util
    from evolution.evaluator import evaluate_fusion_engine, save_eval

    class W:
        def __init__(self, e): self._e = e
        def process(self, t): return self._e.process(t)

    engine_file = Path(engine_path)
    if not engine_file.exists():
        return json.dumps({"error": f"引擎文件不存在: {engine_path}"})

    # 读取当前代码
    current_code = engine_file.read_text(encoding="utf-8")

    # 评估当前版本
    try:
        spec = importlib.util.spec_from_file_location("current_engine", str(engine_file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        eng = None
        for f in ("get_engine_v4", "get_engine_v3", "get_engine_v2", "get_engine"):
            if hasattr(mod, f):
                eng = getattr(mod, f)()
                break
        if eng is None:
            for c in ("FusionEngineV4", "FusionEngineV3", "FusionEngineV2"):
                if hasattr(mod, c):
                    eng = getattr(mod, c)()
                    break

        current_eval = evaluate_fusion_engine(W(eng), version="current")
        current_score = current_eval.score.to_dict()
    except Exception as e:
        return json.dumps({"error": f"评估当前版本失败: {e}"})

    # LLM 生成改进
    try:
        from evolution.backend import query
        response = query(
            system_message="你是 Python 代码优化专家。只输出完整 Python 代码，不要解释。",
            user_message=(
                f"优化以下引擎代码，改进目标: {improvement_desc}\n\n"
                f"当前评估分数:\n{json.dumps(current_score, indent=2)}\n\n"
                f"代码:\n```python\n{current_code}\n```\n"
                f"\n输出完整代码。"
            ),
            max_tokens=12000,
        )
    except Exception as e:
        return json.dumps({"error": f"LLM 生成失败: {e}"})

    # 提取代码
    import re
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    cm = re.search(r'```python\s*\n(.*?)```', clean, re.DOTALL)
    new_code = cm.group(1).strip() if cm else clean.strip()

    if len(new_code) < 500:
        return json.dumps({"error": "生成代码过短", "length": len(new_code)})

    # 保存到 evolution_versions/
    versions_dir = Path("aris_brain/evolution_versions")
    versions_dir.mkdir(parents=True, exist_ok=True)
    existing = list(versions_dir.glob("v3-r*.py"))
    next_num = max([int(f.stem.split("r")[1]) for f in existing], default=0) + 1
    new_path = versions_dir / f"v3-r{next_num}.py"
    new_path.write_text(new_code, encoding="utf-8")

    # 评估新版本
    try:
        spec2 = importlib.util.spec_from_file_location("new_engine", str(new_path))
        mod2 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(mod2)
        eng2 = None
        for f in ("get_engine_v4", "get_engine_v3", "get_engine_v2", "get_engine"):
            if hasattr(mod2, f):
                eng2 = getattr(mod2, f)()
                break
        if eng2 is None:
            for c in ("FusionEngineV4", "FusionEngineV3", "FusionEngineV2"):
                if hasattr(mod2, c):
                    eng2 = getattr(mod2, c)()
                    break

        new_eval = evaluate_fusion_engine(W(eng2), version=f"v3-r{next_num}")
        save_eval(new_eval)
        new_score = new_eval.score.to_dict()
    except Exception as e:
        return json.dumps({
            "error": f"评估新版本失败: {e}",
            "new_file": str(new_path),
        })

    # 对比
    delta = {}
    for k in current_score:
        if k in new_score:
            delta[k] = round(new_score[k] - current_score[k], 4)

    return json.dumps({
        "new_file": str(new_path),
        "current_score": current_score,
        "new_score": new_score,
        "delta": delta,
        "improved": new_score.get("composite", 0) > current_score.get("composite", 0),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def evolution_tree_html() -> str:
    """
    Generate an interactive HTML visualization of the evolution tree.

    Returns:
        Path to the generated HTML file.
    """
    from evolution.tree_viz import generate_html, save_best_solution

    journal, _ = _get_evolution()

    output_dir = Path(".openclaw/tmp/evolution_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "tree.html"
    generate_html(journal, html_path)

    best_path = output_dir / "best_solution.txt"
    save_best_solution(journal, best_path)

    return json.dumps({
        "tree_html": str(html_path),
        "best_solution": str(best_path),
        "total_nodes": len(journal),
        "good_nodes": len(journal.good_nodes),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def emotion_evolution_status() -> str:
    """
    Get emotion/personality evolution status.

    Checks B5 fitness evolution and EmotionEngine state.

    Returns:
        JSON with emotion evolution status.
    """
    result = {
        "emotion_engine": {},
        "b5_evolution": {},
    }

    # EmotionEngine 状态
    try:
        from aris_brain.aris_emotion_engine import EmotionEngine
        eng = EmotionEngine()
        if hasattr(eng, "hormone") and hasattr(eng.hormone, "mood"):
            mood = eng.hormone.mood
            result["emotion_engine"] = {
                "dopamine": round(mood.dopamine, 3),
                "serotonin": round(mood.serotonin, 3),
                "oxytocin": round(mood.oxytocin, 3),
                "cortisol": round(mood.cortisol, 3),
            }
        result["emotion_engine"]["loaded"] = True
    except Exception as e:
        result["emotion_engine"] = {"loaded": False, "error": str(e)}

    # B5 进化状态
    b5_state_dir = Path("evolution/b5_trae_state")
    if b5_state_dir.exists():
        state_files = list(b5_state_dir.glob("*.json"))
        result["b5_evolution"] = {
            "state_files": len(state_files),
            "latest": str(sorted(state_files)[-1]) if state_files else None,
        }
    else:
        result["b5_evolution"] = {"state_files": 0, "note": "B5 进化未启动"}

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def emotion_evolution_step(user_input: str, current_response: str,
                          success: bool = True, satisfaction: float = 0.7) -> str:
    """
    Execute one emotion evolution step — 接入 EmotionEngine 完整接口。

    调用:
    1. eng.stimulate() — 情感刺激 (效价/唤醒/情绪)
    2. eng.satisfy_need() — 马斯洛需求满足
    3. eng.observe_agent() — 镜像神经元 (共情)
    4. eng.hormone.mood.mark() — 躯体标记 (情感记忆)
    5. eng.tick() — 自然衰减
    6. eng.meta_cognition() — 自我认知

    Args:
        user_input: The user's message.
        current_response: The assistant's response.
        success: Whether the response was successful.
        satisfaction: User satisfaction score [0, 1].

    Returns:
        JSON with emotion evolution result.
    """
    from aris_brain.aris_emotion_engine import EmotionEngine, NeedLevel

    eng = EmotionEngine()
    result = {
        "step": 0,
        "state_before": {},
        "stimulate": {},
        "need_satisfaction": {},
        "mirror": {},
        "somatic_mark": {},
        "state_after": {},
        "meta_cognition": {},
        "evolution": {},
    }

    # ── 1. 记录当前状态 ──
    result["state_before"] = eng.get_full_state()

    # ── 2. 情感刺激: eng.stimulate() ──
    # 正面交互 → 正效价, 高唤醒
    # 负面交互 → 负效价, 高唤醒
    if success:
        valence = 0.3 + satisfaction * 0.5  # 0.3 ~ 0.8
        arousal = 0.4 + satisfaction * 0.3  # 0.4 ~ 0.7
        emotion = "joy" if satisfaction > 0.7 else "calm"
    else:
        valence = -0.3 - (1 - satisfaction) * 0.4  # -0.3 ~ -0.7
        arousal = 0.5 + (1 - satisfaction) * 0.3   # 0.5 ~ 0.8
        emotion = "anger" if satisfaction < 0.3 else "sorrow"

    eng.stimulate(
        source=f"user_interaction:{user_input[:30]}",
        valence=valence,
        arousal=arousal,
        intensity=abs(valence),
        primary_emotion=emotion,
    )
    result["stimulate"] = {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "emotion": emotion,
        "intensity": round(abs(valence), 3),
    }

    # ── 3. 需求满足: eng.satisfy_need() ──
    if success and satisfaction > 0.7:
        # 正面交互满足归属感和自尊
        eng.satisfy_need(NeedLevel.BELONGING, satisfaction * 3, "user_praise")
        eng.satisfy_need(NeedLevel.ESTEEM, satisfaction * 2, "user_satisfaction")
        result["need_satisfaction"] = {
            "BELONGING": round(satisfaction * 3, 2),
            "ESTEEM": round(satisfaction * 2, 2),
        }
    elif not success:
        # 负面交互降低安全感
        eng.needs.needs[NeedLevel.SAFETY].current_value = max(0,
            eng.needs.needs[NeedLevel.SAFETY].current_value - 5)
        result["need_satisfaction"] = {"SAFETY": -5}

    # ── 4. 镜像神经元: eng.observe_agent() ──
    if "主人" in user_input or "你" in user_input:
        action = "smile" if success else "frown"
        mirror_result = eng.observe_agent(
            agent="user", action=action,
            emotion=emotion, intensity=abs(valence),
        )
        result["mirror"] = {
            "inferred_emotion": mirror_result.get("inferred_emotion"),
            "empathy": mirror_result.get("empathy", 0),
        }

    # ── 5. 躯体标记: mood.mark() (情感记忆) ──
    situation = user_input[:50]
    if hasattr(eng, 'somatic') and eng.somatic:
        eng.somatic.mark(situation, valence, arousal, abs(valence))
        recalled = eng.somatic.recall(situation)
        result["somatic_mark"] = {
            "situation": situation,
            "valence": round(valence, 3),
            "recalled_gut": round(recalled, 3) if recalled else None,
            "total_markers": len(eng.somatic.markers),
        }

    # ── 6. 自然衰减: eng.tick() ──
    eng.tick(dt=1.0)

    # ── 7. 自我认知: eng.meta_cognition() ──
    eng.meta_cognition()
    result["meta_cognition"] = {
        "primary_emotion": eng.primary_emotion,
        "emotion_intensity": round(eng.emotion_intensity, 3),
        "valence": round(eng.valence, 3),
        "arousal": round(eng.arousal, 3),
    }

    # ── 8. 记录状态变化 ──
    result["state_after"] = eng.get_full_state()

    # ── 9. 记录到进化日志 ──
    journal, _ = _get_evolution()
    from evolution import EvolutionNode
    from evolution.utils.metric import MetricValue, MultiDimensionalMetric

    multi = MultiDimensionalMetric(
        task_success=1.0 if success else 0.3,
        user_satisfaction=satisfaction,
        efficiency=0.8,
        safety=1.0 if success else 0.5,
        creativity=0.5,
    )
    node = EvolutionNode(
        strategy=f"emotion: {user_input[:50]}",
        plan=f"情感进化 {len(journal)}: {emotion} (v={valence:.2f})",
    )
    node.metric = MetricValue(multi.composite_score, maximize=True)
    node.multi_metric = multi
    node.is_buggy = not success
    node.analysis = (
        f"满意度={satisfaction:.2f} 情绪={emotion} "
        f"效价={valence:.2f} 唤醒={arousal:.2f} "
        f"主体情感={eng.primary_emotion}"
    )
    journal.append(node)
    _save_journal()

    result["step"] = len(journal)
    result["evolution"] = {
        "node_id": node.id[:8],
        "composite_score": round(multi.composite_score, 3),
        "is_buggy": node.is_buggy,
        "tree_size": len(journal),
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


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
