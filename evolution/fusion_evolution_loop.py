"""
融合引擎持续进化循环
====================

自动评估 → 生成改进方案 → 应用改进 → 重新评估 → 循环

支持两种 LLM 模式:
  1. 正常模式: 通过 llm_gateway 调用外部 LLM
  2. 握手模式: 通过文件协议让人工/OpenClaw 接管 LLM

用法:
    # 启动进化循环（自动模式）
    python fusion_evolution_loop.py --rounds 10

    # 启动进化循环（握手模式，LLM 不可用时）
    python fusion_evolution_loop.py --rounds 10 --handshake

    # 只评估当前版本
    python fusion_evolution_loop.py --eval-only

印记: 小茜 永远记得主人 — 2026-07-20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# 确保路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evolution.evaluator import (
    evaluate_fusion_engine,
    save_eval,
    load_eval_history,
    compare_versions,
    EvalResult,
)

logger = logging.getLogger("evolution.loop")


def get_llm_backend(use_handshake: bool = False):
    """
    获取 LLM 后端。

    Args:
        use_handshake: 是否使用握手模式

    Returns:
        query 函数
    """
    if use_handshake:
        from evolution.llm_handshake import query_for_evolution
        logger.info("Using handshake mode for LLM")
        return query_for_evolution

    # 尝试正常 gateway
    try:
        from laap_brain.llm_gateway import llm_call
        # 测试连通性
        llm_call([{"role": "user", "content": "ping"}], purpose="evolution")
        logger.info("Using llm_gateway for LLM")

        def gateway_query(system_message, user_message, **kwargs):
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_message})
            return llm_call(messages, purpose="evolution")

        return gateway_query
    except Exception as e:
        logger.warning(f"llm_gateway unavailable ({e}), falling back to handshake")
        from evolution.llm_handshake import query_for_evolution
        return query_for_evolution


def generate_improvements(
    eval_result: EvalResult,
    llm_query,
    fusion_engine_code: str,
) -> list:
    """
    基于评估结果生成改进建议。

    Args:
        eval_result: 当前评估结果
        llm_query: LLM 查询函数
        fusion_engine_code: 当前融合引擎代码

    Returns:
        改进建议列表
    """
    score = eval_result.score

    prompt = f"""你是一个代码优化专家。请分析以下融合引擎的评估结果和代码，生成具体的改进方案。

## 当前评估分数
- 准确性: {score.accuracy:.2f}
- 延迟: {score.latency_ms:.1f}ms
- 鲁棒性: {score.robustness:.2f}
- 覆盖度: {score.coverage:.2f}
- 融合质量: {score.fusion_quality:.2f}
- 综合得分: {score.composite:.4f}

## 失败的测试用例
{json.dumps([d for d in eval_result.details if not d.get("passed")], ensure_ascii=False, indent=2)[:1000]}

## 当前代码 (前200行)
{fusion_engine_code[:3000]}

## 要求
请生成 3-5 个具体的改进方案，每个方案包含:
1. 问题描述
2. 改进方法
3. 具体代码改动
4. 预期效果

用 JSON 格式输出，格式如下:
[
  {{
    "problem": "问题描述",
    "solution": "改进方法",
    "code_changes": "具体代码改动",
    "expected_improvement": "预期效果"
  }}
]"""

    try:
        response = llm_query(
            system_message="你是代码优化专家，专注于认知引擎和NLP融合系统。",
            user_message=prompt,
        )

        # 尝试解析 JSON
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            logger.warning("Failed to parse improvements JSON")
            return [{"problem": "解析失败", "solution": response[:500]}]
    except Exception as e:
        logger.error(f"Failed to generate improvements: {e}")
        return []


def apply_improvements(improvements: list, engine_path: Path) -> bool:
    """
    应用改进方案到融合引擎代码。

    这是简化版 — 只记录改进方案，实际代码修改需要人工确认。
    完整版会自动生成补丁并测试。

    Args:
        improvements: 改进方案列表
        engine_path: 融合引擎文件路径

    Returns:
        是否有可应用的改进
    """
    # 保存改进方案
    improvements_path = engine_path.parent / "pending_improvements.json"
    improvements_path.write_text(
        json.dumps(improvements, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Improvements saved to {improvements_path}")
    return len(improvements) > 0


def run_evolution_loop(
    rounds: int = 10,
    use_handshake: bool = False,
    eval_only: bool = False,
):
    """
    运行进化循环。

    Args:
        rounds: 进化轮数
        use_handshake: 是否使用握手模式
        eval_only: 是否只评估不进化
    """
    # 导入融合引擎
    sys.path.insert(0, str(ROOT / "aris_brain"))

    try:
        import aris_brain.aris_fusion_engine as engine
    except ImportError:
        logger.error("Cannot import fusion engine")
        return

    engine_path = ROOT / "aris_brain" / "aris_fusion_engine.py"
    engine_code = engine_path.read_text(encoding="utf-8") if engine_path.exists() else ""

    # 评估当前版本
    logger.info("=== Evaluating current fusion engine ===")
    current_eval = evaluate_fusion_engine(engine, version="v0")
    save_eval(current_eval)
    logger.info(f"Current score: {current_eval.score.composite:.4f}")
    logger.info(f"  accuracy={current_eval.score.accuracy:.2f} latency={current_eval.score.latency_ms:.1f}ms")
    logger.info(f"  robustness={current_eval.score.robustness:.2f} coverage={current_eval.score.coverage:.2f}")

    if eval_only:
        return

    # 获取 LLM 后端
    llm_query = get_llm_backend(use_handshake)

    # 进化循环
    for round_num in range(1, rounds + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"=== Evolution Round {round_num}/{rounds} ===")
        logger.info(f"{'='*50}")

        # 1. 生成改进方案
        logger.info("Generating improvements...")
        improvements = generate_improvements(current_eval, llm_query, engine_code)

        if not improvements:
            logger.warning("No improvements generated, skipping round")
            continue

        logger.info(f"Generated {len(improvements)} improvements:")
        for i, imp in enumerate(improvements):
            logger.info(f"  {i+1}. {imp.get('problem', '?')[:60]}")

        # 2. 应用改进（保存待确认）
        if apply_improvements(improvements, engine_path):
            logger.info("Improvements pending. Apply them and re-run to see score changes.")

        # 3. 如果是握手模式，等待人工确认
        if use_handshake:
            logger.info("Waiting for human to apply improvements and confirm...")
            from evolution.llm_handshake import handshake_query
            confirm = handshake_query(
                f"已生成 {len(improvements)} 个改进方案。请应用后回复 'done' 继续进化。",
                system="进化循环控制",
                timeout=3600,
            )
            if confirm and "done" in confirm.lower():
                # 重新评估
                import importlib
                importlib.reload(engine)
                current_eval = evaluate_fusion_engine(engine, version=f"v{round_num}")
                save_eval(current_eval)
                logger.info(f"New score: {current_eval.score.composite:.4f}")
            else:
                logger.info("Evolution paused by human.")
                break
        else:
            # 自动模式：记录结果，下一轮继续
            logger.info(f"Round {round_num} complete. Improvements saved for review.")

    # 最终报告
    logger.info(f"\n{'='*50}")
    logger.info("=== Evolution Summary ===")
    logger.info(f"{'='*50}")

    history = load_eval_history()
    if len(history) >= 2:
        comparison = compare_versions(history[0].version, history[-1].version)
        if comparison:
            logger.info(f"Version: {history[0].version} → {history[-1].version}")
            logger.info(f"Score: {comparison['v1']['score']['composite']:.4f} → {comparison['v2']['score']['composite']:.4f}")
            logger.info(f"Delta: {comparison['delta']['composite']:+.4f}")
    else:
        logger.info(f"Current score: {current_eval.score.composite:.4f}")


# ── CLI ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="融合引擎进化循环")
    parser.add_argument("--rounds", type=int, default=10, help="进化轮数")
    parser.add_argument("--handshake", action="store_true", help="使用握手模式（LLM不可用时）")
    parser.add_argument("--eval-only", action="store_true", help="只评估不进化")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_evolution_loop(
        rounds=args.rounds,
        use_handshake=args.handshake,
        eval_only=args.eval_only,
    )
