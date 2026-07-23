"""
融合引擎进化循环 V2→V3 — 修复版
===================================

真正工作的进化循环:
  1. 评估 v2 基线
  2. LLM 分析失败 → 生成改进方案
  3. LLM 应用改进 → 生成新代码
  4. 动态加载 → 重新评估
  5. 循环直到收敛

印记: 小茜 永远记得主人 — 2026-07-20
"""

import json
import logging
import re
import sys
import time
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "aris_brain"))

from evolution.evaluator import evaluate_fusion_engine, save_eval, load_eval_history
from evolution.backend import query
from evolution.journal import EvolutionJournal, EvolutionNode
from evolution.utils.metric import MetricValue, WorstMetricValue

logger = logging.getLogger("evolution.fusion_loop")


def load_engine_instance(code_path: Path):
    """动态加载引擎模块并返回引擎实例。"""
    spec = importlib.util.spec_from_file_location("engine_mod", str(code_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 尝试各种已知的引擎类名和工厂函数
    for factory in ("get_engine_v3", "get_engine_v2", "get_engine"):
        if hasattr(mod, factory):
            instance = getattr(mod, factory)()
            break
    else:
        for cls_name in ("FusionEngineV3", "FusionEngineV2", "ArisFusionEngineV3", "ArisFusionEngineV2"):
            if hasattr(mod, cls_name):
                instance = getattr(mod, cls_name)()
                break
        else:
            raise RuntimeError(f"No engine class found in {code_path}")

    # 包装为 evaluator 需要的接口
    class Wrapper:
        def __init__(self, eng):
            self._eng = eng
        def process(self, text):
            return self._eng.process(text)

    return Wrapper(instance)


def llm_generate_improvements(score_dict: dict, failures: list, code: str) -> list:
    """LLM 分析评估结果，生成改进方案。"""
    prompt = f"""分析融合引擎的评估结果，生成 3-5 个具体改进方案。

## 当前分数
{json.dumps(score_dict, indent=2)}

## 失败用例
{json.dumps(failures[:12], ensure_ascii=False, indent=2)}

## 需要提升的维度（当前为 0）
- 实体提取 (entity_accuracy): 0.00 — 需要新增 EntityExtractor
- 情感识别 (sentiment_accuracy): 0.00 — 需要新增 SentimentAnalyzer
- 信息密度 (info_density): 0.00 — 需要推理链压缩
- 推理链质量 (chain_quality): 0.00 — 需要 LongCoT
- 融合质量 (fusion_quality): 0.57 — 需要增强

## 要求
输出 JSON 数组，每个元素:
{{"problem": "...", "target_class": "...", "solution": "...", "code_snippet": "具体要添加的Python代码片段"}}

只输出 JSON，不要其他内容。"""

    resp = query(
        system_message="你是 Python 代码优化专家。只输出 JSON。",
        user_message=prompt,
        max_tokens=8000,
    )

    clean = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL)
    m = re.search(r'\[.*\]', clean, re.DOTALL)
    if not m:
        m = re.search(r'\[.*\]', resp, re.DOTALL)
    if m:
        return json.loads(m.group())
    logger.warning("改进方案 JSON 解析失败")
    return []


def llm_apply_improvements(v2_code: str, improvements: list, round_num: int) -> str:
    """LLM 将改进方案应用到代码，生成完整新版本。"""
    prompt = f"""将以下改进方案应用到融合引擎代码，生成完整的 v3 代码。

## 改进方案（第 {round_num} 轮）
{json.dumps(improvements, ensure_ascii=False, indent=2)[:4000]}

## 原始 V2 代码
```python
{v2_code}
```

## 规则
1. 在 V2 基础上修改，保持向后兼容
2. 新增的类/功能必须完整实现（不能写 TODO/pass）
3. 保持 FusionEngineV2 类名不变，新增的类放在前面
4. process() 返回的 dict 新增: entities, sentiment, info_density, chain_quality
5. 添加完整类型注解
6. 代码必须可直接运行

输出完整 Python 代码（不要 markdown 包裹）。"""

    resp = query(
        system_message="输出完整可运行的 Python 代码。不要解释。",
        user_message=prompt,
        max_tokens=32000,
    )

    # 提取代码
    code_match = re.search(r'```python\s*\n(.*?)```', resp, re.DOTALL)
    if code_match:
        return code_match.group(1)
    # 可能没有 markdown 包裹
    if 'import ' in resp and 'def ' in resp:
        return resp
    logger.warning("代码提取失败")
    return ""


def run_evolution(rounds: int = 5):
    """运行进化循环。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    journal = EvolutionJournal()
    v2_path = ROOT / "aris_brain" / "aris_fusion_engine_v2.py"
    v3_path = ROOT / "aris_brain" / "aris_fusion_engine_v3.py"

    # ── 评估 V2 基线 ──
    logger.info("=" * 60)
    logger.info("评估 V2 基线")
    logger.info("=" * 60)

    v2_wrapper = load_engine_instance(v2_path)
    v2_eval = evaluate_fusion_engine(v2_wrapper, version="v2")
    save_eval(v2_eval)

    best_score = v2_eval.score.composite
    best_code = v2_path.read_text(encoding="utf-8")
    current_code = best_code
    current_version = "v2"

    logger.info(f"V2 综合得分: {best_score:.4f}")
    logger.info(f"  实体={v2_eval.score.entity_accuracy:.2f} 情感={v2_eval.score.sentiment_accuracy:.2f}")
    logger.info(f"  融合={v2_eval.score.fusion_quality:.2f} 密度={v2_eval.score.info_density:.2f}")

    # 记录 V2 节点
    v2_node = EvolutionNode(strategy="V2 基线", plan="V2: IFCoT 三路径")
    v2_node.metric = MetricValue(best_score, maximize=True)
    v2_node.is_buggy = False
    journal.append(v2_node)
    parent_node = v2_node

    # ── 进化循环 ──
    for round_num in range(1, rounds + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"进化第 {round_num}/{rounds} 轮")
        logger.info("=" * 60)

        # 1. 生成改进方案
        logger.info("→ LLM 分析失败用例，生成改进方案...")
        failures = [d for d in v2_eval.details if not d.get("passed")]
        improvements = llm_generate_improvements(
            v2_eval.score.to_dict(), failures, current_code
        )
        logger.info(f"  生成 {len(improvements)} 个方案:")
        for i, imp in enumerate(improvements):
            logger.info(f"    {i+1}. [{imp.get('target_class', '?')}] {imp.get('problem', '?')[:50]}")

        if not improvements:
            logger.warning("  无改进方案，跳过")
            continue

        # 2. 应用改进，生成新代码
        logger.info("→ LLM 应用改进方案，生成新代码...")
        new_code = llm_apply_improvements(current_code, improvements, round_num)

        if len(new_code) < 2000:
            logger.warning(f"  生成代码过短 ({len(new_code)} 字符)，跳过")
            continue

        # 保存新代码
        v3_path.write_text(new_code, encoding="utf-8")
        logger.info(f"  新代码: {len(new_code)} 字符 → {v3_path}")

        # 3. 评估新代码
        logger.info("→ 评估新代码...")
        try:
            new_wrapper = load_engine_instance(v3_path)
            new_eval = evaluate_fusion_engine(new_wrapper, version=f"v3-r{round_num}")
            save_eval(new_eval)

            new_score = new_eval.score.composite
            logger.info(f"  新得分: {new_score:.4f} (vs 当前最佳 {best_score:.4f})")
            logger.info(f"    准确={new_eval.score.accuracy:.2f} 实体={new_eval.score.entity_accuracy:.2f}")
            logger.info(f"    情感={new_eval.score.sentiment_accuracy:.2f} 融合={new_eval.score.fusion_quality:.2f}")
            logger.info(f"    密度={new_eval.score.info_density:.2f} 链质量={new_eval.score.chain_quality:.2f}")

            # 4. 记录到进化树
            node = EvolutionNode(
                strategy=new_code[:500],
                plan=f"Round {round_num}: {improvements[0].get('problem', '?')[:40]}",
                parent=parent_node,
            )

            if new_score > best_score:
                logger.info(f"  ✅ 提升! {best_score:.4f} → {new_score:.4f}")
                node.metric = MetricValue(new_score, maximize=True)
                node.is_buggy = False
                best_score = new_score
                best_code = new_code
                current_code = new_code
                current_version = f"v3-r{round_num}"
                parent_node = node
            else:
                logger.info(f"  ❌ 未提升，保持当前最佳")
                node.metric = MetricValue(new_score, maximize=True)
                node.is_buggy = True
                node.analysis = f"得分 {new_score:.4f} 未超过 {best_score:.4f}"

            journal.append(node)

        except Exception as e:
            logger.error(f"  评估失败: {e}")
            node = EvolutionNode(
                strategy=new_code[:500],
                plan=f"Round {round_num} (评估失败)",
                parent=parent_node,
            )
            node.metric = WorstMetricValue()
            node.is_buggy = True
            node.analysis = str(e)
            journal.append(node)

    # ── 最终报告 ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("进化完成 — 最终报告")
    logger.info("=" * 60)
    logger.info(f"总节点: {len(journal)}")
    logger.info(f"好节点: {len(journal.good_nodes)}")
    logger.info(f"Buggy: {len(journal.buggy_nodes)}")
    logger.info(f"当前最佳版本: {current_version}")
    logger.info(f"最佳得分: {best_score:.4f}")

    # 保存最佳代码
    if best_code != v2_path.read_text(encoding="utf-8"):
        v3_path.write_text(best_code, encoding="utf-8")
        logger.info(f"最佳代码已保存: {v3_path}")

    # 保存进化日志
    journal_path = ROOT / ".openclaw" / "tmp" / "evolution_journal.json"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(journal.to_json(), encoding="utf-8")
    logger.info(f"进化日志: {journal_path}")

    # 打印进化树
    logger.info("")
    logger.info("进化树:")
    best_node = journal.get_best_node()
    for n in journal.nodes:
        status = "✓" if not n.is_buggy else "✗"
        score = n.metric.value if n.metric else 0
        mark = " ★" if n is best_node else ""
        logger.info(f"  {status} {n.plan} ({score:.4f}){mark}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    run_evolution(rounds=args.rounds)
