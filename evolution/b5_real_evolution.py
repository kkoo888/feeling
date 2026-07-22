# -*- coding: utf-8 -*-
"""
B5 真正进化 — 使用 CodeEvolutionAgent.step() 完整进化流程
=========================================================
核心区别 (vs 之前的脚本):
  1. 使用 step() 而非直接调 _improve() — search_policy() 自动决定 draft/improve/debug/explore
  2. 持久化 EvolutionJournal 跨 10 轮 — generate_summary() 有完整记忆
  3. 走 parse_exec_result() — LLM 评估执行结果, 完整进化闭环
  4. exec_callback 跑 B5 fitness — 真实评估指标

流程 (每轮):
  step() → search_policy() → _improve/_draft/_debug/_explore
        → query() [写 pending → Trae 回 response]  ← LLM 生成代码
        → exec_callback(strategy)  ← B5 fitness 评估
        → parse_exec_result() → query() [写 pending → Trae 回 response]  ← LLM 评估结果
        → journal.append(node)  ← 累积记忆

用法: python b5_real_evolution.py
  环境变量 EVOLUTION_BACKEND=trae 自动设置
  进程阻塞在 query() 的 pending 文件上, 等待 Trae 写 response
"""
import sys
import os
import json
import time
import tempfile
import traceback

# ── 路径设置 ──
ROOT = r"c:\Users\CC\Desktop\feeling"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "laap", "agi"))
sys.path.insert(0, os.path.join(ROOT, "aris_brain"))

# 启用 Trae 后端
os.environ["EVOLUTION_BACKEND"] = "trae"
os.environ["TRAE_LLM_TIMEOUT"] = "600"  # 10分钟超时, 给子代理足够时间

# 清理旧的 bridge 文件
BRIDGE_DIR = os.path.join(ROOT, ".trae_llm_bridge")
os.makedirs(BRIDGE_DIR, exist_ok=True)
for f in os.listdir(BRIDGE_DIR):
    if f.endswith(".json"):
        os.remove(os.path.join(BRIDGE_DIR, f))

B5_PATH = os.path.join(ROOT, "aris_brain", "causal_feature_extractor.py")
STATE_DIR = os.path.join(ROOT, "evolution", "b5_trae_state")
os.makedirs(STATE_DIR, exist_ok=True)

NUM_ROUNDS = 10


def create_exec_callback(round_num):
    """创建 B5 fitness 评估回调, 供 step() 使用。

    step() 会调用 exec_callback(strategy, True) → 返回 ExecutionResult
    """
    from evolution.runner import ExecutionResult

    def exec_callback(strategy: str, reset: bool) -> ExecutionResult:
        """评估 B5 代码变体的 fitness"""
        output = []
        exc_type = None

        try:
            # 语法检查
            compile(strategy, f"<b5_round_{round_num}>", "exec")

            # 写到临时文件
            tmp_path = os.path.join(tempfile.gettempdir(), f"b5_eval_r{round_num}.py")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(strategy)

            # 运行 fitness
            from b5_fitness import run_fitness
            fit = run_fitness(tmp_path)

            score = fit.get("score", 0.0)
            details = fit.get("details", "")
            edges = fit.get("edges", 0)
            paths = fit.get("treatment_paths", 0)

            output.append(f"=== B5 Fitness Evaluation (Round {round_num}) ===\n")
            output.append(f"Score: {score:.3f}\n")
            output.append(f"Edges: {edges}\n")
            output.append(f"Paths: {paths}\n")
            output.append(f"Details: {details}\n")
            output.append(f"Var Count: {fit.get('var_count', 0)}\n")
            output.append(f"NonZero Ratio: {fit.get('non_zero_var_ratio', 0):.2%}\n")

            if "error" in fit:
                output.append(f"WARNING: {fit['error']}")
                exc_type = "FitnessError"

        except SyntaxError as e:
            exc_type = "SyntaxError"
            output.append(f"SyntaxError: {e}")
            output.append(f"File: {e.filename}, Line: {e.lineno}")
            output.append(f"Text: {e.text}")
        except Exception as e:
            exc_type = type(e).__name__
            output.append(f"Error: {type(e).__name__}: {e}")
            output.append(traceback.format_exc())

        return ExecutionResult(
            term_out=output,
            exec_time=0.0,
            exc_type=exc_type,
        )

    return exec_callback


def run_evolution():
    """运行 10 轮真正的进化"""
    from evolution.journal import EvolutionJournal
    from evolution.code_evolution import CodeEvolutionAgent

    # ── 创建 ONE 持久 journal (跨所有 10 轮) ──
    journal = EvolutionJournal()

    # ── 创建 CodeEvolutionAgent ──
    task_desc = (
        "B5 causal_feature_extractor: 优化因果变量提取器, "
        "提升 discover() 发现的边数、treatment→outcome 路径数、变量覆盖度。"
        "约束: 纯 Python stdlib, 保持 extract_causal_observation 和 "
        "get_variable_roles 两个函数的接口不变。"
        "当前基线: score=5.393, edges=7, paths=1, zero_var=3。"
        "目标: score>=8.0, edges>=20, paths>=4, zero_var=0。"
    )

    agent = CodeEvolutionAgent(
        task_desc=task_desc,
        journal=journal,
        model="trae",
        temperature=0.7,
        num_drafts=1,       # 第一轮走 draft, 之后走 improve
        debug_prob=0.2,    # 20% 概率修复 bug
        explore_prob=0.15,  # 15% 概率探索新方向
        eval_metric="B5_fitness",
        eval_direction="maximize",
    )

    # 读取当前 B5 代码作为初始种子
    with open(B5_PATH, "r", encoding="utf-8") as f:
        initial_code = f.read()

    # 先手动注入基线节点 (让 improve 有 parent)
    from evolution.journal import EvolutionNode
    from evolution.utils.metric import MetricValue

    baseline_node = EvolutionNode(
        strategy=initial_code,
        plan="基线代码 — 原始 B5 causal_feature_extractor",
    )
    # 评估基线
    exec_cb = create_exec_callback(-1)
    baseline_result = exec_cb(initial_code, True)
    baseline_node.absorb_exec_result(baseline_result)

    # 手动设置基线 metric (不走 parse_exec_result 避免额外 LLM 调用)
    from b5_fitness import run_fitness
    import tempfile as _tf
    _tmp = os.path.join(_tf.gettempdir(), "b5_baseline.py")
    with open(_tmp, "w", encoding="utf-8") as f:
        f.write(initial_code)
    baseline_fit = run_fitness(_tmp)
    baseline_score = baseline_fit.get("score", 5.393)
    baseline_node.metric = MetricValue(baseline_score, maximize=True)
    baseline_node.is_buggy = False
    baseline_node.analysis = f"基线: score={baseline_score:.3f} {baseline_fit.get('details','')}"

    journal.append(baseline_node)
    print(f"[初始化] 基线节点已注入: score={baseline_score:.3f}")
    print(f"  {baseline_fit.get('details', '')}")

    # ── 10 轮进化 ──
    for round_num in range(NUM_ROUNDS):
        print(f"\n{'='*55}")
        print(f"  Round {round_num} 开始")
        print(f"  Journal 节点数: {len(journal)}")
        best = journal.get_best_node()
        if best and best.metric:
            print(f"  当前最优: score={best.metric.value:.3f}")
        print(f"{'='*55}")

        # 创建本轮的 exec_callback
        exec_callback = create_exec_callback(round_num)

        # 执行 step() — 这会:
        # 1. search_policy() 决定 draft/improve/debug/explore
        # 2. 调用对应方法 → query() → 写 pending 文件 (阻塞等 Trae response)
        # 3. exec_callback(strategy) → B5 fitness 评估
        # 4. parse_exec_result() → query() → 写 pending 文件 (阻塞等 Trae response)
        # 5. journal.append(node)
        try:
            agent.step(exec_callback)
        except TimeoutError as e:
            print(f"[Round {round_num}] 超时: {e}")
            continue  # 不中断, 继续下一轮
        except Exception as e:
            print(f"[Round {round_num}] 错误: {type(e).__name__}: {e}")
            traceback.print_exc()
            # 即使出错也继续下一轮
            continue

        # 检查本轮结果并保存 — 包在 try/except 里防止保存错误中断进化
        try:
            node = journal[-1]  # 最后追加的节点
            score = node.metric.value if node.metric and node.metric.value is not None else 0.0
            stage = node.stage_name
            is_buggy = node.is_buggy

            print(f"\n[Round {round_num}] 完成:")
            print(f"  阶段: {stage}")
            print(f"  得分: {score:.3f}")
            print(f"  有Bug: {is_buggy}")
            print(f"  分析: {node.analysis[:200] if node.analysis else 'N/A'}")

            # 保存 champion
            best = journal.get_best_node()
            if best and best.strategy:
                champ_path = os.path.join(STATE_DIR, f"champion_r{round_num}.py")
                with open(champ_path, "w", encoding="utf-8") as f:
                    f.write(best.strategy)

            # 保存本轮结果
            result_path = os.path.join(STATE_DIR, f"result_r{round_num}.json")
            result_data = {
                "round": round_num,
                "stage": stage,
                "score": score,
                "is_buggy": is_buggy,
                "analysis": node.analysis,
                "best_score": best.metric.value if best and best.metric else 0,
                "journal_size": len(journal),
            }

            # 用正则提取 fitness 细节
            if not is_buggy and node.term_out:
                import re
                m_score = re.search(r"Score:\s*([\d.]+)", node.term_out)
                if m_score:
                    result_data["fitness_score"] = float(m_score.group(1))
                m_edges = re.search(r"Edges:\s*(\d+)", node.term_out)
                if m_edges:
                    result_data["edges"] = int(m_edges.group(1))
                m_paths = re.search(r"Paths:\s*(\d+)", node.term_out)
                if m_paths:
                    result_data["paths"] = int(m_paths.group(1))

            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            # 保存 journal 摘要
            journal_path = os.path.join(STATE_DIR, f"journal_r{round_num}.json")
            journal_summary = {
                "total_nodes": len(journal),
                "good_nodes": len(journal.good_nodes),
                "buggy_nodes": len(journal.buggy_nodes),
                "best_score": best.metric.value if best and best.metric else 0,
                "stages": {
                    "draft": len([n for n in journal if n.stage_name == "draft"]),
                    "improve": len([n for n in journal if n.stage_name == "improve"]),
                    "debug": len([n for n in journal if n.stage_name == "debug"]),
                },
                "nodes": [
                    {
                        "step": n.step,
                        "stage": n.stage_name,
                        "score": n.metric.value if n.metric else None,
                        "is_buggy": n.is_buggy,
                        "plan": n.plan[:100] if n.plan else None,
                        "analysis": n.analysis[:200] if n.analysis else None,
                    }
                for n in journal
            ],
            }
            with open(journal_path, "w", encoding="utf-8") as f:
                json.dump(journal_summary, f, ensure_ascii=False, indent=2)

            print(f"  Champion 已保存: champion_r{round_num}.py")
            print(f"  结果已保存: result_r{round_num}.json")
        except Exception as e:
            print(f"[Round {round_num}] 保存结果时出错 (进化继续): {type(e).__name__}: {e}")
            traceback.print_exc()

    # ── 最终汇总 ──
    print(f"\n{'='*55}")
    print(f"  进化完成 — {NUM_ROUNDS} 轮")
    print(f"{'='*55}")

    best = journal.get_best_node()
    if best:
        print(f"  最优得分: {best.metric.value:.3f}")
        print(f"  基线得分: {baseline_score:.3f}")
        print(f"  提升: +{(best.metric.value - baseline_score):.3f} ({((best.metric.value - baseline_score) / baseline_score * 100):.1f}%)")

        # 保存最终 champion
        final_path = os.path.join(STATE_DIR, "champion_final.py")
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(best.strategy)

        # 最终汇总
        summary = {
            "baseline_score": baseline_score,
            "best_score": best.metric.value,
            "improvement": best.metric.value - baseline_score,
            "improvement_pct": ((best.metric.value - baseline_score) / baseline_score * 100),
            "total_rounds": NUM_ROUNDS,
            "total_nodes": len(journal),
            "good_nodes": len(journal.good_nodes),
            "buggy_nodes": len(journal.buggy_nodes),
        }
        summary_path = os.path.join(STATE_DIR, "evolution_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"  最终 champion: champion_final.py")
        print(f"  汇总: evolution_summary.json")

    # 所有轮次的得分曲线
    scores = []
    for n in journal:
        s = n.metric.value if n.metric else 0.0
        scores.append({"round": n.step, "stage": n.stage_name, "score": s, "buggy": n.is_buggy})

    curve_path = os.path.join(STATE_DIR, "score_curve.json")
    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print(f"\n  得分曲线:")
    for s in scores:
        marker = "BUG" if s["buggy"] else "OK"
        print(f"    Round {s['round']:2d} [{s['stage']:7s}] score={s['score']:.3f} [{marker}]")


if __name__ == "__main__":
    print("=" * 55)
    print("  B5 真正进化系统 — CodeEvolutionAgent.step()")
    print("  10 轮完整进化 (search_policy → improve/debug/explore)")
    print("=" * 55)

    run_evolution()
