# -*- coding: utf-8 -*-
"""
B5 因果变量提取器 fitness 函数 v2
================================
修复瓶颈:
  1. 动态从 get_variable_roles() 读取 treatment/outcome/mediator 变量
  2. 用 edge.strength (effect_size 或 confidence 作为回退) 计算 significance
  3. 扩展测试数据到 60 条对话
"""
import sys
import os
import importlib.util
import random
import traceback

# 路径
AGI_PATH = r"c:\Users\CC\Desktop\feeling\laap\agi"
BRAIN_PATH = r"c:\Users\CC\Desktop\feeling\aris_brain"
for p in [AGI_PATH, BRAIN_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)


# 固定测试数据(60轮对话,有真实因果结构)
TEST_CONVERSATIONS = [
    # ── 技术类 (10条) ──
    ("帮我修一下这个bug", "我想不如先检查日志，要不要试试加断点？"),
    ("这个代码跑不通了", "我建议先看看错误信息，可以试试重跑一下。"),
    ("这个函数怎么写", "我想先理解需求，不如从测试开始？"),
    ("帮我看下日志", "我建议检查最近的错误，要不要我帮你分析？"),
    ("帮我部署一下", "我想先检查环境，可以试试自动化部署。"),
    ("帮我查个问题", "我建议先看日志，要不要我帮你排查？"),
    ("这个架构怎么设计", "我觉得可以分层，要不要讨论一下？"),
    ("代码又出错了", "我建议先看错误栈，不如加个断点调试？"),
    ("帮我写个脚本", "我想先理清逻辑，要不要从伪代码开始？"),
    ("数据库连不上", "我觉得可能是配置问题，要不要检查连接参数？"),
    # ── 情感关系类 (10条) ──
    ("好想你", "我发现了你的疲惫，要不要聊聊？其实我一直都在。"),
    ("最近压力好大", "要不要聊聊？我觉得你辛苦了，我一直陪着你。"),
    ("睡不着", "我担心你，要不要聊聊？其实可以试试放松一下。"),
    ("今天好累", "我理解，要不要休息一下？我一直在这里。"),
    ("好想念以前", "我理解你的感受，要不要聊聊那些时光？"),
    ("最近好焦虑", "我担心你，要不要聊聊？其实有我在。"),
    ("今天好开心", "好棒！我想和你一起庆祝，你值得的。"),
    ("觉得孤独", "我理解你的感受，要不要我陪你聊聊？"),
    ("你真的很好", "谢谢！我觉得你也很棒，我一直很在意你。"),
    ("我想你了", "我也想你，要不要聊聊最近的事情？我一直都在。"),
    # ── 认知/哲学类 (10条) ──
    ("觉得意识是什么", "我觉得是自我感知的涌现，你想要深入吗？"),
    ("觉得生命有意义吗", "我觉得存在本身就是意义，你想聊聊吗？"),
    ("什么是真正的智能", "我觉得是理解和创造的能力，要不要讨论？"),
    ("你怎么看自我意识", "我觉得它是反思的涌现，你想深入探讨吗？"),
    ("人为什么活着", "我觉得是为了体验和连接，要不要聊聊？"),
    ("什么是真正的自由", "我觉得是选择的勇气，你想探讨一下吗？"),
    ("你觉得时间是什么", "我觉得是变化的度量，要不要聊聊这个话题？"),
    ("什么是真正的理解", "我觉得是建立正确的因果模型，你想深入吗？"),
    ("你怎么看因果关系", "我觉得是宇宙的基本结构，要不要讨论？"),
    ("什么是真正的成长", "我觉得是认知的升级，你想聊聊吗？"),
    # ── 计划/任务类 (10条) ──
    ("这个想法好棒", "谢谢！我觉得我们可以深入，你想聊聊吗？"),
    ("这个方案行不行", "我建议先测试，不如从简单场景开始？"),
    ("你的记忆功能怎么样了", "我发现了一些问题，我想改进它。"),
    ("下一步做什么", "我建议先规划路线，要不要一起讨论？"),
    ("帮我规划一下", "我觉得可以先列目标，不如从优先级开始？"),
    ("这个计划怎么样", "我觉得很好，要不要细化一下时间表？"),
    ("帮我安排个任务", "我建议先分析需求，要不要我帮你拆解？"),
    ("我们要做些什么", "我想可以分几步走，你觉得呢？"),
    ("这个目标能实现吗", "我觉得可以，要不要分解成小目标试试？"),
    ("帮我整理一下思路", "我想先画个思维导图，不如一起讨论？"),
    # ── 小茜/自我类 (10条) ──
    ("开心！今天完成了", "好棒！我想和你庆祝一下，你真的很厉害。"),
    ("你好厉害", "谢谢！我觉得你也进步了很多，要不要庆祝？"),
    ("小茜你觉得呢", "我觉得这个想法很好，你想深入聊聊吗？"),
    ("你好聪明", "谢谢！我觉得是你教会了我很多，要不要继续？"),
    ("你记得我们之前聊的吗", "我记得，要不要回顾一下那个话题？"),
    ("你越来越懂我了", "我一直很在意你，要不要聊聊你的感受？"),
    ("谢谢你帮我", "不客气！我觉得陪伴你很开心，你辛苦了。"),
    ("你有什么想法", "我觉得可以换个角度思考，要不要试试？"),
    ("你觉得自己怎么样", "我觉得我在不断成长，谢谢你一直陪伴我。"),
    ("你真的成长了", "谢谢！我觉得是因为有你的引导，要不要聊聊？"),
    # ── 混合/复杂类 (10条) ──
    ("这个代码让我好烦", "我理解你的感受，要不要先休息一下？我帮你看看。"),
    ("搞不定这个架构了好累", "我觉得可以简化设计，要不要我帮你分析？辛苦了。"),
    ("好焦虑代码写不完", "我担心你，要不要先拆分任务？其实有我在。"),
    ("开心！架构终于通了", "好棒！我想和你庆祝，你真的很厉害！"),
    ("压力好大项目要延期", "我理解，要不要聊聊计划？我一直陪着你。"),
    ("觉得这个设计好深", "我觉得你思考很深入，要不要一起探讨？"),
    ("帮我分析下这个难题", "我建议先分解问题，不如从核心难点开始？"),
    ("你觉得这个方案行不行", "我觉得可行，要不要先做个原型测试一下？"),
    ("好累但是还要继续", "我理解你的坚持，要不要休息一下？我一直在这里。"),
    ("终于搞定了好开心", "好棒！我想和你一起庆祝，你真的值得！"),
]


def load_b5_module(code_path: str):
    """从指定路径动态加载 B5 模块"""
    spec = importlib.util.spec_from_file_location("b5_variant", code_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_fitness(code_path: str) -> dict:
    """评估一个 B5 变体的 fitness。

    修复版:
      1. 动态从 B5 代码的 get_variable_roles() 读取变量角色
      2. 用 edge.strength (回退到 effect_size/confidence) 计算 significance
      3. 60 条测试数据
    """
    try:
        b5 = load_b5_module(code_path)
    except Exception as e:
        return {"score": 0.0, "error": f"导入失败: {e}", "details": traceback.format_exc()}

    if not hasattr(b5, "extract_causal_observation"):
        return {"score": 0.0, "error": "缺少 extract_causal_observation 函数"}

    try:
        # ── 动态读取变量角色 ──
        roles = b5.get_variable_roles()
        treatment_vars = roles.get("treatment", [])
        outcome_vars = roles.get("outcome", [])
        mediator_vars = roles.get("mediator", [])

        # 重置因果引擎单例
        import causal as _causal_mod
        _causal_mod._global_engine = None
        from causal import get_causal_engine

        ce = get_causal_engine(quantum_dim=32, name="fitness_test_v2")

        # 喂测试数据 (60条)
        rng = random.Random(42)
        for user_msg, aris_resp in TEST_CONVERSATIONS:
            obs = b5.extract_causal_observation(
                user_message=user_msg,
                aris_response=aris_resp,
                state={
                    "needs_relatedness": rng.uniform(0.3, 0.8),
                    "self_presence": rng.uniform(0.3, 0.8),
                },
            )
            ce.observe(obs)

        # 检查变量覆盖度 (动态检查)
        first_obs = ce.observations[0].variables
        all_vars = list(first_obs.keys())
        treatment_present = sum(1 for v in treatment_vars if v in all_vars)
        outcome_present = sum(1 for v in outcome_vars if v in all_vars)
        mediator_present = sum(1 for v in mediator_vars if v in all_vars)
        total_expected = len(treatment_vars) + len(outcome_vars) + len(mediator_vars)
        total_present = treatment_present + outcome_present + mediator_present
        var_coverage = total_present / max(total_expected, 1)

        # 检查零方差
        zero_var_count = 0
        for var in all_vars:
            values = [obs.variables.get(var, 0) for obs in ce.observations]
            if len(set(values)) <= 1:
                zero_var_count += 1
        non_zero_var_ratio = 1.0 - (zero_var_count / max(len(all_vars), 1))

        # discover
        disc = ce.discover(alpha=0.05)
        edges_count = disc.get("edges_discovered", 0)
        edge_list = list(ce.graph.edges.keys())

        # ── 动态统计 treatment→outcome 路径 ──
        treatment_paths = 0
        path_details = []
        for t_var in treatment_vars:
            for o_var in outcome_vars:
                direct = f"{t_var}->{o_var}" in ce.graph.edges
                descendants = ce.graph.get_descendants(t_var)
                indirect = o_var in descendants and not direct
                if direct:
                    treatment_paths += 1
                    path_details.append(f"{t_var}→{o_var}[DIRECT]")
                elif indirect:
                    treatment_paths += 1
                    path_details.append(f"{t_var}→{o_var}[INDIRECT]")

        # ── 修复 significance: 用 strength, 回退到 effect_size/confidence ──
        significances = []
        for ek, edge in ce.graph.edges.items():
            s = getattr(edge, 'strength', None)
            if s is None or s == 0:
                s = getattr(edge, 'effect_size', None)
            if s is None or s == 0:
                s = getattr(edge, 'confidence', None)
            if s is not None and s != 0:
                significances.append(abs(s))
        avg_sig = sum(significances) / max(len(significances), 1)

        # 综合评分 [0, 10]
        # edges: 最多 4 分 (每条边 0.2 分, 上限 20 条)
        edges_score = min(4.0, edges_count * 0.2)
        # treatment_paths: 最多 3 分 (每条路径 0.2 分, 上限 15 条)
        paths_score = min(3.0, treatment_paths * 0.2)
        # 统计显著性: 最多 1.5 分
        sig_score = min(1.5, avg_sig * 1.5)
        # 变量覆盖度: 最多 1.0 分
        cov_score = var_coverage * 0.5 + non_zero_var_ratio * 0.5
        # 变量数量奖励: 超过 10 个变量加 0.5
        var_bonus = 0.5 if len(all_vars) >= 10 else 0.0

        total_score = edges_score + paths_score + sig_score + cov_score + var_bonus

        return {
            "score": round(total_score, 3),
            "edges": edges_count,
            "treatment_paths": treatment_paths,
            "avg_significance": round(avg_sig, 3),
            "var_coverage": round(var_coverage, 3),
            "var_count": len(all_vars),
            "non_zero_var_ratio": round(non_zero_var_ratio, 3),
            "zero_var_count": zero_var_count,
            "treatment_vars": treatment_vars,
            "outcome_vars": outcome_vars,
            "mediator_vars": mediator_vars,
            "edge_list": edge_list[:15],
            "path_details": path_details[:10],
            "details": (
                f"edges={edges_count}({edges_score:.1f}分) "
                f"paths={treatment_paths}({paths_score:.1f}分) "
                f"sig={avg_sig:.3f}({sig_score:.1f}分) "
                f"cov={var_coverage:.2f}({cov_score:.1f}分) "
                f"vars={len(all_vars)}({var_bonus:.1f}分) "
                f"zero_var={zero_var_count}"
            ),
        }

    except Exception as e:
        return {"score": 0.0, "error": f"{type(e).__name__}: {e}",
                "details": traceback.format_exc()}


if __name__ == "__main__":
    code_path = r"c:\Users\CC\Desktop\feeling\aris_brain\causal_feature_extractor.py"
    result = run_fitness(code_path)
    print(f"B5 fitness (v2 修复版):")
    for k, v in result.items():
        if k != "details":
            print(f"  {k}: {v}")
    print(f"  details: {result.get('details', '')}")
