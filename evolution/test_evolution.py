"""
feeling 项目自进化系统 - 测试脚本

测试内容：
1. 树搜索逻辑
2. 节点创建和进化
3. 指标计算
4. 多维指标
5. 收敛检测
6. 序列化/反序列化
"""

import sys
import os
import json
import time

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接导入测试所需的模块，避免触发后端依赖
# run.py 会导入 agent.py（需要后端），所以收敛检测和报告函数单独实现
from evolution.utils.metric import (
    MetricValue,
    WorstMetricValue,
    MultiDimensionalMetric,
    WorstMultiDimensionalMetric,
)
from evolution.runner import ExecutionResult, ConversationMetrics
from evolution.journal import EvolutionNode, EvolutionJournal


def test_metric_value():
    """测试单维指标 MetricValue"""
    print("=" * 50)
    print("测试 MetricValue...")

    # 基本创建和比较
    m1 = MetricValue(0.8, maximize=True)
    m2 = MetricValue(0.9, maximize=True)
    m3 = MetricValue(0.7, maximize=True)

    assert m2 > m1, "m2 应该大于 m1"
    assert m1 > m3, "m1 应该大于 m3"
    assert not (m3 > m2), "m3 不应该大于 m2"

    # 最小化指标
    m4 = MetricValue(0.1, maximize=False)
    m5 = MetricValue(0.2, maximize=False)
    assert m4 > m5, "最小化指标中 m4(0.1) 应该优于 m5(0.2)"

    # 最差指标
    worst = WorstMetricValue()
    assert m1 > worst, "任何有效指标都应该优于最差指标"
    assert worst.is_worst, "最差指标的 is_worst 应该为 True"

    # 排序
    metrics = [m1, m2, m3, worst]
    sorted_metrics = sorted(metrics, reverse=True)
    assert sorted_metrics[0] == m2, "排序后第一个应该是 m2(0.9)"

    print(f"  m1={m1}, m2={m2}, m3={m3}")
    print(f"  最差指标: {worst}")
    print(f"  排序结果: {[str(m) for m in sorted_metrics]}")
    print("✓ MetricValue 测试通过\n")


def test_multi_dimensional_metric():
    """测试多维指标 MultiDimensionalMetric"""
    print("=" * 50)
    print("测试 MultiDimensionalMetric...")

    # 基本创建
    mm1 = MultiDimensionalMetric(
        task_success=0.9,
        user_satisfaction=0.8,
        efficiency=0.7,
        safety=0.95,
        creativity=0.6,
    )

    mm2 = MultiDimensionalMetric(
        task_success=0.8,
        user_satisfaction=0.9,
        efficiency=0.8,
        safety=0.9,
        creativity=0.7,
    )

    print(f"  mm1: {mm1}")
    print(f"  mm2: {mm2}")
    print(f"  mm1 综合分数: {mm1.composite_score:.4f}")
    print(f"  mm2 综合分数: {mm2.composite_score:.4f}")

    # 比较
    assert mm1 != mm2, "两个不同指标应该不相等"
    assert (mm1 > mm2) or (mm2 > mm1), "两个指标应该可以比较"

    # 最差指标
    worst = WorstMultiDimensionalMetric()
    assert mm1 > worst, "有效指标应该优于最差指标"
    assert worst.is_worst, "最差指标的 is_worst 应该为 True"
    print(f"  最差指标: {worst}")

    # 字典摘要
    summary = mm1.to_dict_summary()
    assert "task_success" in summary
    assert "composite" in summary
    print(f"  mm1 字典摘要: {summary}")

    # 边界值测试
    mm3 = MultiDimensionalMetric(
        task_success=1.5,  # 超出范围
        user_satisfaction=-0.1,  # 低于范围
        efficiency=0.5,
        safety=0.5,
        creativity=0.5,
    )
    assert mm3.task_success == 1.0, "超出范围应该被裁剪到 1.0"
    assert mm3.user_satisfaction == 0.0, "低于范围应该被裁剪到 0.0"
    print(f"  边界值裁剪: task_success={mm3.task_success}, user_satisfaction={mm3.user_satisfaction}")

    print("✓ MultiDimensionalMetric 测试通过\n")


def test_evolution_node():
    """测试进化节点 EvolutionNode"""
    print("=" * 50)
    print("测试 EvolutionNode...")

    # 创建根节点
    node1 = EvolutionNode(
        strategy="使用简单的规则引擎处理用户输入",
        plan="初始策略：基于规则的处理方案",
    )
    assert node1.parent is None
    assert node1.stage_name == "draft"
    assert node1.is_leaf
    assert node1.debug_depth == 0
    print(f"  根节点: id={node1.id[:8]}, stage={node1.stage_name}")

    # 创建子节点（改进）
    node2 = EvolutionNode(
        strategy="在规则引擎基础上加入 ML 分类器",
        plan="改进策略：增加 ML 能力",
        parent=node1,
    )
    assert node2.parent is node1
    assert node2 in node1.children
    assert node2.stage_name == "improve"
    assert not node1.is_leaf  # node1 现在不是叶节点了
    print(f"  子节点: id={node2.id[:8]}, stage={node2.stage_name}")

    # 模拟 buggy 状态并创建调试节点
    node2.is_buggy = True
    node2.metric = WorstMetricValue()

    node3 = EvolutionNode(
        strategy="修复 ML 分类器的输入格式问题",
        plan="调试修复：修正输入格式",
        parent=node2,
    )
    assert node3.stage_name == "debug"
    assert node3.debug_depth == 1
    print(f"  调试节点: id={node3.id[:8]}, stage={node3.stage_name}, debug_depth={node3.debug_depth}")

    # 测试 absorb_exec_result
    exec_result = ExecutionResult(
        term_out=["执行成功", "输出结果：准确率 0.85"],
        exec_time=2.5,
        exc_type=None,
    )
    node3.absorb_exec_result(exec_result)
    assert node3.exec_time == 2.5
    assert node3.exc_type is None
    print(f"  执行结果: exec_time={node3.exec_time}, exc_type={node3.exc_type}")

    # 设置指标
    node3.metric = MetricValue(0.85, maximize=True)
    node3.is_buggy = False
    node3.analysis = "策略执行成功，准确率达到 0.85"

    print("✓ EvolutionNode 测试通过\n")
    return node1, node2, node3


def test_evolution_journal():
    """测试进化日志 EvolutionJournal"""
    print("=" * 50)
    print("测试 EvolutionJournal...")

    journal = EvolutionJournal()

    # 创建多个节点
    nodes = []
    for i in range(5):
        node = EvolutionNode(
            strategy=f"策略 {i}: 测试策略内容 {i}",
            plan=f"计划 {i}: 测试计划 {i}",
        )
        node.metric = MetricValue(0.5 + i * 0.1, maximize=True)
        node.is_buggy = (i == 2)  # 第 3 个节点有 bug
        node.analysis = f"分析 {i}: 策略执行结果描述"
        if node.is_buggy:
            node.metric = WorstMetricValue()
        journal.append(node)
        nodes.append(node)

    # 基本属性
    assert len(journal) == 5
    assert journal[0] is nodes[0]
    print(f"  节点数: {len(journal)}")

    # 草稿节点（所有节点都没有 parent）
    assert len(journal.draft_nodes) == 5
    print(f"  草稿节点数: {len(journal.draft_nodes)}")

    # buggy 节点
    assert len(journal.buggy_nodes) == 1
    print(f"  Buggy 节点数: {len(journal.buggy_nodes)}")

    # 好节点
    assert len(journal.good_nodes) == 4
    print(f"  好节点数: {len(journal.good_nodes)}")

    # 最佳节点
    best = journal.get_best_node()
    assert best is nodes[4]  # 指标最高的节点
    print(f"  最佳节点: metric={best.metric}")

    # 摘要
    summary = journal.generate_summary()
    assert "设计" in summary or "验证指标" in summary
    print(f"  摘要长度: {len(summary)} 字符")

    # 统计
    stats = journal.get_evolution_stats()
    assert stats["total_nodes"] == 5
    assert stats["good_nodes"] == 4
    assert stats["buggy_nodes"] == 1
    print(f"  统计: {stats}")

    # 迭代器
    count = 0
    for node in journal:
        count += 1
    assert count == 5
    print(f"  迭代器测试: 遍历了 {count} 个节点")

    print("✓ EvolutionJournal 测试通过\n")
    return journal


def test_tree_structure():
    """测试树结构"""
    print("=" * 50)
    print("测试树结构...")

    # 构建一个小型进化树
    root = EvolutionNode(strategy="根策略", plan="根计划")
    root.metric = MetricValue(0.6, maximize=True)
    root.is_buggy = False

    child1 = EvolutionNode(strategy="改进策略1", plan="改进1", parent=root)
    child1.metric = MetricValue(0.7, maximize=True)
    child1.is_buggy = False

    child2 = EvolutionNode(strategy="改进策略2", plan="改进2", parent=root)
    child2.metric = MetricValue(0.8, maximize=True)
    child2.is_buggy = False

    # child1 的子节点（调试路径）
    child1.is_buggy = True
    child1.metric = WorstMetricValue()

    debug1 = EvolutionNode(strategy="调试策略1", plan="调试1", parent=child1)
    debug1.metric = MetricValue(0.65, maximize=True)
    debug1.is_buggy = False

    # child2 的子节点
    child3 = EvolutionNode(strategy="深度改进", plan="深度改进", parent=child2)
    child3.metric = MetricValue(0.85, maximize=True)
    child3.is_buggy = False

    # 验证树结构
    assert root.is_leaf is False
    assert len(root.children) == 2
    assert child1 in root.children
    assert child2 in root.children
    assert debug1.parent is child1
    assert debug1.stage_name == "debug"
    assert debug1.debug_depth == 1
    assert child3.stage_name == "improve"

    print(f"  根节点子节点数: {len(root.children)}")
    print(f"  child1 子节点数: {len(child1.children)}")
    print(f"  child2 子节点数: {len(child2.children)}")
    print(f"  debug1 调试深度: {debug1.debug_depth}")

    # 放入日志测试最佳节点
    journal = EvolutionJournal()
    for node in [root, child1, child2, debug1, child3]:
        journal.append(node)

    best = journal.get_best_node()
    assert best is child3  # 指标 0.85
    print(f"  最佳节点指标: {best.metric}")

    # 测试 get_best_node(only_good=False)
    best_all = journal.get_best_node(only_good=False)
    assert best_all is child3
    print(f"  全局最佳节点指标: {best_all.metric}")

    print("✓ 树结构测试通过\n")


def _local_check_convergence(
    journal: EvolutionJournal,
    window_size: int = 5,
    improvement_threshold: float = 0.01,
) -> bool:
    """本地收敛检测（避免导入 run.py 触发后端依赖）"""
    good_nodes = journal.good_nodes
    if len(good_nodes) < window_size:
        return False
    recent_metrics = [n.metric.value for n in good_nodes[-window_size:]]
    valid_metrics = [m for m in recent_metrics if m is not None]
    if len(valid_metrics) < 2:
        return False
    min_metric = min(valid_metrics)
    max_metric = max(valid_metrics)
    if abs(max_metric) > 1e-10:
        improvement = (max_metric - min_metric) / abs(max_metric)
    else:
        improvement = max_metric - min_metric
    return improvement < improvement_threshold


def _local_format_report(journal: EvolutionJournal) -> str:
    """本地报告生成（避免导入 run.py 触发后端依赖）"""
    stats = journal.get_evolution_stats()
    best_node = journal.get_best_node()
    lines = [
        "=" * 60,
        "进化报告",
        "=" * 60,
        f"总节点数: {stats['total_nodes']}",
        f"好节点数: {stats['good_nodes']}",
        f"Buggy 节点数: {stats['buggy_nodes']}",
    ]
    if best_node:
        lines.append(f"最佳策略指标: {best_node.metric.value:.4f}")
        if best_node.multi_metric:
            lines.append(f"多维评估: {best_node.multi_metric}")
    lines.append("=" * 60)
    return "\n".join(lines)


def test_convergence_detection():
    """测试收敛检测"""
    print("=" * 50)
    print("测试收敛检测...")

    journal = EvolutionJournal()

    # 添加指标相近的好节点（模拟收敛）
    for i in range(7):
        node = EvolutionNode(
            strategy=f"策略 {i}",
            plan=f"计划 {i}",
        )
        node.metric = MetricValue(0.85 + i * 0.001, maximize=True)
        node.is_buggy = False
        journal.append(node)

    # 应该检测到收敛
    converged = _local_check_convergence(journal, window_size=5, improvement_threshold=0.01)
    assert converged, "应该检测到收敛"
    print(f"  收敛检测结果: {converged}")

    # 添加一个有显著改进的节点
    node_new = EvolutionNode(
        strategy="重大突破策略",
        plan="重大突破",
    )
    node_new.metric = MetricValue(0.95, maximize=True)
    node_new.is_buggy = False
    journal.append(node_new)

    # 不应该检测到收敛
    converged2 = _local_check_convergence(journal, window_size=5, improvement_threshold=0.01)
    assert not converged2, "不应该检测到收敛"
    print(f"  收敛检测结果（有突破后）: {converged2}")

    # 测试节点不足的情况
    small_journal = EvolutionJournal()
    node_small = EvolutionNode(strategy="s", plan="p")
    node_small.metric = MetricValue(0.5, maximize=True)
    node_small.is_buggy = False
    small_journal.append(node_small)

    converged3 = _local_check_convergence(small_journal, window_size=5)
    assert not converged3, "节点不足时不应该收敛"
    print(f"  收敛检测结果（节点不足）: {converged3}")

    print("✓ 收敛检测测试通过\n")


def test_evolution_report():
    """测试进化报告"""
    print("=" * 50)
    print("测试进化报告...")

    journal = EvolutionJournal()

    # 创建几个节点
    for i in range(3):
        node = EvolutionNode(
            strategy=f"策略 {i}: " + "x" * 100,
            plan=f"计划 {i}",
        )
        node.metric = MetricValue(0.7 + i * 0.1, maximize=True)
        node.is_buggy = False
        node.analysis = f"分析 {i}: 策略执行成功"
        node.multi_metric = MultiDimensionalMetric(
            task_success=0.8 + i * 0.05,
            user_satisfaction=0.7 + i * 0.1,
            efficiency=0.6 + i * 0.1,
            safety=0.9,
            creativity=0.5 + i * 0.1,
        )
        journal.append(node)

    report = _local_format_report(journal)
    print(report)
    assert "进化报告" in report
    assert "最佳策略指标" in report
    assert "多维评估" in report

    print("✓ 进化报告测试通过\n")


def test_serialization():
    """测试序列化/反序列化"""
    print("=" * 50)
    print("测试序列化...")

    from evolution.utils.serialize import dumps_json, loads_json

    journal = EvolutionJournal()
    for i in range(3):
        node = EvolutionNode(
            strategy=f"策略 {i}",
            plan=f"计划 {i}",
        )
        node.metric = MetricValue(0.7 + i * 0.1, maximize=True)
        node.is_buggy = False
        journal.append(node)

    # 创建树结构
    node_child = EvolutionNode(
        strategy="子策略",
        plan="子计划",
        parent=journal[0],
    )
    node_child.metric = MetricValue(0.85, maximize=True)
    node_child.is_buggy = False
    journal.append(node_child)

    # 序列化
    json_str = dumps_json(journal)
    print(f"  序列化大小: {len(json_str)} 字节")

    # 反序列化
    loaded_journal = loads_json(json_str, EvolutionJournal)
    assert len(loaded_journal) == len(journal)
    assert loaded_journal[0].strategy == journal[0].strategy

    # 验证树结构恢复
    assert loaded_journal[3].parent is loaded_journal[0]
    assert loaded_journal[3] in loaded_journal[0].children
    print(f"  反序列化节点数: {len(loaded_journal)}")
    print(f"  树结构恢复: parent={loaded_journal[3].parent.id[:8]}, children_count={len(loaded_journal[0].children)}")

    print("✓ 序列化测试通过\n")


def test_conversation_metrics():
    """测试对话指标"""
    print("=" * 50)
    print("测试 ConversationMetrics...")

    metrics = ConversationMetrics(
        success=True,
        response_time=2.5,
        token_count=150,
        relevance_score=0.9,
        coherence_score=0.85,
        helpfulness_score=0.8,
    )

    assert metrics.success is True
    assert metrics.response_time == 2.5
    assert metrics.relevance_score == 0.9

    print(f"  成功: {metrics.success}")
    print(f"  响应时间: {metrics.response_time}s")
    print(f"  相关性: {metrics.relevance_score}")
    print(f"  连贯性: {metrics.coherence_score}")
    print(f"  有用性: {metrics.helpfulness_score}")

    # 失败情况
    failed_metrics = ConversationMetrics(
        success=False,
        response_time=0.1,
        error_message="连接超时",
        error_type="TimeoutError",
    )
    assert failed_metrics.success is False
    assert failed_metrics.error_type == "TimeoutError"
    print(f"  失败指标: error_type={failed_metrics.error_type}")

    print("✓ ConversationMetrics 测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("feeling 项目自进化系统 - 测试套件")
    print("=" * 60 + "\n")

    tests = [
        test_metric_value,
        test_multi_dimensional_metric,
        test_evolution_node,
        test_evolution_journal,
        test_tree_structure,
        test_convergence_detection,
        test_evolution_report,
        test_serialization,
        test_conversation_metrics,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test_fn.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败, 共 {passed + failed} 个")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
