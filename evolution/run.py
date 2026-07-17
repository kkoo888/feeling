"""
进化运行器：feeling 项目自进化系统的主循环

从 AIDE 的 run.py 改造而来：
- 保持主循环不变
- 改为调用新的 AgentRunner
- 加入进化收敛检测
"""

import logging
import math

from .agent import EvolutionAgent
from .runner import AgentRunner
from .journal import EvolutionJournal, EvolutionNode

logger = logging.getLogger("evolution")


def check_convergence(
    journal: EvolutionJournal,
    window_size: int = 5,
    improvement_threshold: float = 0.01,
) -> bool:
    """
    检查进化是否已收敛。

    收敛条件：
    最近 window_size 个好节点的指标没有显著提升（低于阈值）。

    Args:
        journal: 进化日志
        window_size: 滑动窗口大小
        improvement_threshold: 改进阈值

    Returns:
        True 表示已收敛
    """
    good_nodes = journal.good_nodes
    if len(good_nodes) < window_size:
        return False

    # 取最近 window_size 个好节点的指标
    recent_metrics = [n.metric.value for n in good_nodes[-window_size:]]

    # 过滤 None 值
    valid_metrics = [m for m in recent_metrics if m is not None]
    if len(valid_metrics) < 2:
        return False

    # 计算最大改进幅度
    min_metric = min(valid_metrics)
    max_metric = max(valid_metrics)

    # 归一化改进幅度
    if abs(max_metric) > 1e-10:
        improvement = (max_metric - min_metric) / abs(max_metric)
    else:
        improvement = max_metric - min_metric

    converged = improvement < improvement_threshold

    if converged:
        logger.info(
            f"进化收敛检测: 最近 {window_size} 个节点的改进幅度 {improvement:.4f} "
            f"低于阈值 {improvement_threshold}"
        )

    return converged


def run_evolution(
    task_desc: str,
    model: str = "gpt-4-turbo",
    temperature: float = 0.7,
    feedback_model: str | None = None,
    feedback_temp: float = 0.3,
    max_steps: int = 20,
    num_drafts: int = 3,
    debug_prob: float = 0.2,
    explore_prob: float = 0.15,
    max_debug_depth: int = 3,
    convergence_window: int = 5,
    convergence_threshold: float = 0.01,
    enable_convergence_check: bool = True,
) -> EvolutionJournal:
    """
    运行进化过程。

    这是 feeling 项目自进化系统的主入口。

    Args:
        task_desc: 任务描述
        model: 策略生成使用的模型
        temperature: 策略生成的温度
        feedback_model: 反馈评估使用的模型
        feedback_temp: 反馈评估的温度
        max_steps: 最大进化步数
        num_drafts: 初始草稿数量
        debug_prob: 调试概率
        explore_prob: 探索概率
        max_debug_depth: 最大调试深度
        convergence_window: 收敛检测窗口大小
        convergence_threshold: 收敛阈值
        enable_convergence_check: 是否启用收敛检测

    Returns:
        EvolutionJournal: 进化日志，包含所有策略和评估结果
    """
    logger.info(f"开始进化过程: {task_desc[:100]}...")

    # 初始化日志和 Agent
    journal = EvolutionJournal()
    agent = EvolutionAgent(
        task_desc=task_desc,
        journal=journal,
        model=model,
        temperature=temperature,
        feedback_model=feedback_model,
        feedback_temp=feedback_temp,
        num_drafts=num_drafts,
        debug_prob=debug_prob,
        explore_prob=explore_prob,
        max_debug_depth=max_debug_depth,
    )

    # 初始化 AgentRunner
    runner = AgentRunner(
        task_desc=task_desc,
        model=model,
        temperature=temperature,
    )

    # 将 runner 附加到 agent（用于多维指标计算）
    agent._last_runner = runner

    # 定义执行回调
    def exec_callback(strategy: str, reset_session: bool = True) -> 'ExecutionResult':
        return runner.run(strategy, reset_session)

    # 主进化循环
    global_step = 0
    while global_step < max_steps:
        logger.info(f"进化步骤 {global_step + 1}/{max_steps}")

        agent.step(exec_callback=exec_callback)
        global_step = len(journal)

        # 输出当前状态
        best_node = journal.get_best_node()
        if best_node:
            logger.info(
                f"当前最佳指标: {best_node.metric.value:.4f} "
                f"(节点 {best_node.id[:8]})"
            )

        # 收敛检测
        if enable_convergence_check and global_step >= convergence_window:
            if check_convergence(journal, convergence_window, convergence_threshold):
                logger.info(f"进化在步骤 {global_step} 收敛，提前终止")
                break

    # 清理
    runner.cleanup_session()

    # 输出最终统计
    stats = journal.get_evolution_stats()
    logger.info(f"进化完成: {stats}")

    return journal


def format_evolution_report(journal: EvolutionJournal) -> str:
    """
    格式化进化报告。

    Args:
        journal: 进化日志

    Returns:
        格式化的报告字符串
    """
    stats = journal.get_evolution_stats()
    best_node = journal.get_best_node()

    report_lines = [
        "=" * 60,
        "进化报告",
        "=" * 60,
        f"总节点数: {stats['total_nodes']}",
        f"好节点数: {stats['good_nodes']}",
        f"Buggy 节点数: {stats['buggy_nodes']}",
        f"阶段分布: 草稿={stats['stages']['draft']}, "
        f"改进={stats['stages']['improve']}, "
        f"调试={stats['stages']['debug']}",
        "-" * 60,
    ]

    if best_node:
        report_lines.extend([
            f"最佳策略指标: {best_node.metric.value:.4f}",
            f"最佳策略 ID: {best_node.id}",
            f"最佳策略设计: {best_node.plan}",
            "-" * 60,
            "最佳策略内容:",
            best_node.strategy[:500] + ("..." if len(best_node.strategy) > 500 else ""),
        ])

        if best_node.multi_metric:
            report_lines.extend([
                "-" * 60,
                "多维评估:",
                f"  任务完成度: {best_node.multi_metric.task_success:.2f}",
                f"  用户满意度: {best_node.multi_metric.user_satisfaction:.2f}",
                f"  执行效率: {best_node.multi_metric.efficiency:.2f}",
                f"  安全性: {best_node.multi_metric.safety:.2f}",
                f"  创造性: {best_node.multi_metric.creativity:.2f}",
                f"  综合分数: {best_node.multi_metric.composite_score:.4f}",
            ])

    report_lines.append("=" * 60)
    return "\n".join(report_lines)
