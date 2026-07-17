"""
feeling 项目 - 自进化系统 (Evolution System)

从 AIDE 项目改造而来，核心变化：
- code → strategy（策略）
- CodeInterpreter → AgentRunner（Agent 对话执行器）
- 新增 _explore() 探索全新方向
- 新增多维指标 MultiDimensionalMetric
- 新增加收敛检测

核心组件：
- EvolutionAgent: 进化决策引擎
- EvolutionNode: 进化树节点
- EvolutionJournal: 进化日志
- AgentRunner: 策略执行器
- run_evolution: 主运行入口
"""

# 基础组件（无外部依赖）
from .journal import EvolutionNode, EvolutionJournal
from .runner import AgentRunner, ExecutionResult, ConversationMetrics
from .utils.metric import (
    MetricValue,
    WorstMetricValue,
    MultiDimensionalMetric,
    WorstMultiDimensionalMetric,
)

# 高级组件（依赖 LLM 后端，延迟导入）
def __getattr__(name):
    if name == "EvolutionAgent":
        from .agent import EvolutionAgent
        return EvolutionAgent
    elif name == "run_evolution":
        from .run import run_evolution
        return run_evolution
    elif name == "format_evolution_report":
        from .run import format_evolution_report
        return format_evolution_report
    elif name == "check_convergence":
        from .run import check_convergence
        return check_convergence
    raise AttributeError(f"module 'evolution' has no attribute {name!r}")


__all__ = [
    "EvolutionAgent",
    "EvolutionNode",
    "EvolutionJournal",
    "AgentRunner",
    "ExecutionResult",
    "ConversationMetrics",
    "run_evolution",
    "format_evolution_report",
    "check_convergence",
    "MetricValue",
    "WorstMetricValue",
    "MultiDimensionalMetric",
    "WorstMultiDimensionalMetric",
]
