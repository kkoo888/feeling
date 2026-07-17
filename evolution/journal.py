"""
进化日志：feeling 项目自进化系统的核心数据结构

包含：
- 生成的策略样本（strategy）
- 策略之间的树形关系
- 执行结果
- 多维评估信息
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from dataclasses_json import DataClassJsonMixin
from .runner import ExecutionResult
from .utils.metric import MetricValue, MultiDimensionalMetric, WorstMultiDimensionalMetric
from .utils.response import trim_long_string


@dataclass(eq=False)
class EvolutionNode(DataClassJsonMixin):
    """进化树中的单个节点。包含策略、执行结果和评估信息。"""

    # ---- 策略 & 计划 ----
    strategy: str  # 原 AIDE 的 code 字段，现为策略文本/行为规则/参数
    plan: str = field(default=None, kw_only=True)  # type: ignore

    # ---- 通用属性 ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["EvolutionNode"] = field(default=None, kw_only=True)
    children: set["EvolutionNode"] = field(default_factory=set, kw_only=True)

    # ---- 执行信息 ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- 评估 ----
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    multi_metric: MultiDimensionalMetric = field(default=None, kw_only=True)  # type: ignore
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve", "explore"]:
        """
        返回节点的阶段：
        - "draft": 初始策略草稿
        - "debug": 调试修复
        - "improve": 改进优化
        - "explore": 探索全新方向
        """
        if self.parent is None:
            return "draft"
        if self.parent.is_buggy:
            return "debug"
        return "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """吸收执行结果"""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """获取执行输出（截断后的）"""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """检查是否为叶节点"""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, EvolutionNode) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        当前调试路径的长度：
        - 0: 非调试节点
        - 1: 父节点有 bug 但祖父节点没有
        - n: 连续 n 次调试步骤
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore


@dataclass
class EvolutionJournal(DataClassJsonMixin):
    """进化日志：管理进化树中的所有节点。"""

    nodes: list[EvolutionNode] = field(default_factory=list)

    def __getitem__(self, idx: int) -> EvolutionNode:
        return self.nodes[idx]

    def __len__(self) -> int:
        """返回节点数量"""
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def append(self, node: EvolutionNode) -> None:
        """追加新节点到日志"""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[EvolutionNode]:
        """返回所有初始草稿节点"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[EvolutionNode]:
        """返回所有被认为有 bug 的节点"""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[EvolutionNode]:
        """返回所有无 bug 的节点"""
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[MetricValue]:
        """返回所有指标值的历史"""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good=True) -> None | EvolutionNode:
        """返回目前找到的最佳策略（指标最高的节点）"""
        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes
        return max(nodes, key=lambda n: n.metric)

    def generate_summary(self, include_strategy: bool = False) -> str:
        """生成日志摘要，供 Agent 参考"""
        summary = []
        for n in self.good_nodes:
            summary_part = f"设计: {n.plan}\n"
            if include_strategy:
                summary_part += f"策略: {n.strategy}\n"
            summary_part += f"结果: {n.analysis}\n"
            summary_part += f"验证指标: {n.metric.value}\n"
            if n.multi_metric is not None:
                summary_part += f"多维评估: {n.multi_metric}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)

    def get_evolution_stats(self) -> dict:
        """获取进化统计信息"""
        total = len(self.nodes)
        good = len(self.good_nodes)
        buggy = len(self.buggy_nodes)
        best = self.get_best_node()

        stats = {
            "total_nodes": total,
            "good_nodes": good,
            "buggy_nodes": buggy,
            "best_metric": best.metric.value if best else None,
            "stages": {
                "draft": len([n for n in self.nodes if n.stage_name == "draft"]),
                "improve": len([n for n in self.nodes if n.stage_name == "improve"]),
                "debug": len([n for n in self.nodes if n.stage_name == "debug"]),
            },
        }
        return stats
