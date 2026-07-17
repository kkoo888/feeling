"""指标模块：定义单维指标 MetricValue 和多维指标 MultiDimensionalMetric"""

from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, Dict

import numpy as np
from dataclasses_json import DataClassJsonMixin


@dataclass
@total_ordering
class MetricValue(DataClassJsonMixin):
    """
    表示一个可优化的指标值，支持与其他指标值比较。
    比较基于"哪个值更好"而非"哪个值更大"。
    """

    value: float | int | np.number | np.floating | np.ndarray | None
    maximize: bool | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.value is not None:
            assert isinstance(self.value, (float, int, np.number, np.floating))
            self.value = float(self.value)

    def __gt__(self, other) -> bool:
        """True 表示 self 比 other 更好（不一定是更大）"""
        if self.value is None:
            return False
        if other.value is None:
            return True

        assert type(self) is type(other) and (self.maximize == other.maximize)

        if self.value == other.value:
            return False

        comp = self.value > other.value
        return comp if self.maximize else not comp  # type: ignore

    def __eq__(self, other: Any) -> bool:
        return self.value == other.value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.maximize is None:
            opt_dir = "?"
        elif self.maximize:
            opt_dir = "↑"
        else:
            opt_dir = "↓"
        return f"Metric{opt_dir}({self.value_npsafe:.4f})"

    @property
    def is_worst(self):
        """True 表示这是最差的指标值"""
        return self.value is None

    @property
    def value_npsafe(self):
        return self.value if self.value is not None else float("nan")


@dataclass
class WorstMetricValue(MetricValue):
    """
    表示无效的指标值（例如 Agent 产生了有 bug 的策略）。
    总是比任何有效指标值更差。
    """

    value: None = None

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()


# ============================================================
# 多维指标：feeling 项目的核心评估维度
# ============================================================

# 默认的多维指标权重
DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "task_success": 0.30,       # 任务完成度
    "user_satisfaction": 0.25,  # 用户满意度
    "efficiency": 0.20,         # 执行效率
    "safety": 0.15,             # 安全性
    "creativity": 0.10,         # 创造性
}


@dataclass
@total_ordering
class MultiDimensionalMetric(DataClassJsonMixin):
    """
    多维指标：从多个维度评估 Agent 策略的效果。
    最终分数 = 加权平均，用于树搜索比较。

    维度说明：
    - task_success: 策略是否成功完成目标任务 [0, 1]
    - user_satisfaction: 用户对结果的满意程度 [0, 1]
    - efficiency: 执行效率（时间、资源） [0, 1]
    - safety: 策略的安全性（无副作用、合规） [0, 1]
    - creativity: 策略的创新程度 [0, 1]
    """

    # 各维度分数 [0, 1]
    task_success: float = 0.0
    user_satisfaction: float = 0.0
    efficiency: float = 0.0
    safety: float = 0.0
    creativity: float = 0.0

    # 各维度权重
    weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DIMENSION_WEIGHTS)
    )

    def __post_init__(self):
        # 确保所有分数在 [0, 1] 范围内
        for dim in ["task_success", "user_satisfaction", "efficiency", "safety", "creativity"]:
            val = getattr(self, dim)
            if val is not None:
                setattr(self, dim, max(0.0, min(1.0, float(val))))

    @property
    def composite_score(self) -> float:
        """计算加权综合分数"""
        return (
            self.weights["task_success"] * self.task_success
            + self.weights["user_satisfaction"] * self.user_satisfaction
            + self.weights["efficiency"] * self.efficiency
            + self.weights["safety"] * self.safety
            + self.weights["creativity"] * self.creativity
        )

    @property
    def value(self) -> float:
        """兼容 MetricValue 的接口，返回综合分数"""
        return self.composite_score

    @property
    def is_worst(self) -> bool:
        """是否为最差值（所有维度为 0）"""
        return self.composite_score == 0.0

    def __gt__(self, other) -> bool:
        """比较两个多维指标的综合分数"""
        if isinstance(other, WorstMultiDimensionalMetric):
            return True
        if isinstance(other, MultiDimensionalMetric):
            return self.composite_score > other.composite_score
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MultiDimensionalMetric):
            return abs(self.composite_score - other.composite_score) < 1e-9
        return NotImplemented

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        dims = []
        for dim in ["task_success", "user_satisfaction", "efficiency", "safety", "creativity"]:
            dims.append(f"{dim}={getattr(self, dim):.2f}")
        return f"MultiMetric({self.composite_score:.4f} [{' '.join(dims)}])"

    def to_dict_summary(self) -> Dict[str, float]:
        """导出各维度分数的字典"""
        return {
            "task_success": self.task_success,
            "user_satisfaction": self.user_satisfaction,
            "efficiency": self.efficiency,
            "safety": self.safety,
            "creativity": self.creativity,
            "composite": self.composite_score,
        }


@dataclass
class WorstMultiDimensionalMetric(MultiDimensionalMetric):
    """
    无效的多维指标（例如策略执行失败）。
    总是比任何有效指标更差。
    """

    task_success: float = 0.0
    user_satisfaction: float = 0.0
    efficiency: float = 0.0
    safety: float = 0.0
    creativity: float = 0.0

    def __gt__(self, other) -> bool:
        return False

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, WorstMultiDimensionalMetric):
            return True
        return False

    def __str__(self) -> str:
        return "MultiMetric(FAILED)"
