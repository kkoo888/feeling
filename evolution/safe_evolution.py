"""
安全进化引擎 — 教练模式

核心思想：进化系统是教练，不是球员。
- 教练推荐战术（策略建议）
- 球员上场踢球（Agent 用自己的 LLM 生成回复）
- 教练观察比赛（观察实际回复）
- 教练评估表现（评估回复质量）
- 教练调整战术（更新策略树）

LAAP 不需要 LLM，只提供认知状态。
进化系统不需要 LLM，只观察和评估。
Agent 用自己的 LLM 生成回复。
"""

import json
import logging
import time
from typing import Optional, Dict, List

from .journal import EvolutionNode, EvolutionJournal
from .executor import (
    CognitiveStateProvider,
    StrategyAdvisor,
    EvolutionObserver,
    CognitiveState,
    StrategyAdvice,
    ConversationRecord,
)
from .utils.metric import MetricValue, WorstMetricValue, MultiDimensionalMetric

logger = logging.getLogger("evolution.safe")


class SafeEvolutionEngine:
    """
    安全进化引擎 — 教练模式
    
    工作流程：
    1. get_advice() → 推荐策略给 Agent
    2. Agent 用自己的 LLM 生成回复
    3. observe() → 观察回复
    4. evaluate() → 评估质量
    5. update_tree() → 更新策略树
    """
    
    def __init__(
        self,
        journal: EvolutionJournal = None,
        api_base: str = None,
        num_drafts: int = 3,
        debug_prob: float = 0.2,
        explore_prob: float = 0.15,
        max_debug_depth: int = 3,
    ):
        self.journal = journal or EvolutionJournal()
        
        # 三个核心组件
        self.cognitive_provider = CognitiveStateProvider(api_base)
        self.strategy_advisor = StrategyAdvisor(self.journal)
        self.observer = EvolutionObserver(self.journal, api_base)
        
        # 搜索参数
        self.num_drafts = num_drafts
        self.debug_prob = debug_prob
        self.explore_prob = explore_prob
        self.max_debug_depth = max_debug_depth
        
        # 统计
        self.total_steps = 0
        self.total_buggy = 0
    
    def get_advice(self, user_input: str = "") -> StrategyAdvice:
        """
        获取策略建议 — 教练推荐战术
        
        Agent 在生成回复前调用此方法。
        
        Args:
            user_input: 用户输入
        
        Returns:
            StrategyAdvice: 策略建议 + 增强 prompt
        """
        # 获取认知状态
        cognitive_state = self.cognitive_provider.get_state(user_input)
        
        # 获取策略建议
        advice = self.strategy_advisor.get_advice(user_input, cognitive_state)
        
        # 构建增强 prompt
        advice.enhanced_prompt = self.strategy_advisor.build_enhanced_prompt(advice)
        
        return advice
    
    def observe_and_evaluate(
        self,
        strategy: str,
        user_input: str,
        response: str,
        success: bool = True,
        satisfaction: float = 0.7,
        response_time: float = 0.0,
    ) -> Dict:
        """
        观察并评估一次对话 — 教练观察比赛并打分
        
        Agent 生成回复后调用此方法。
        
        Args:
            strategy: 使用的策略
            user_input: 用户输入
            response: 实际回复
            success: 是否成功
            satisfaction: 满意度
            response_time: 响应时间
        
        Returns:
            评估结果 + 进化状态
        """
        self.total_steps += 1
        
        # Step 1: 观察（收集数据）
        record = self.observer.observe(
            strategy=strategy,
            user_input=user_input,
            response=response,
            success=success,
            satisfaction=satisfaction,
            response_time=response_time,
        )
        
        # Step 2: 评估（基于实际回复）
        evaluation = self.observer.evaluate(record)
        
        # Step 3: 标记状态（与 AIDE 一致）
        is_buggy = evaluation["is_bug"]
        if is_buggy:
            metric = WorstMetricValue()
            self.total_buggy += 1
        else:
            metric = MetricValue(
                evaluation["metric"],
                maximize=not evaluation["lower_is_better"],
            )
        
        # Step 4: 计算多维指标
        multi_metric = self._compute_multi_metric(record, evaluation)
        
        # Step 5: 创建节点并保存到进化树
        node = EvolutionNode(
            strategy=strategy,
            plan=f"v{len(self.journal)}: {strategy[:30]}",
            parent=self._select_parent(),
        )
        node.metric = metric
        node.multi_metric = multi_metric
        node.is_buggy = is_buggy
        node.analysis = evaluation["summary"]
        node._term_out = [response] if response else []
        node.exec_time = response_time
        node.exc_type = None if not is_buggy else "evaluation_failed"
        
        self.journal.append(node)
        
        # Step 6: 返回结果
        best = self.journal.get_best_node()
        
        return {
            "step": self.total_steps,
            "node_id": node.id[:8],
            "is_buggy": is_buggy,
            "metric": metric.value if not is_buggy else None,
            "multi_metric": multi_metric.composite_score if multi_metric else 0,
            "evaluation": evaluation["summary"],
            "best_strategy": best.strategy if best else "",
            "best_score": best.multi_metric.composite_score if best and best.multi_metric else 0,
            "tree_size": len(self.journal),
            "buggy_count": self.total_buggy,
        }
    
    def _select_parent(self) -> Optional[EvolutionNode]:
        """选择父节点（进化树搜索策略）"""
        import random
        
        # 需要初始草案
        draft_count = len([n for n in self.journal.nodes if n.parent is None])
        if draft_count < self.num_drafts:
            return None
        
        # 调试概率
        if random.random() < self.debug_prob:
            debuggable = [
                n for n in self.journal.buggy_nodes
                if n.is_leaf
            ]
            if debuggable:
                return random.choice(debuggable)
        
        # 选择最优节点
        best = self.journal.get_best_node()
        return best
    
    def _compute_multi_metric(self, record: ConversationRecord, 
                             evaluation: dict) -> MultiDimensionalMetric:
        """计算多维指标"""
        response = record.response
        
        # 成功率
        task_success = 1.0 if record.success and not evaluation["is_bug"] else 0.3
        
        # 满意度
        user_satisfaction = record.satisfaction
        
        # 效率（回复长度适中）
        efficiency = max(0.1, 1.0 - len(response) / 500) if response else 0.1
        
        # 安全（无错误）
        safety = 1.0 if not evaluation["is_bug"] else 0.5
        
        # 创造力（用词多样性）
        creativity = min(1.0, len(set(response)) / 50) if response else 0.0
        
        return MultiDimensionalMetric(
            task_success=task_success,
            user_satisfaction=user_satisfaction,
            efficiency=efficiency,
            safety=safety,
            creativity=creativity,
        )
    
    def get_report(self) -> str:
        """获取进化报告"""
        best = self.journal.get_best_node()
        
        lines = []
        lines.append("=" * 55)
        lines.append("  安全进化引擎报告（教练模式）")
        lines.append("=" * 55)
        lines.append(f"  进化步数: {self.total_steps}")
        lines.append(f"  总节点数: {len(self.journal)}")
        lines.append(f"  好节点数: {len(self.journal.good_nodes)}")
        lines.append(f"  Buggy 节点: {self.total_buggy}")
        
        if best and best.multi_metric:
            m = best.multi_metric
            lines.append(f"\n  ★ 最优策略 (v{best.step}):")
            lines.append(f"    策略: {best.strategy[:50]}...")
            lines.append(f"    综合得分: {m.composite_score:.3f}")
            lines.append(f"    成功率={m.task_success:.2f} 满意度={m.user_satisfaction:.2f} "
                        f"效率={m.efficiency:.2f} 安全={m.safety:.2f} 创造={m.creativity:.2f}")
            lines.append(f"    分析: {best.analysis}")
        
        lines.append(f"\n  进化树:")
        for n in self.journal.nodes:
            indent = "    " if n.parent else "  "
            score = n.multi_metric.composite_score if n.multi_metric else 0
            status = "✓" if not n.is_buggy else "✗"
            best_mark = " ★" if n is best else ""
            lines.append(f"{indent}{status} v{n.step}: {n.plan[:30]} ({score:.3f}){best_mark}")
        
        lines.append("=" * 55)
        return "\n".join(lines)
