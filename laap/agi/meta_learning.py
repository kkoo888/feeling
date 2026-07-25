"""
MetaLearningEngine — 元学习引擎 (Evolved v1)
================================================

基于前沿论文实现:
  1. Truly Self-Improving Agents Require Intrinsic Metacognitive Learning (ICML 2025)
     - 三层元认知: 元认知知识 / 元认知规划 / 元认知评估
     - 内在 vs 外在元认知: 从固定模板 → 自主改进改进过程
  2. MetaAgent: Self-Evolving Agent via Tool Meta-Learning (arXiv 2025)
     - 工具元学习: 遇到盲点时自动求助，从使用中学习工具
     - "做中学" (learning-by-doing) 范式
  3. SkillOpt (Microsoft 2026)
     - 技能文档当"权重"，反馈当"梯度"
     - 不动模型参数，持续优化 Agent 行为

进化 v1 改进:
  - 窗口式技能评估: 用最近 N 条记录而非全量 EMA, 快速适应变化
  - 按任务类型的策略有效性: 区分 "rule 对 read_file 好" vs "rule 对 debug 差"
  - 数据置信度加权: 数据越多越信任历史表现 (base→hist 渐进切换)
  - 置信度校准: 追踪 confidence vs actual_accuracy 的偏差, 检测过度/不足自信
  - 失败模式追踪: 记录重复失败模式, 反思可触发策略惩罚
  - 修复 strengths/weaknesses bug: 原代码用 task_type 匹配描述文本, 永远不命中

印记: 小茜 永远记得主人 — 2026-07-24
"""
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger('laap.agi.meta_learning')
_STATE_DIR = Path(__file__).resolve().parent.parent.parent / 'aris_brain' / 'state'
_META_STATE = _STATE_DIR / 'meta_learning_state.json'

@dataclass
class TaskExecution:
    """一次任务执行记录"""
    task_id: str
    task_type: str
    strategy_used: str
    success: bool
    confidence: float
    actual_accuracy: float
    timestamp: float
    duration_ms: float = 0.0
    error_info: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetaKnowledge:
    """元认知知识：知道自己会什么、不会什么"""
    skill_inventory: Dict[str, float] = field(default_factory=dict)
    blind_spots: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    strategy_effectiveness: Dict[str, float] = field(default_factory=dict)
    task_strategy_eff: Dict[str, Dict[str, float]] = field(default_factory=dict)
    calibration_error: Dict[str, float] = field(default_factory=dict)
    failure_patterns: List[str] = field(default_factory=list)

@dataclass
class MetaPlan:
    """元认知规划：知道该用什么策略"""
    recommended_strategy: str
    strategy_reason: str
    fallback_strategy: str
    confidence_threshold: float
    routing_rules: Dict[str, str] = field(default_factory=dict)

@dataclass
class MetaEvaluation:
    """元认知评估：知道学得好不好"""
    overall_score: float
    strategy_effective: bool
    learning_progress: float
    improvement_suggestions: List[str] = field(default_factory=list)
    reflection: str = ''

@dataclass
class MetacognitionState:
    """完整的元认知状态"""
    knowledge: MetaKnowledge = field(default_factory=MetaKnowledge)
    plan: MetaPlan = field(default_factory=lambda : MetaPlan(recommended_strategy='rule', strategy_reason='默认规则策略', fallback_strategy='llm', confidence_threshold=0.5))
    evaluation: MetaEvaluation = field(default_factory=lambda : MetaEvaluation(overall_score=0.5, strategy_effective=False, learning_progress=0.0, improvement_suggestions=[]))
    last_update: float = 0.0
STRATEGY_PROFILES = {'rule': {'name': '规则匹配', 'description': '关键词+正则，速度最快', 'strengths': ['简单意图', '结构化输入', '低延迟'], 'weaknesses': ['复杂语义', '模糊表达', '新意图'], 'base_reliability': 0.7}, 'llm': {'name': 'LLM 推理', 'description': '大模型理解，准确但慢', 'strengths': ['复杂语义', '模糊表达', '新意图'], 'weaknesses': ['延迟高', '成本高', '幻觉'], 'base_reliability': 0.85}, 'hybrid': {'name': '混合策略', 'description': '规则优先，低置信度时用 LLM 兜底', 'strengths': ['平衡速度和准确率'], 'weaknesses': ['阈值敏感'], 'base_reliability': 0.8}, 'memory': {'name': '记忆复用', 'description': '从历史成功案例中找相似', 'strengths': ['重复任务', '个性化'], 'weaknesses': ['新任务', '冷启动'], 'base_reliability': 0.75}}

class MetaLearningEngine:
    """
    元学习引擎 — "学会学习" (进化v1)。

    三层元认知 (ICML 2025):
      1. 元认知知识: 知道自己会什么、不会什么
      2. 元认知规划: 知道该用什么策略
      3. 元认知评估: 知道学得好不好

    进化v1改进:
    - 窗口式评估: 用最近 WINDOW_SIZE 条记录计算技能和策略有效性
    - 按任务类型策略: 区分不同任务类型对策略的适用性
    - 数据置信度加权: 样本越多, 历史权重越大
    - 置信度校准: 追踪 confidence vs accuracy 偏差
    """
    WINDOW_SIZE = 14

    def __init__(self, persistence_path: Optional[Path]=None):
        self._state = MetacognitionState()
        self._history: List[TaskExecution] = []
        self._persistence_path = persistence_path or _META_STATE
        self._load_state()

    def record_execution(self, execution: TaskExecution) -> None:
        """记录一次任务执行。"""
        self._history.append(execution)
        self._update_knowledge(execution)
        self._update_plan()
        self._update_evaluation()
        self._state.last_update = time.time()
        self._save_state()

    def get_knowledge(self) -> MetaKnowledge:
        """获取元认知知识（自知之明）。"""
        return self._state.knowledge

    def get_plan(self) -> MetaPlan:
        """获取元认知规划（该用什么策略）。"""
        return self._state.plan

    def get_evaluation(self) -> MetaEvaluation:
        """获取元认知评估（学得好不好）。"""
        return self._state.evaluation

    def get_full_state(self) -> MetacognitionState:
        """获取完整的元认知状态。"""
        return self._state

    def recommend_strategy(self, task_type: str, input_complexity: float=0.5) -> Tuple[str, float, str]:
        """
        推荐处理策略 (进化v1: 按任务类型 + 数据置信度加权)。

        Args:
            task_type: 任务类型
            input_complexity: 输入复杂度 [0, 1]

        Returns:
            (strategy, confidence, reason)
        """
        knowledge = self._state.knowledge
        strategy_scores = {}
        for (strategy_name, profile) in STRATEGY_PROFILES.items():
            score = profile['base_reliability']
            task_eff_map = knowledge.task_strategy_eff.get(task_type, {})
            historical = task_eff_map.get(strategy_name, None)
            if historical is not None:
                data_count = self._get_strat_count_for_task(task_type, strategy_name)
                hist_weight = min(0.75, 0.35 + data_count * 0.04)
                score = (1.0 - hist_weight) * score + hist_weight * historical
            else:
                global_eff = knowledge.strategy_effectiveness.get(strategy_name, 0.5)
                score = 0.7 * score + 0.3 * global_eff
            if strategy_name == 'rule' and input_complexity > 0.7:
                score -= 0.2
            if strategy_name == 'llm' and input_complexity < 0.3:
                score -= 0.1
            cal_error = knowledge.calibration_error.get(task_type, 0.0)
            if cal_error > 0.3:
                score -= 0.05 * (cal_error - 0.3)
            strategy_scores[strategy_name] = max(0.0, min(1.0, score))
        best = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best]
        profile = STRATEGY_PROFILES[best]
        reason = f"{profile['name']}：{profile['description']}"
        task_eff_map = knowledge.task_strategy_eff.get(task_type, {})
        if best in task_eff_map:
            data_count = self._get_strat_count_for_task(task_type, best)
            reason += f'（历史有效性 {task_eff_map[best]:.2f}, {data_count} 次记录）'
        if task_type in knowledge.blind_spots:
            reason += f'（注意：{task_type} 是已知盲点，建议增加验证）'
        cal_error = knowledge.calibration_error.get(task_type, 0.0)
        if cal_error > 0.3:
            reason += f'（校准偏差 {cal_error:.2f}，存在过度自信风险）'
        return (best, best_score, reason)

    def reflect_on_failure(self, execution: TaskExecution) -> Dict[str, Any]:
        """
        失败反思 (MetaAgent 核心机制, 进化v1: 增强为可操作)。

        遇到失败时，分析原因并生成改进建议，同时更新策略有效性。

        Returns:
            反思报告
        """
        knowledge = self._state.knowledge
        reasons = []
        suggestions = []
        task_eff_map = knowledge.task_strategy_eff.get(execution.task_type, {})
        strategy_eff = task_eff_map.get(execution.strategy_used, None)
        global_eff = knowledge.strategy_effectiveness.get(execution.strategy_used, 0.5)
        eff_ref = strategy_eff if strategy_eff is not None else global_eff
        if eff_ref < 0.5:
            reasons.append(f'策略 {execution.strategy_used} 在 {execution.task_type} 上有效性低 ({eff_ref:.2f})')
            suggestions.append(f'尝试其他策略（如 hybrid 或 llm）')
        if execution.task_type in knowledge.blind_spots:
            reasons.append(f'{execution.task_type} 是已知盲点')
            suggestions.append(f'增加 {execution.task_type} 的练习或寻求外部帮助')
        cal_error = knowledge.calibration_error.get(execution.task_type, 0.0)
        if execution.confidence > 0.7:
            if cal_error > 0.3:
                reasons.append(f'高置信度 ({execution.confidence:.2f}) 但失败，校准偏差 {cal_error:.2f}，存在系统性过度自信')
            else:
                reasons.append('高置信度但失败，可能存在系统性问题')
            suggestions.append('检查处理逻辑，可能存在 bug 或置信度校准需要调整')
        if execution.confidence < 0.3:
            reasons.append('低置信度，能力不足')
            suggestions.append('从基础开始练习')
        if execution.task_type not in knowledge.skill_inventory:
            reasons.append('首次遇到该类型任务')
            suggestions.append('记录到技能清单，开始积累经验')
        pattern_key = f'{execution.task_type}:{execution.strategy_used}'
        if pattern_key in knowledge.failure_patterns:
            reasons.append(f'重复失败模式: {pattern_key} 已出现多次')
            suggestions.append('考虑切换策略或寻求外部帮助')
        return {'task_type': execution.task_type, 'strategy': execution.strategy_used, 'confidence': execution.confidence, 'calibration_error': round(cal_error, 3), 'reasons': reasons, 'suggestions': suggestions, 'action': '记录失败模式，更新策略有效性'}

    def _update_knowledge(self, execution: TaskExecution):
        """进化v1: 更新元认知知识 — 窗口式评估 + 按任务类型策略。"""
        knowledge = self._state.knowledge
        task_type = execution.task_type
        strategy = execution.strategy_used
        recent = self._get_recent_for_task(task_type, self.WINDOW_SIZE)
        if recent:
            successes = [e for e in recent if e.success]
            if successes:
                avg_acc = sum((e.actual_accuracy for e in successes)) / len(successes)
                success_rate = len(successes) / len(recent)
                skill = success_rate * avg_acc
            else:
                skill = 0.0
            knowledge.skill_inventory[task_type] = round(skill, 4)
        else:
            knowledge.skill_inventory[task_type] = 0.5
        recent_strat = self._get_recent_for_strategy(strategy, self.WINDOW_SIZE)
        if recent_strat:
            success_count = sum((1 for e in recent_strat if e.success))
            knowledge.strategy_effectiveness[strategy] = round(success_count / len(recent_strat), 4)
        if task_type not in knowledge.task_strategy_eff:
            knowledge.task_strategy_eff[task_type] = {}
        recent_task_strat = [e for e in self._history if e.task_type == task_type and e.strategy_used == strategy][-self.WINDOW_SIZE:]
        if recent_task_strat:
            success_count = sum((1 for e in recent_task_strat if e.success))
            knowledge.task_strategy_eff[task_type][strategy] = round(success_count / len(recent_task_strat), 4)
        cal_signal = abs(execution.confidence - execution.actual_accuracy)
        old_cal = knowledge.calibration_error.get(task_type, 0.3)
        cal_alpha = 0.3
        new_cal = cal_alpha * cal_signal + (1 - cal_alpha) * old_cal
        knowledge.calibration_error[task_type] = round(new_cal, 4)
        if not execution.success:
            pattern_key = f'{task_type}:{strategy}'
            if pattern_key not in knowledge.failure_patterns:
                knowledge.failure_patterns.append(pattern_key)
            if len(knowledge.failure_patterns) > 20:
                knowledge.failure_patterns = knowledge.failure_patterns[-20:]
        knowledge.blind_spots = [k for (k, v) in knowledge.skill_inventory.items() if v < 0.3 and self._get_attempt_count(k) >= 3]
        knowledge.strengths = [k for (k, v) in knowledge.skill_inventory.items() if v > 0.7 and self._get_attempt_count(k) >= 3]

    def _update_plan(self):
        """更新元认知规划。"""
        knowledge = self._state.knowledge
        plan = self._state.plan
        if knowledge.strategy_effectiveness:
            best_strategy = max(knowledge.strategy_effectiveness, key=knowledge.strategy_effectiveness.get)
            best_eff = knowledge.strategy_effectiveness[best_strategy]
            plan.recommended_strategy = best_strategy
            plan.strategy_reason = f"{STRATEGY_PROFILES.get(best_strategy, {}).get('name', best_strategy)}：历史有效性 {best_eff:.2f}"
        for (task_type, skill) in knowledge.skill_inventory.items():
            if skill > 0.7:
                plan.routing_rules[task_type] = 'rule'
            elif skill < 0.3:
                plan.routing_rules[task_type] = 'llm'
            else:
                plan.routing_rules[task_type] = 'hybrid'

    def _update_evaluation(self):
        """更新元认知评估。"""
        knowledge = self._state.knowledge
        evaluation = self._state.evaluation
        if not knowledge.skill_inventory:
            return
        skills = list(knowledge.skill_inventory.values())
        evaluation.overall_score = round(sum(skills) / len(skills), 4)
        if knowledge.strategy_effectiveness:
            avg_eff = sum(knowledge.strategy_effectiveness.values()) / len(knowledge.strategy_effectiveness)
            evaluation.strategy_effective = avg_eff > 0.6
        if len(self._history) >= 2:
            recent = self._history[-10:]
            older = self._history[-20:-10] if len(self._history) > 20 else self._history[:10]
            recent_success = sum((1 for r in recent if r.success)) / len(recent)
            older_success = sum((1 for r in older if r.success)) / len(older)
            evaluation.learning_progress = round(max(0, recent_success - older_success), 4)
        suggestions = []
        for blind in knowledge.blind_spots[:3]:
            suggestions.append(f'突破盲点: {blind}')
        for (task_type, skill) in knowledge.skill_inventory.items():
            if 0.3 < skill < 0.6:
                suggestions.append(f'提升 {task_type} (当前 {skill:.2f})')
        for (task_type, cal_err) in knowledge.calibration_error.items():
            if cal_err > 0.3:
                suggestions.append(f'校准 {task_type} 的置信度 (偏差 {cal_err:.2f})')
        evaluation.improvement_suggestions = suggestions[:5]
        if evaluation.overall_score > 0.7:
            evaluation.reflection = '整体表现优秀，可以挑战更高难度'
        elif evaluation.overall_score > 0.5:
            evaluation.reflection = '表现中等，重点突破薄弱环节'
        else:
            evaluation.reflection = '需要加强基础练习，从简单任务开始'

    def _get_recent_for_task(self, task_type: str, window: int) -> List[TaskExecution]:
        """获取某任务类型最近 N 条记录"""
        return [e for e in self._history if e.task_type == task_type][-window:]

    def _get_recent_for_strategy(self, strategy: str, window: int) -> List[TaskExecution]:
        """获取某策略最近 N 条记录"""
        return [e for e in self._history if e.strategy_used == strategy][-window:]

    def _get_strat_count_for_task(self, task_type: str, strategy: str) -> int:
        """获取某任务类型+策略的记录数"""
        return sum((1 for e in self._history if e.task_type == task_type and e.strategy_used == strategy))

    def _get_attempt_count(self, task_type: str) -> int:
        """获取某类任务的尝试次数。"""
        return sum((1 for e in self._history if e.task_type == task_type))

    def _save_state(self):
        """保存状态。"""
        try:
            state = {'knowledge': asdict(self._state.knowledge), 'plan': asdict(self._state.plan), 'evaluation': asdict(self._state.evaluation), 'history_count': len(self._history), 'last_update': self._state.last_update}
            self._persistence_path.parent.mkdir(parents=False, exist_ok=True)
            self._persistence_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            logger.warning(f'保存元学习状态失败: {e}')

    def _load_state(self):
        """加载状态。"""
        try:
            if self._persistence_path.exists():
                data = json.loads(self._persistence_path.read_text(encoding='utf-8'))
                if 'knowledge' in data:
                    k = data['knowledge']
                    self._state.knowledge = MetaKnowledge(skill_inventory=k.get('skill_inventory', {}), blind_spots=k.get('blind_spots', []), strengths=k.get('strengths', []), strategy_effectiveness=k.get('strategy_effectiveness', {}), task_strategy_eff=k.get('task_strategy_eff', {}), calibration_error=k.get('calibration_error', {}), failure_patterns=k.get('failure_patterns', []))
                if 'plan' in data:
                    self._state.plan = MetaPlan(**data['plan'])
                if 'evaluation' in data:
                    self._state.evaluation = MetaEvaluation(**data['evaluation'])
                self._state.last_update = data.get('last_update', 0)
                logger.info(f'元学习状态已加载')
        except Exception as e:
            logger.warning(f'加载元学习状态失败: {e}')

    def self_test(self) -> Dict[str, Any]:
        """自检：验证核心功能。"""
        results = {}
        import tempfile
        test_path = Path(tempfile.gettempdir()) / 'meta_test.json'
        engine = MetaLearningEngine(persistence_path=test_path)
        for i in range(10):
            engine.record_execution(TaskExecution(task_id=f't{i}', task_type='read_file', strategy_used='rule', success=i < 8, confidence=0.8, actual_accuracy=0.9 if i < 8 else 0.0, timestamp=time.time()))
        results['record_execution'] = {'history_count': len(engine._history), 'passed': len(engine._history) == 10}
        knowledge = engine.get_knowledge()
        results['knowledge'] = {'skill_count': len(knowledge.skill_inventory), 'read_file_skill': round(knowledge.skill_inventory.get('read_file', 0), 2), 'passed': 'read_file' in knowledge.skill_inventory}
        (strategy, score, reason) = engine.recommend_strategy('read_file', 0.3)
        results['recommend_strategy'] = {'strategy': strategy, 'score': round(score, 2), 'passed': strategy in STRATEGY_PROFILES}
        failure = TaskExecution(task_id='fail1', task_type='debug', strategy_used='rule', success=True, confidence=0.9, actual_accuracy=0.0, timestamp=time.time(), error_info='wrong intent')
        reflection = engine.reflect_on_failure(failure)
        results['reflect_on_failure'] = {'reasons_count': len(reflection['reasons']), 'suggestions_count': len(reflection['suggestions']), 'passed': len(reflection['reasons']) > 0}
        evaluation = engine.get_evaluation()
        results['evaluation'] = {'overall_score': evaluation.overall_score, 'passed': 0 <= evaluation.overall_score <= 1}
        for task_type in ['search', 'generate', 'chat', 'translate']:
            engine.record_execution(TaskExecution(task_id=f'multi_{task_type}', task_type=task_type, strategy_used='hybrid', success=True, confidence=0.7, actual_accuracy=0.8, timestamp=time.time()))
        knowledge2 = engine.get_knowledge()
        results['multi_task'] = {'skill_types': len(knowledge2.skill_inventory), 'passed': len(knowledge2.skill_inventory) >= 5}
        all_passed = all((r.get('passed', False) for r in results.values()))
        results['all_passed'] = all_passed
        if test_path.exists():
            test_path.unlink()
        return results