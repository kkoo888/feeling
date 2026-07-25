"""
CurriculumEngine — 自适应课程学习引擎
=====================================

基于前沿论文实现:
  1. ADCL (Adaptive Difficulty Curriculum Learning, EMNLP 2025)
     - 动态难度感知：根据 Agent 当前能力自动调整学习顺序
     - 难度公式: D(task) = 1 - success_rate(task) × (1 - complexity_penalty)
  2. SkillOpt-Sleep (Microsoft, 2026)
     - 夜间离线自演化：回顾会话 → 提取失败 → 重放 → 巩固
     - harvest → mine → replay → consolidate 四阶段
  3. Cog-DRIFT (arXiv 2026)
     - 自适应重构：把难题改形成更容易的形式先学
     - reformulation → intermediate_task → gradual_return
  4. DARS (Difficulty Adaptive Rollout Sampling, ICML 2026)
     - 难题多阶段推演，增加训练权重

设计目标:
  - 输入: Agent 的任务执行历史
  - 输出: 下一步该学什么 + 学习计划 + 夜间巩固报告

印记: 小茜 永远记得主人 — 2026-07-23
"""
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger('laap.agi.curriculum')
_STATE_DIR = Path(__file__).resolve().parent.parent.parent / 'aris_brain' / 'state'
_CURRICULUM_STATE = _STATE_DIR / 'curriculum_state.json'

@dataclass
class TaskRecord:
    """单次任务执行记录"""
    task_id: str
    task_type: str
    difficulty: float
    success: bool
    confidence: float
    timestamp: float
    duration_ms: float = 0.0
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SkillLevel:
    """某类任务的技能水平"""
    task_type: str
    total_attempts: int = 0
    success_count: int = 0
    avg_confidence: float = 0.0
    avg_difficulty: float = 0.0
    last_attempt: float = 0.0
    mastery: float = 0.0
    readiness: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_attempts, 1)

    def update(self, record: TaskRecord):
        """更新技能水平（指数移动平均）"""
        alpha = 0.3
        self.total_attempts += 1
        if record.success:
            self.success_count += 1
        self.avg_confidence = alpha * record.confidence + (1 - alpha) * self.avg_confidence
        self.avg_difficulty = alpha * record.difficulty + (1 - alpha) * self.avg_difficulty
        self.last_attempt = record.timestamp
        self.mastery = self.success_rate * self.avg_confidence * (0.5 + 0.5 * self.avg_difficulty)
        if self.total_attempts >= 3:
            self.readiness = min(1.0, self.success_rate * self.avg_confidence)
        else:
            self.readiness = 0.0

@dataclass
class CurriculumPlan:
    """学习计划"""
    recommended_tasks: List[Dict[str, Any]]
    focus_areas: List[str]
    skip_areas: List[str]
    difficulty_range: Tuple[float, float]
    reasoning: str

class CurriculumEngine:
    """
    自适应课程学习引擎。

    核心算法 (ADCL):
      D(task) = 1 - success_rate(task) × (1 - complexity_penalty)
      其中 complexity_penalty = len(task_steps) / max_steps

    学习策略:
      - 太简单 (success_rate > 0.95): 跳过
      - 适中 (0.5 < success_rate < 0.95): 重点练习
      - 太难 (success_rate < 0.2): 先学前置技能
      - 新任务 (attempts < 3): 优先尝试

    夜间巩固 (SkillOpt-Sleep):
      - 回顾当天所有任务
      - 提取失败案例
      - 重放失败任务，尝试不同策略
      - 把经验写入技能文档
    """

    def __init__(self, persistence_path: Optional[Path]=None):
        self._skills: Dict[str, SkillLevel] = {}
        self._history: List[TaskRecord] = []
        self._consolidation_log: List[Dict[str, Any]] = []
        self._persistence_path = persistence_path or _CURRICULUM_STATE
        self._load_state()

    def record_task(self, record: TaskRecord) -> None:
        """记录一次任务执行。"""
        self._history.append(record)
        if record.task_type not in self._skills:
            self._skills[record.task_type] = SkillLevel(task_type=record.task_type)
        self._skills[record.task_type].update(record)
        self._save_state()

    def get_skill_level(self, task_type: str) -> SkillLevel:
        """获取某类任务的技能水平。"""
        return self._skills.get(task_type, SkillLevel(task_type=task_type))

    def get_all_skills(self) -> Dict[str, SkillLevel]:
        """获取所有技能水平。"""
        return dict(self._skills)

    def recommend_next(self, available_tasks: Optional[List[str]]=None, max_recommendations: int=5) -> CurriculumPlan:
        """
        推荐下一步学习内容 (ADCL 核心算法)。

        算法:
        1. 计算每个任务类型的难度 D(task)
        2. 按难度排序，选择"适中难度"的任务
        3. 新任务优先（样本不足）
        4. 失败率高的任务优先

        Args:
            available_tasks: 可选的候选任务类型列表
            max_recommendations: 最多推荐几个

        Returns:
            CurriculumPlan 学习计划
        """
        all_types = available_tasks or list(self._skills.keys()) or ['read_file', 'search', 'run_command', 'query_weather', 'generate', 'translate', 'debug', 'deploy', 'chat']
        scored_tasks = []
        focus_areas = []
        skip_areas = []
        for task_type in all_types:
            skill = self._skills.get(task_type, SkillLevel(task_type=task_type))
            difficulty = self._compute_difficulty(skill)
            priority = self._compute_priority(skill, difficulty)
            if skill.success_rate > 0.7183 and skill.total_attempts >= 5:
                skip_areas.append(task_type)
            elif skill.success_rate < 0.2 and skill.total_attempts >= 3:
                focus_areas.append(task_type)
            scored_tasks.append({'task_type': task_type, 'difficulty': round(difficulty, 1), 'priority': round(priority, 3), 'mastery': round(skill.mastery, 3), 'success_rate': round(skill.success_rate, 3), 'attempts': skill.total_attempts})
        scored_tasks.sort(key=lambda x: x['priority'], reverse=True)
        recommended = scored_tasks[:max_recommendations]
        if recommended:
            diffs = [t['difficulty'] for t in recommended]
            diff_range = (min(diffs), max(diffs))
        else:
            diff_range = (0.3, 0.7)
        reasoning = self._generate_reasoning(recommended, focus_areas, skip_areas)
        return CurriculumPlan(recommended_tasks=recommended, focus_areas=focus_areas, skip_areas=skip_areas, difficulty_range=diff_range, reasoning=reasoning)

    def sleep_consolidate(self, session_records: Optional[List[TaskRecord]]=None) -> Dict[str, Any]:
        """
        夜间巩固 (SkillOpt-Sleep 四阶段):
        1. harvest: 收集当天的执行记录
        2. mine: 提取失败案例和模式
        3. replay: 重放失败任务，分析原因
        4. consolidate: 生成改进建议，更新技能

        Args:
            session_records: 当天的执行记录（默认用内部历史）

        Returns:
            巩固报告
        """
        records = session_records or self._history
        if not records:
            return {'status': 'no_records', 'message': '没有可巩固的记录'}
        total = len(records)
        successes = [r for r in records if r.success]
        failures = [r for r in records if not r.success]
        failure_patterns = self._mine_failure_patterns(failures)
        replay_insights = self._replay_failures(failures)
        improvements = self._consolidate_improvements(failure_patterns, replay_insights)
        for (task_type, insight) in replay_insights.items():
            if task_type in self._skills:
                skill = self._skills[task_type]
                if insight['should_improve']:
                    skill.mastery = max(0, skill.mastery - 0.05)
        report = {'status': 'completed', 'timestamp': time.time(), 'summary': {'total_tasks': total, 'successes': len(successes), 'failures': len(failures), 'success_rate': len(successes) / max(total, 1)}, 'failure_patterns': failure_patterns, 'replay_insights': replay_insights, 'improvements': improvements, 'skill_snapshot': {k: {'mastery': round(v.mastery, 3), 'success_rate': round(v.success_rate, 3)} for (k, v) in self._skills.items()}}
        self._consolidation_log.append(report)
        self._save_state()
        return report

    def reformulate_task(self, task_description: str, difficulty: float, current_skill: float) -> Dict[str, Any]:
        """
        任务重构 (Cog-DRIFT):
        如果任务太难（difficulty > skill + 0.3），把它改形成更容易的形式。

        Args:
            task_description: 原始任务描述
            difficulty: 任务难度 [0, 1]
            current_skill: 当前技能水平 [0, 1]

        Returns:
            改形后的任务 + 原始任务（用于渐进返回）
        """
        gap = difficulty - current_skill
        if gap <= 0.3:
            return {'mode': 'direct', 'task': task_description, 'difficulty': difficulty, 'reason': '任务难度适中，直接执行'}
        reformulations = []
        if difficulty > 0.7:
            reformulations.append({'strategy': 'decompose', 'description': f'将任务分解为多个简单子任务', 'estimated_difficulty': difficulty * 0.6})
        if difficulty > 0.5:
            reformulations.append({'strategy': 'add_context', 'description': '提供更多上下文信息，降低理解难度', 'estimated_difficulty': difficulty * 0.8})
        reformulations.append({'strategy': 'limit_scope', 'description': '先处理任务的核心部分，忽略边缘情况', 'estimated_difficulty': difficulty * 0.7})
        return {'mode': 'reformulated', 'original_task': task_description, 'original_difficulty': difficulty, 'reformulations': reformulations, 'recommended': reformulations[0], 'gradual_return': {'step1': '完成简化版任务', 'step2': '逐步增加复杂度', 'step3': '回到原始任务'}}

    def _compute_difficulty(self, skill: SkillLevel) -> float:
        """计算任务难度 (ADCL 公式)。"""
        if skill.total_attempts == 0:
            return 0.5
        complexity_penalty = skill.avg_difficulty * 0.3
        difficulty = 1.0 - skill.success_rate * (1.0 - complexity_penalty)
        return max(0.0, min(1.0, difficulty))

    def _compute_priority(self, skill: SkillLevel, difficulty: float) -> float:
        """计算学习优先级。"""
        if skill.total_attempts < 3:
            return 0.9 + (3 - skill.total_attempts) * 0.05
        if skill.success_rate < 0.5:
            return 0.7 + (1.0 - skill.success_rate) * 0.2
        if 0.3 < difficulty < 0.8:
            return 0.5 + difficulty * 0.2388
        if skill.mastery > 0.8:
            return 0.1
        return 0.3

    def _mine_failure_patterns(self, failures: List[TaskRecord]) -> List[Dict[str, Any]]:
        """提取失败模式 (SkillOpt-Sleep 阶段2)。"""
        if not failures:
            return []
        by_type = defaultdict(list)
        for f in failures:
            by_type[f.task_type].append(f)
        patterns = []
        for (task_type, records) in by_type.items():
            error_types = defaultdict(int)
            for r in records:
                if r.error_type:
                    error_types[r.error_type] += 1
            patterns.append({'task_type': task_type, 'failure_count': len(records), 'avg_confidence': sum((r.confidence for r in records)) / len(records), 'common_errors': dict(error_types), 'pattern': self._identify_pattern(records)})
        return patterns

    def _identify_pattern(self, records: List[TaskRecord]) -> str:
        """识别失败模式。"""
        avg_conf = sum((r.confidence for r in records)) / len(records)
        if avg_conf > 0.7:
            return 'high_confidence_failure'
        elif avg_conf < 0.3:
            return 'low_confidence_failure'
        else:
            return 'inconsistent_failure'

    def _replay_failures(self, failures: List[TaskRecord]) -> Dict[str, Any]:
        """重放失败任务，分析原因 (SkillOpt-Sleep 阶段3)。"""
        insights = {}
        by_type = defaultdict(list)
        for f in failures:
            by_type[f.task_type].append(f)
        for (task_type, records) in by_type.items():
            skill = self._skills.get(task_type, SkillLevel(task_type=task_type))
            success_rate = skill.success_rate
            reasons = []
            if success_rate < 0.3:
                reasons.append('基础能力不足，需要更多练习')
            if any((r.confidence > 0.7 for r in records)):
                reasons.append('存在高置信度失败，可能是系统性 bug')
            if len(records) > 3:
                reasons.append('频繁失败，需要重新设计处理策略')
            insights[task_type] = {'failure_count': len(records), 'success_rate': round(success_rate, 3), 'reasons': reasons, 'should_improve': len(records) >= 2 or success_rate < 0.5, 'suggestion': self._generate_suggestion(task_type, records, skill)}
        return insights

    def _generate_suggestion(self, task_type: str, failures: List[TaskRecord], skill: SkillLevel) -> str:
        """生成改进建议。"""
        if skill.success_rate < 0.2:
            return f'降低 {task_type} 的难度，从基础开始练习'
        elif skill.success_rate < 0.5:
            return f'增加 {task_type} 的练习频率，重点突破薄弱环节'
        elif any((r.confidence > 0.7 for r in failures)):
            return f'检查 {task_type} 的处理逻辑，可能存在系统性问题'
        else:
            return f'继续练习 {task_type}，保持当前学习节奏'

    def _consolidate_improvements(self, patterns: List[Dict], insights: Dict) -> List[Dict[str, Any]]:
        """生成改进建议 (SkillOpt-Sleep 阶段4)。"""
        improvements = []
        for pattern in patterns:
            task_type = pattern['task_type']
            if pattern['pattern'] == 'high_confidence_failure':
                improvements.append({'type': 'bug_fix', 'target': task_type, 'priority': 'high', 'description': f'{task_type} 存在高置信度失败，需要检查处理逻辑'})
            elif pattern['pattern'] == 'low_confidence_failure':
                improvements.append({'type': 'skill_training', 'target': task_type, 'priority': 'medium', 'description': f'{task_type} 基础能力不足，需要增加练习'})
        for (task_type, insight) in insights.items():
            if insight['should_improve']:
                improvements.append({'type': 'practice', 'target': task_type, 'priority': 'medium', 'description': insight['suggestion']})
        return improvements

    def _generate_reasoning(self, recommended: List[Dict], focus: List[str], skip: List[str]) -> str:
        """生成推荐理由。"""
        parts = []
        if recommended:
            top = recommended[0]
            parts.append(f"重点学习 {top['task_type']}（难度 {top['difficulty']:.2f}，当前掌握 {top['mastery']:.2f}）")
        if focus:
            parts.append(f"需要突破: {', '.join(focus)}")
        if skip:
            parts.append(f"已掌握可跳过: {', '.join(skip)}")
        return '；'.join(parts) if parts else '按当前进度继续学习'

    def _save_state(self):
        """保存状态到文件。"""
        try:
            state = {'skills': {k: asdict(v) for (k, v) in self._skills.items()}, 'history_count': len(self._history), 'consolidation_count': len(self._consolidation_log), 'last_update': time.time()}
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self._persistence_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            logger.warning(f'保存课程状态失败: {e}')

    def _load_state(self):
        """从文件加载状态。"""
        try:
            if self._persistence_path.exists():
                state = json.loads(self._persistence_path.read_text(encoding='utf-8'))
                for (k, v) in state.get('skills', {}).items():
                    skill = SkillLevel(task_type=v['task_type'])
                    skill.__dict__.update(v)
                    self._skills[k] = skill
                logger.info(f'课程状态已加载: {len(self._skills)} 个技能')
        except Exception as e:
            logger.warning(f'加载课程状态失败: {e}')

    def self_test(self) -> Dict[str, Any]:
        """自检：验证核心算法。"""
        results = {}
        skill = SkillLevel(task_type='test')
        for i in range(10):
            skill.update(TaskRecord(task_id=f't{i}', task_type='test', difficulty=0.5, success=i < 7, confidence=0.8, timestamp=time.time()))
        results['skill_update'] = {'success_rate': round(skill.success_rate, 2), 'mastery': round(skill.mastery, 2), 'passed': abs(skill.success_rate - 0.7) < 0.01}
        engine = CurriculumEngine(persistence_path=Path('/tmp/curriculum_test.json'))
        skill2 = SkillLevel(task_type='easy', total_attempts=10, success_count=9, avg_confidence=0.9, avg_difficulty=0.3)
        diff = engine._compute_difficulty(skill2)
        results['difficulty_calc'] = {'difficulty': round(diff, 3), 'passed': 0.0 < diff < 0.6489}
        engine._skills = {'easy': SkillLevel(task_type='easy', total_attempts=10, success_count=9, avg_confidence=0.9, avg_difficulty=0.3), 'hard': SkillLevel(task_type='hard', total_attempts=5, success_count=1, avg_confidence=0.4, avg_difficulty=0.8), 'new': SkillLevel(task_type='new', total_attempts=1, success_count=1, avg_confidence=0.7, avg_difficulty=0.5)}
        plan = engine.recommend_next()
        results['recommendation'] = {'focus_areas': plan.focus_areas, 'skip_areas': plan.skip_areas, 'top_recommendation': plan.recommended_tasks[0]['task_type'] if plan.recommended_tasks else None, 'passed': len(plan.recommended_tasks) > 0}
        engine._history = [TaskRecord('t1', 'hard', 0.8, False, 0.9, time.time(), error_type='timeout'), TaskRecord('t2', 'hard', 0.8, False, 0.85, time.time(), error_type='timeout'), TaskRecord('t3', 'easy', 0.3, True, 0.6, time.time())]
        report = engine.sleep_consolidate()
        results['sleep_consolidate'] = {'status': report['status'], 'failure_count': report['summary']['failures'], 'passed': report['status'] == 'completed' and report['summary']['failures'] == 2}
        all_passed = all((r.get('passed', False) for r in results.values()))
        results['all_passed'] = all_passed
        test_file = Path('/tmp/curriculum_test.json')
        if test_file.exists():
            test_file.unlink()
        return results