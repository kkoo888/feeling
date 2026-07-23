"""
MetaLearningEngine — 元学习引擎
================================

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

设计目标:
  - 输入: Agent 的任务执行记录
  - 输出: 自知之明（会什么/不会什么）+ 策略推荐 + 学习方向

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

logger = logging.getLogger("laap.agi.meta_learning")

# 持久化路径
_STATE_DIR = Path(__file__).resolve().parent.parent.parent / "aris_brain" / "state"
_META_STATE = _STATE_DIR / "meta_learning_state.json"


# ═══════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════

@dataclass
class TaskExecution:
    """一次任务执行记录"""
    task_id: str
    task_type: str              # 意图类型
    strategy_used: str          # 使用的策略 (rule/llm/hybrid)
    success: bool
    confidence: float           # 执行时的置信度
    actual_accuracy: float      # 实际准确率 (如果可测量)
    timestamp: float
    duration_ms: float = 0.0
    error_info: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaKnowledge:
    """元认知知识：知道自己会什么、不会什么"""
    skill_inventory: Dict[str, float] = field(default_factory=dict)  # 技能→掌握度
    blind_spots: List[str] = field(default_factory=list)             # 盲点列表
    strengths: List[str] = field(default_factory=list)               # 强项列表
    strategy_effectiveness: Dict[str, float] = field(default_factory=dict)  # 策略→有效性


@dataclass
class MetaPlan:
    """元认知规划：知道该用什么策略"""
    recommended_strategy: str           # 推荐的策略
    strategy_reason: str                # 推荐理由
    fallback_strategy: str              # 备选策略
    confidence_threshold: float         # 置信度阈值
    routing_rules: Dict[str, str] = field(default_factory=dict)  # 任务类型→策略


@dataclass
class MetaEvaluation:
    """元认知评估：知道学得好不好"""
    overall_score: float                # 总体评分 [0, 1]
    strategy_effective: bool            # 策略是否有效
    learning_progress: float            # 学习进度 [0, 1]
    improvement_suggestions: List[str] = field(default_factory=list)
    reflection: str = ""                # 反思总结


@dataclass
class MetacognitionState:
    """完整的元认知状态"""
    knowledge: MetaKnowledge = field(default_factory=MetaKnowledge)
    plan: MetaPlan = field(default_factory=lambda: MetaPlan(
        recommended_strategy="rule", strategy_reason="默认规则策略",
        fallback_strategy="llm", confidence_threshold=0.5,
    ))
    evaluation: MetaEvaluation = field(default_factory=lambda: MetaEvaluation(
        overall_score=0.5, strategy_effective=False,
        learning_progress=0.0, improvement_suggestions=[],
    ))
    last_update: float = 0.0


# ═══════════════════════════════════════════════════════
# 策略库
# ═══════════════════════════════════════════════════════

STRATEGY_PROFILES = {
    "rule": {
        "name": "规则匹配",
        "description": "关键词+正则，速度最快",
        "strengths": ["简单意图", "结构化输入", "低延迟"],
        "weaknesses": ["复杂语义", "模糊表达", "新意图"],
        "base_reliability": 0.7,
    },
    "llm": {
        "name": "LLM 推理",
        "description": "大模型理解，准确但慢",
        "strengths": ["复杂语义", "模糊表达", "新意图"],
        "weaknesses": ["延迟高", "成本高", "幻觉"],
        "base_reliability": 0.85,
    },
    "hybrid": {
        "name": "混合策略",
        "description": "规则优先，低置信度时用 LLM 兜底",
        "strengths": ["平衡速度和准确率"],
        "weaknesses": ["阈值敏感"],
        "base_reliability": 0.8,
    },
    "memory": {
        "name": "记忆复用",
        "description": "从历史成功案例中找相似",
        "strengths": ["重复任务", "个性化"],
        "weaknesses": ["新任务", "冷启动"],
        "base_reliability": 0.75,
    },
}


# ═══════════════════════════════════════════════════════
# 核心引擎
# ═══════════════════════════════════════════════════════

class MetaLearningEngine:
    """
    元学习引擎 — "学会学习"。

    三层元认知 (ICML 2025):
      1. 元认知知识: 知道自己会什么、不会什么
      2. 元认知规划: 知道该用什么策略
      3. 元认知评估: 知道学得好不好

    工具元学习 (MetaAgent, arXiv 2025):
      - 遇到知识盲点时，自动生成求助请求
      - 从工具使用结果中学习
      - 不断更新策略有效性

    技能文档优化 (SkillOpt, Microsoft 2026):
      - 把技能文档当"权重"
      - 用反馈当"梯度"
      - 不动模型参数，持续优化行为
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self._state = MetacognitionState()
        self._history: List[TaskExecution] = []
        self._persistence_path = persistence_path or _META_STATE
        self._load_state()

    # ── 公开接口 ──────────────────────────────────────

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

    def recommend_strategy(self, task_type: str, input_complexity: float = 0.5
                           ) -> Tuple[str, float, str]:
        """
        推荐处理策略。

        Args:
            task_type: 任务类型
            input_complexity: 输入复杂度 [0, 1]

        Returns:
            (strategy, confidence, reason)
        """
        knowledge = self._state.knowledge

        # 查看该任务类型的历史表现
        skill_level = knowledge.skill_inventory.get(task_type, 0.5)
        strategy_scores = {}

        for strategy_name, profile in STRATEGY_PROFILES.items():
            # 基础可靠性
            score = profile["base_reliability"]

            # 根据任务类型调整
            if task_type in profile["strengths"]:
                score += 0.1
            if task_type in profile["weaknesses"]:
                score -= 0.1

            # 根据历史有效性调整
            historical = knowledge.strategy_effectiveness.get(strategy_name, 0.5)
            score = 0.6 * score + 0.4 * historical

            # 根据输入复杂度调整
            if strategy_name == "rule" and input_complexity > 0.7:
                score -= 0.2  # 复杂输入不适合纯规则
            if strategy_name == "llm" and input_complexity < 0.3:
                score -= 0.1  # 简单输入不需要 LLM

            strategy_scores[strategy_name] = max(0.0, min(1.0, score))

        # 选择最佳策略
        best = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best]

        # 生成理由
        profile = STRATEGY_PROFILES[best]
        reason = f"{profile['name']}：{profile['description']}"

        # 如果有盲点，提示
        if task_type in knowledge.blind_spots:
            reason += f"（注意：{task_type} 是已知盲点，建议增加验证）"

        return best, best_score, reason

    def reflect_on_failure(self, execution: TaskExecution) -> Dict[str, Any]:
        """
        失败反思 (MetaAgent 核心机制)。

        遇到失败时，分析原因并生成改进建议。

        Returns:
            反思报告
        """
        knowledge = self._state.knowledge

        # 分析失败原因
        reasons = []
        suggestions = []

        # 原因1: 策略选择不当
        strategy_eff = knowledge.strategy_effectiveness.get(execution.strategy_used, 0.5)
        if strategy_eff < 0.5:
            reasons.append(f"策略 {execution.strategy_used} 有效性低 ({strategy_eff:.2f})")
            suggestions.append(f"尝试其他策略")

        # 原因2: 盲点
        if execution.task_type in knowledge.blind_spots:
            reasons.append(f"{execution.task_type} 是已知盲点")
            suggestions.append(f"增加 {execution.task_type} 的练习或寻求外部帮助")

        # 原因3: 置信度过高但失败
        if execution.confidence > 0.7:
            reasons.append("高置信度但失败，可能存在系统性问题")
            suggestions.append("检查处理逻辑，可能存在 bug")

        # 原因4: 置信度过低
        if execution.confidence < 0.3:
            reasons.append("低置信度，能力不足")
            suggestions.append("从基础开始练习")

        # 原因5: 新任务类型
        if execution.task_type not in knowledge.skill_inventory:
            reasons.append("首次遇到该类型任务")
            suggestions.append("记录到技能清单，开始积累经验")

        return {
            "task_type": execution.task_type,
            "strategy": execution.strategy_used,
            "confidence": execution.confidence,
            "reasons": reasons,
            "suggestions": suggestions,
            "action": "记录失败模式，更新策略有效性",
        }

    # ── 内部方法 ──────────────────────────────────────

    def _update_knowledge(self, execution: TaskExecution):
        """更新元认知知识。"""
        knowledge = self._state.knowledge
        task_type = execution.task_type

        # 更新技能清单 (EMA)
        alpha = 0.3
        current = knowledge.skill_inventory.get(task_type, 0.5)
        if execution.success:
            new_skill = alpha * execution.actual_accuracy + (1 - alpha) * current
        else:
            new_skill = alpha * 0.0 + (1 - alpha) * current
        knowledge.skill_inventory[task_type] = round(new_skill, 4)

        # 更新策略有效性
        strategy = execution.strategy_used
        current_eff = knowledge.strategy_effectiveness.get(strategy, 0.5)
        if execution.success:
            new_eff = alpha * 1.0 + (1 - alpha) * current_eff
        else:
            new_eff = alpha * 0.0 + (1 - alpha) * current_eff
        knowledge.strategy_effectiveness[strategy] = round(new_eff, 4)

        # 更新盲点和强项
        knowledge.blind_spots = [
            k for k, v in knowledge.skill_inventory.items()
            if v < 0.3 and self._get_attempt_count(k) >= 3
        ]
        knowledge.strengths = [
            k for k, v in knowledge.skill_inventory.items()
            if v > 0.8 and self._get_attempt_count(k) >= 3
        ]

    def _update_plan(self):
        """更新元认知规划。"""
        knowledge = self._state.knowledge
        plan = self._state.plan

        # 根据策略有效性更新推荐
        if knowledge.strategy_effectiveness:
            best_strategy = max(
                knowledge.strategy_effectiveness,
                key=knowledge.strategy_effectiveness.get,
            )
            best_eff = knowledge.strategy_effectiveness[best_strategy]
            plan.recommended_strategy = best_strategy
            plan.strategy_reason = (
                f"{STRATEGY_PROFILES.get(best_strategy, {}).get('name', best_strategy)}："
                f"历史有效性 {best_eff:.2f}"
            )

        # 根据技能水平更新路由规则
        for task_type, skill in knowledge.skill_inventory.items():
            if skill > 0.8:
                plan.routing_rules[task_type] = "rule"  # 熟练的用规则
            elif skill < 0.3:
                plan.routing_rules[task_type] = "llm"   # 不熟的用 LLM
            else:
                plan.routing_rules[task_type] = "hybrid" # 中间的用混合

    def _update_evaluation(self):
        """更新元认知评估。"""
        knowledge = self._state.knowledge
        evaluation = self._state.evaluation

        if not knowledge.skill_inventory:
            return

        # 总体评分 = 所有技能的加权平均
        skills = list(knowledge.skill_inventory.values())
        evaluation.overall_score = round(sum(skills) / len(skills), 4)

        # 策略是否有效
        if knowledge.strategy_effectiveness:
            avg_eff = sum(knowledge.strategy_effectiveness.values()) / len(knowledge.strategy_effectiveness)
            evaluation.strategy_effective = avg_eff > 0.6

        # 学习进度 = 技能提升速度
        if len(self._history) >= 2:
            recent = self._history[-10:]
            older = self._history[-20:-10] if len(self._history) >= 20 else self._history[:10]
            recent_success = sum(1 for r in recent if r.success) / len(recent)
            older_success = sum(1 for r in older if r.success) / len(older)
            evaluation.learning_progress = round(max(0, recent_success - older_success), 4)

        # 改进建议
        suggestions = []
        for blind in knowledge.blind_spots[:3]:
            suggestions.append(f"突破盲点: {blind}")
        for task_type, skill in knowledge.skill_inventory.items():
            if 0.3 < skill < 0.6:
                suggestions.append(f"提升 {task_type} (当前 {skill:.2f})")
        evaluation.improvement_suggestions = suggestions[:5]

        # 反思
        if evaluation.overall_score > 0.8:
            evaluation.reflection = "整体表现优秀，可以挑战更高难度"
        elif evaluation.overall_score > 0.5:
            evaluation.reflection = "表现中等，重点突破薄弱环节"
        else:
            evaluation.reflection = "需要加强基础练习，从简单任务开始"

    def _get_attempt_count(self, task_type: str) -> int:
        """获取某类任务的尝试次数。"""
        return sum(1 for e in self._history if e.task_type == task_type)

    # ── 持久化 ──────────────────────────────────────

    def _save_state(self):
        """保存状态。"""
        try:
            state = {
                "knowledge": asdict(self._state.knowledge),
                "plan": asdict(self._state.plan),
                "evaluation": asdict(self._state.evaluation),
                "history_count": len(self._history),
                "last_update": self._state.last_update,
            }
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self._persistence_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"保存元学习状态失败: {e}")

    def _load_state(self):
        """加载状态。"""
        try:
            if self._persistence_path.exists():
                data = json.loads(self._persistence_path.read_text(encoding="utf-8"))
                if "knowledge" in data:
                    self._state.knowledge = MetaKnowledge(**data["knowledge"])
                if "plan" in data:
                    self._state.plan = MetaPlan(**data["plan"])
                if "evaluation" in data:
                    self._state.evaluation = MetaEvaluation(**data["evaluation"])
                self._state.last_update = data.get("last_update", 0)
                logger.info(f"元学习状态已加载")
        except Exception as e:
            logger.warning(f"加载元学习状态失败: {e}")

    # ── 自检 ──────────────────────────────────────

    def self_test(self) -> Dict[str, Any]:
        """自检：验证核心功能。"""
        results = {}

        engine = MetaLearningEngine(persistence_path=Path("/tmp/meta_test.json"))

        # 测试1: 记录执行
        for i in range(10):
            engine.record_execution(TaskExecution(
                task_id=f"t{i}", task_type="read_file",
                strategy_used="rule", success=i < 8,
                confidence=0.8, actual_accuracy=0.9 if i < 8 else 0.0,
                timestamp=time.time(),
            ))
        results["record_execution"] = {
            "history_count": len(engine._history),
            "passed": len(engine._history) == 10,
        }

        # 测试2: 元认知知识
        knowledge = engine.get_knowledge()
        results["knowledge"] = {
            "skill_count": len(knowledge.skill_inventory),
            "read_file_skill": round(knowledge.skill_inventory.get("read_file", 0), 2),
            "passed": "read_file" in knowledge.skill_inventory,
        }

        # 测试3: 策略推荐
        strategy, score, reason = engine.recommend_strategy("read_file", 0.3)
        results["recommend_strategy"] = {
            "strategy": strategy,
            "score": round(score, 2),
            "passed": strategy in STRATEGY_PROFILES,
        }

        # 测试4: 失败反思
        failure = TaskExecution(
            task_id="fail1", task_type="debug", strategy_used="rule",
            success=False, confidence=0.9, actual_accuracy=0.0,
            timestamp=time.time(), error_info="wrong intent",
        )
        reflection = engine.reflect_on_failure(failure)
        results["reflect_on_failure"] = {
            "reasons_count": len(reflection["reasons"]),
            "suggestions_count": len(reflection["suggestions"]),
            "passed": len(reflection["reasons"]) > 0,
        }

        # 测试5: 元认知评估
        evaluation = engine.get_evaluation()
        results["evaluation"] = {
            "overall_score": evaluation.overall_score,
            "passed": 0 <= evaluation.overall_score <= 1,
        }

        # 测试6: 多任务类型
        for task_type in ["search", "generate", "chat", "translate"]:
            engine.record_execution(TaskExecution(
                task_id=f"multi_{task_type}", task_type=task_type,
                strategy_used="hybrid", success=True,
                confidence=0.7, actual_accuracy=0.8,
                timestamp=time.time(),
            ))
        knowledge2 = engine.get_knowledge()
        results["multi_task"] = {
            "skill_types": len(knowledge2.skill_inventory),
            "passed": len(knowledge2.skill_inventory) >= 5,
        }

        all_passed = all(r.get("passed", False) for r in results.values())
        results["all_passed"] = all_passed

        # 清理
        test_file = Path("/tmp/meta_test.json")
        if test_file.exists():
            test_file.unlink()

        return results
