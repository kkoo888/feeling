"""
策略执行器 — 进化系统的"观察者"角色

核心改变：进化系统不生成回复，只观察和评估。

架构：
  LAAP（外接大脑）→ 提供认知状态（不需要 LLM）
  进化系统 → 提供策略建议（不需要 LLM）
  小茜（Agent）→ 用自己的 LLM 生成回复
  进化系统 → 观察回复，评估质量，更新策略树

进化系统 = 教练，不是球员。
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import os

import requests

logger = logging.getLogger("evolution.executor")

# LAAP API 基址：优先读 env，默认 :11530（与项目其他模块统一）
LAAP_API_BASE = os.environ.get("LAAP_API_BASE", "http://localhost:11530")


@dataclass
class CognitiveState:
    """LAAP 认知状态（外接大脑提供的数据）"""
    needs: Dict[str, float] = field(default_factory=dict)
    valence: float = 0.0
    arousal: float = 0.0
    attention_focus: str = ""
    cognitive_cycle: int = 0
    energy: float = 10.0
    raw: Dict = field(default_factory=dict)


@dataclass
class StrategyAdvice:
    """进化系统给出的策略建议"""
    strategy: str = ""           # 策略文本
    source: str = ""             # 来源（draft/improve/debug/explore）
    confidence: float = 0.0      # 置信度
    cognitive_state: CognitiveState = None  # 当前认知状态
    enhanced_prompt: str = ""    # 增强后的 system prompt


@dataclass
class ConversationRecord:
    """对话记录（进化系统观察到的数据）"""
    strategy: str = ""           # 使用的策略
    user_input: str = ""         # 用户输入
    response: str = ""           # 实际回复
    success: bool = False        # 是否成功
    satisfaction: float = 0.0    # 满意度
    response_time: float = 0.0   # 响应时间
    cognitive_state: Dict = None # 执行时的认知状态
    timestamp: float = 0.0       # 时间戳


class CognitiveStateProvider:
    """
    认知状态提供器 — 从 LAAP 获取认知状态
    
    这是 LAAP 作为"外接大脑"的核心接口。
    LAAP 不需要 LLM，只提供认知数据。
    """
    
    def __init__(self, api_base: str = None):
        self.api_base = api_base or LAAP_API_BASE
    
    def get_state(self, user_input: str = "") -> CognitiveState:
        """
        从 LAAP 获取认知状态
        
        Args:
            user_input: 用户输入（用于上下文感知）
        
        Returns:
            CognitiveState: 认知状态数据
        """
        try:
            resp = requests.post(
                f"{self.api_base}/v1/cognitive_state",
                json={"input": user_input},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            state = data.get("state", {})
            
            return CognitiveState(
                needs=state.get("needs", {}),
                valence=state.get("valence", 0),
                arousal=state.get("arousal", 0),
                attention_focus=state.get("attention_focus", ""),
                cognitive_cycle=state.get("cognitive_cycle", 0),
                energy=state.get("energy", 10),
                raw=data,
            )
        except Exception as e:
            logger.warning(f"获取认知状态失败: {e}")
            return CognitiveState()
    
    def reflect(self, response: str, success: bool = False, 
                connection: bool = False) -> bool:
        """
        向 LAAP 反思注入
        
        Args:
            response: 实际回复
            success: 是否成功
            connection: 是否加强连接
        
        Returns:
            是否成功
        """
        try:
            resp = requests.post(
                f"{self.api_base}/v1/reflect",
                json={
                    "output": response,
                    "feedback": {"success": success, "connection": connection},
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("updated", False)
        except Exception as e:
            logger.warning(f"反思注入失败: {e}")
            return False


class StrategyAdvisor:
    """
    策略顾问 — 从进化树中推荐最优策略
    
    进化系统的核心价值：
    - 不生成回复
    - 只推荐策略
    - 基于历史表现选择最优
    """
    
    def __init__(self, journal=None):
        self.journal = journal
    
    def get_advice(self, user_input: str = "", 
                   cognitive_state: CognitiveState = None) -> StrategyAdvice:
        """
        从进化树中推荐最优策略
        
        Args:
            user_input: 用户输入
            cognitive_state: 当前认知状态
        
        Returns:
            StrategyAdvice: 策略建议
        """
        if not self.journal or len(self.journal.nodes) == 0:
            return StrategyAdvice(
                strategy="保持温和友好的语气回复，简短而温暖",
                source="default",
                confidence=0.5,
                cognitive_state=cognitive_state,
            )
        
        # 获取最优策略
        best = self.journal.get_best_node()
        if best:
            return StrategyAdvice(
                strategy=best.strategy,
                source=f"evolution_v{best.step}",
                confidence=best.multi_metric.composite_score if best.multi_metric else 0.5,
                cognitive_state=cognitive_state,
            )
        
        return StrategyAdvice(
            strategy="保持温和友好的语气回复",
            source="fallback",
            confidence=0.3,
            cognitive_state=cognitive_state,
        )
    
    def build_enhanced_prompt(self, advice: StrategyAdvice) -> str:
        """
        构建增强后的 system prompt
        
        将策略建议 + 认知状态组合成 system prompt，
        供 Agent 的 LLM 使用。
        
        Args:
            advice: 策略建议
        
        Returns:
            增强后的 system prompt
        """
        parts = []
        
        # 策略注入
        parts.append(f"## 行为策略")
        parts.append(f"策略: {advice.strategy}")
        parts.append(f"来源: {advice.source}")
        parts.append(f"置信度: {advice.confidence:.2f}")
        
        # 认知状态注入
        if advice.cognitive_state:
            cs = advice.cognitive_state
            parts.append(f"")
            parts.append(f"## 认知状态")
            parts.append(f"需求: {cs.needs}")
            parts.append(f"效价: {cs.valence:.2f} (越正越积极)")
            parts.append(f"唤醒度: {cs.arousal:.2f} (越高越活跃)")
            parts.append(f"注意力: {cs.attention_focus}")
            parts.append(f"能量: {cs.energy:.1f}")
        
        return "\n".join(parts)


class EvolutionObserver:
    """
    进化观察器 — 观察对话结果并评估
    
    这是进化系统的核心：
    - 观察实际对话
    - 评估回复质量
    - 更新策略树
    
    对应 AIDE 的 parse_exec_result()
    """
    
    def __init__(self, journal=None, api_base: str = None):
        self.journal = journal
        self.cognitive_provider = CognitiveStateProvider(api_base)
        self._history: List[ConversationRecord] = []
    
    def observe(
        self,
        strategy: str,
        user_input: str,
        response: str,
        success: bool = True,
        satisfaction: float = 0.7,
        response_time: float = 0.0,
    ) -> ConversationRecord:
        """
        观察一次对话 — 对应 AIDE 的 "执行代码并收集输出"
        
        Args:
            strategy: 使用的策略
            user_input: 用户输入
            response: 实际回复
            success: 是否成功
            satisfaction: 满意度 [0, 1]
            response_time: 响应时间
        
        Returns:
            ConversationRecord: 对话记录
        """
        # 获取当前认知状态
        cognitive_state = self.cognitive_provider.get_state(user_input)
        
        # 创建对话记录
        record = ConversationRecord(
            strategy=strategy,
            user_input=user_input,
            response=response,
            success=success,
            satisfaction=satisfaction,
            response_time=response_time,
            cognitive_state=cognitive_state.raw if cognitive_state else {},
            timestamp=time.time(),
        )
        
        self._history.append(record)
        
        # 反思注入到 LAAP
        self.cognitive_provider.reflect(
            response=response,
            success=success,
            connection=satisfaction > 0.7,
        )
        
        return record
    
    def evaluate(self, record: ConversationRecord) -> dict:
        """
        评估一次对话 — 对应 AIDE 的 parse_exec_result()
        
        关键：基于实际回复评估，不是凭空猜。
        
        Args:
            record: 对话记录
        
        Returns:
            评估结果 {is_bug, summary, metric, lower_is_better}
        """
        # 基于实际回复的评估（不需要 LLM）
        response = record.response
        
        # 检查是否有 bug
        is_bug = False
        bug_reasons = []
        
        if not response:
            is_bug = True
            bug_reasons.append("无回复")
        elif "fallback" in response.lower():
            is_bug = True
            bug_reasons.append("fallback 响应")
        elif len(response) < 5:
            is_bug = True
            bug_reasons.append("回复过短")
        elif "error" in response.lower():
            is_bug = True
            bug_reasons.append("包含错误信息")
        
        # 计算质量分数
        if is_bug:
            metric = None
        else:
            # 多维度评估
            length_score = min(1.0, len(response) / 200)  # 长度适中
            diversity_score = min(1.0, len(set(response)) / 50)  # 用词多样
            satisfaction_score = record.satisfaction  # 用户满意度
            
            metric = round(
                0.4 * satisfaction_score +
                0.3 * length_score +
                0.3 * diversity_score,
                3
            )
        
        summary = f"回复长度={len(response)}, 满意度={record.satisfaction:.2f}"
        if bug_reasons:
            summary += f", 问题={', '.join(bug_reasons)}"
        
        return {
            "is_bug": is_bug,
            "summary": summary,
            "metric": metric,
            "lower_is_better": False,
        }
    
    def get_history(self) -> List[ConversationRecord]:
        """获取观察历史"""
        return self._history
