"""
代码进化引擎 — AIDE 原始能力的恢复

在代码空间中搜索最优代码，同时保留策略进化能力。
两个进化系统可以同时运行。
"""

import logging
import random
from typing import Optional

from .agent import EvolutionAgent
from .journal import EvolutionNode, EvolutionJournal
from .runner import ExecutionResult
from .utils.metric import MetricValue, MultiDimensionalMetric

logger = logging.getLogger("evolution.code")


class CodeEvolutionAgent(EvolutionAgent):
    """
    代码进化 Agent — AIDE 原始能力
    
    在代码空间中搜索最优代码：
    - _draft(): 生成初始代码方案
    - _improve(): 改进现有代码
    - _debug(): 修复代码 bug
    - _explore(): 探索全新代码方案
    
    与策略进化 Agent 共享同一套树搜索逻辑，
    但 prompt 和评估方式针对代码优化。
    """

    def __init__(
        self,
        task_desc: str,
        journal: EvolutionJournal,
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        num_drafts: int = 3,
        debug_prob: float = 0.2,
        explore_prob: float = 0.15,
        max_debug_depth: int = 3,
        eval_metric: str = "accuracy",
        eval_direction: str = "maximize",
    ):
        """
        初始化代码进化 Agent。
        
        Args:
            task_desc: 任务描述（如 "预测房价"、"分类图片"）
            journal: 进化日志
            model: LLM 模型
            temperature: 采样温度
            num_drafts: 初始草案数量
            debug_prob: 调试概率
            explore_prob: 探索概率
            max_debug_depth: 最大调试深度
            eval_metric: 评估指标名称（如 "accuracy"、"RMSE"）
            eval_direction: 评估方向（"maximize" 或 "minimize"）
        """
        super().__init__(
            task_desc=task_desc,
            journal=journal,
            model=model,
            temperature=temperature,
            num_drafts=num_drafts,
            debug_prob=debug_prob,
            explore_prob=explore_prob,
            max_debug_depth=max_debug_depth,
        )
        self.eval_metric = eval_metric
        self.eval_direction = eval_direction

    def _draft(self) -> EvolutionNode:
        """生成初始代码方案"""
        prompt = {
            "Introduction": (
                "You are a senior software engineer. "
                "Write a complete, working Python program that solves the given task. "
                "The code should be self-contained and executable."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {
                "Code guidelines": [
                    "Write a single-file Python program that is self-contained.",
                    "The code should implement the solution and print the evaluation metric.",
                    "Include all necessary imports.",
                    "Handle edge cases and errors gracefully.",
                    f"Optimize for {self.eval_metric}.",
                    "Add comments explaining key logic.",
                ],
                "Response format": (
                    "First write a brief plan (2-3 sentences), then provide the code in a single code block."
                ),
            },
        }
        plan, code = self._generate_from_prompt(prompt)
        return EvolutionNode(strategy=code, plan=plan)

    def _improve(self, parent: EvolutionNode) -> EvolutionNode:
        """改进现有代码"""
        prompt = {
            "Introduction": (
                "You are a senior software engineer. "
                "Improve the given code to achieve better performance. "
                "Make a single, specific, atomic improvement."
            ),
            "Task description": self.task_desc,
            "Current code": parent.strategy,
            "Current performance": f"{self.eval_metric}: {parent.metric.value if parent.metric else 'unknown'}",
            "Memory": self.journal.generate_summary(),
            "Instructions": {
                "Improvement guidelines": [
                    "Make ONE specific improvement (not multiple changes).",
                    "The improvement should be atomic and testable.",
                    f"Focus on improving {self.eval_metric}.",
                    "Explain why this improvement should help.",
                ],
                "Response format": (
                    "First describe the improvement (2-3 sentences), then provide the improved code."
                ),
            },
        }
        plan, code = self._generate_from_prompt(prompt)
        return EvolutionNode(strategy=code, plan=plan, parent=parent)

    def _debug(self, parent: EvolutionNode) -> EvolutionNode:
        """修复代码 bug"""
        prompt = {
            "Introduction": (
                "You are a senior software engineer. "
                "The code has a bug. Fix it based on the error output."
            ),
            "Task description": self.task_desc,
            "Buggy code": parent.strategy,
            "Error output": parent.term_out,
            "Instructions": {
                "Debug guidelines": [
                    "Identify the root cause of the error.",
                    "Fix the bug with minimal changes.",
                    "Ensure the fix doesn't introduce new bugs.",
                ],
                "Response format": (
                    "First explain the bug and fix (2-3 sentences), then provide the fixed code."
                ),
            },
        }
        plan, code = self._generate_from_prompt(prompt)
        return EvolutionNode(strategy=code, plan=plan, parent=parent)

    def _generate_from_prompt(self, prompt: dict) -> tuple:
        """用 LLM 生成代码"""
        from .backend import query
        from .utils.response import extract_code, extract_text_up_to_code

        # 将 prompt 转为字符串
        prompt_str = self._format_prompt(prompt)

        for _ in range(3):
            try:
                response = query(
                    system_message=prompt_str,
                    user_message=None,
                    model=self.model,
                    temperature=self.temperature,
                )
                code = extract_code(response)
                plan = extract_text_up_to_code(response)
                if code and plan:
                    return plan, code
            except Exception as e:
                logger.warning(f"代码生成失败: {e}")

        return "生成失败", ""

    def _format_prompt(self, prompt: dict) -> str:
        """将 prompt dict 格式化为字符串"""
        parts = []
        for key, value in prompt.items():
            if isinstance(value, dict):
                parts.append(f"## {key}")
                for k, v in value.items():
                    if isinstance(v, list):
                        parts.append(f"### {k}")
                        for item in v:
                            parts.append(f"- {item}")
                    else:
                        parts.append(f"**{k}**: {v}")
            elif isinstance(value, list):
                parts.append(f"## {key}")
                for item in value:
                    parts.append(f"- {item}")
            else:
                parts.append(f"## {key}\n{value}")
        return "\n\n".join(parts)


class DualEvolutionEngine:
    """
    双进化引擎 — 同时运行策略进化和代码进化
    
    两个独立的进化树，共享同一个 LLM 后端，
    但各自独立搜索、独立评估。
    """
    
    def __init__(
        self,
        strategy_task: str = "优化小茜的对话策略",
        code_task: str = "优化代码生成质量",
        model: str = "gpt-4-turbo",
    ):
        # 策略进化
        self.strategy_journal = EvolutionJournal()
        self.strategy_agent = EvolutionAgent(
            task_desc=strategy_task,
            journal=self.strategy_journal,
            model=model,
        )
        
        # 代码进化
        self.code_journal = EvolutionJournal()
        self.code_agent = CodeEvolutionAgent(
            task_desc=code_task,
            journal=self.code_journal,
            model=model,
            eval_metric="quality",
            eval_direction="maximize",
        )
    
    def strategy_step(self, user_input: str, response: str, 
                      success: bool, satisfaction: float) -> dict:
        """执行一步策略进化"""
        # 计算指标
        multi = MultiDimensionalMetric(
            task_success=1.0 if success else 0.3,
            user_satisfaction=satisfaction,
            efficiency=max(0.1, 1.0 - len(response) / 1000),
            safety=1.0,
            creativity=min(1.0, len(set(response)) / 50),
        )
        
        parent = self.strategy_journal.get_best_node()
        node = EvolutionNode(
            strategy=response,
            plan=f"策略进化 v{len(self.strategy_journal)}",
            parent=parent,
        )
        node.metric = MetricValue(multi.composite_score, maximize=True)
        node.multi_metric = multi
        node.is_buggy = not success
        self.strategy_journal.append(node)
        
        best = self.strategy_journal.get_best_node()
        return {
            "type": "strategy",
            "step": len(self.strategy_journal),
            "current_score": multi.composite_score,
            "best_score": best.multi_metric.composite_score if best and best.multi_metric else 0,
            "best_strategy": best.strategy if best else "",
        }
    
    def code_step(self, code: str, success: bool, metric_value: float) -> dict:
        """执行一步代码进化"""
        multi = MultiDimensionalMetric(
            task_success=1.0 if success else 0.3,
            user_satisfaction=0.7,
            efficiency=0.8,
            safety=1.0,
            creativity=0.5,
        )
        
        parent = self.code_journal.get_best_node()
        node = EvolutionNode(
            strategy=code,
            plan=f"代码进化 v{len(self.code_journal)}",
            parent=parent,
        )
        node.metric = MetricValue(metric_value, maximize=True)
        node.multi_metric = multi
        node.is_buggy = not success
        self.code_journal.append(node)
        
        best = self.code_journal.get_best_node()
        return {
            "type": "code",
            "step": len(self.code_journal),
            "current_metric": metric_value,
            "best_metric": best.metric.value if best and best.metric else 0,
            "best_code_length": len(best.strategy) if best else 0,
        }
    
    def get_report(self) -> str:
        """获取双进化报告"""
        lines = []
        lines.append("=" * 55)
        lines.append("  双进化引擎状态")
        lines.append("=" * 55)
        
        # 策略进化
        best_s = self.strategy_journal.get_best_node()
        lines.append(f"\n  📊 策略进化:")
        lines.append(f"    节点数: {len(self.strategy_journal)}")
        if best_s and best_s.multi_metric:
            lines.append(f"    最优得分: {best_s.multi_metric.composite_score:.3f}")
            lines.append(f"    最优策略: {best_s.strategy[:40]}...")
        
        # 代码进化
        best_c = self.code_journal.get_best_node()
        lines.append(f"\n  💻 代码进化:")
        lines.append(f"    节点数: {len(self.code_journal)}")
        if best_c and best_c.metric:
            lines.append(f"    最优指标: {best_c.metric.value:.3f}")
            lines.append(f"    代码长度: {len(best_c.strategy)} 字符")
        
        lines.append("\n" + "=" * 55)
        return "\n".join(lines)
