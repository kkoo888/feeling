"""
Agent 运行器：执行策略并与 Agent 对话，收集执行指标

从 AIDE 的 interpreter.py 改造而来：
- 原 CodeInterpreter 在子进程中执行 Python 代码
- 现 AgentRunner 通过 LLM 执行策略对话
- 收集对话指标（成功率、响应时间、满意度等）
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable

from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("evolution")


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    策略执行结果。
    包含输出、执行时间和异常信息。
    与 AIDE 的 ExecutionResult 接口兼容。
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


@dataclass
class ConversationMetrics(DataClassJsonMixin):
    """对话指标：记录一次策略执行的详细度量"""

    # 基础指标
    success: bool = False          # 是否成功完成
    response_time: float = 0.0     # 响应时间（秒）
    token_count: int = 0           # 使用的 token 数量

    # 质量指标（由 LLM 评估）
    relevance_score: float = 0.0   # 相关性 [0, 1]
    coherence_score: float = 0.0   # 连贯性 [0, 1]
    helpfulness_score: float = 0.0 # 有用性 [0, 1]

    # 错误信息
    error_message: str | None = None
    error_type: str | None = None


# 策略执行回调类型
StrategyExecCallback = Callable[[str, bool], ExecutionResult]


class AgentRunner:
    """
    Agent 运行器：执行策略并收集指标。

    与 AIDE 的 Interpreter 对应，但不是执行 Python 代码，
    而是将策略文本发送给 LLM 后端执行 Agent 对话。

    工作流程：
    1. 接收策略文本（strategy）
    2. 通过 LLM 执行策略对话
    3. 收集执行输出和指标
    4. 返回 ExecutionResult（与 AIDE 接口兼容）
    """

    def __init__(
        self,
        task_desc: str,
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        timeout: int = 3600,
    ):
        """
        初始化 Agent 运行器。

        Args:
            task_desc: 任务描述，提供给策略执行的上下文
            model: 使用的 LLM 模型
            temperature: 采样温度
            timeout: 超时时间（秒）
        """
        self.task_desc = task_desc
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.last_metrics: ConversationMetrics | None = None

    def run(self, strategy: str, reset_session: bool = True) -> ExecutionResult:
        """
        执行一个策略，通过 LLM 对话获取结果。

        这是 AgentRunner 的核心方法，与 AIDE Interpreter.run() 接口兼容。

        Args:
            strategy: 策略文本（可以是 prompt、行为规则、参数等）
            reset_session: 是否重置会话（保留接口兼容，Agent 对话总是新会话）

        Returns:
            ExecutionResult: 包含输出和执行信息
        """
        logger.debug(f"AgentRunner executing strategy (reset_session={reset_session})")

        start_time = time.time()
        output: list[str] = []
        exc_type: str | None = None
        exc_info: dict | None = None
        exc_stack: list[tuple] | None = None

        try:
            # 构建策略执行的 prompt
            exec_prompt = self._build_exec_prompt(strategy)

            # 调用 LLM 执行
            from .backend import query

            result = query(
                system_message=exec_prompt,
                user_message=None,
                model=self.model,
                temperature=self.temperature,
            )

            # 记录输出
            output.append(f"=== 策略执行结果 ===\n")
            output.append(result if isinstance(result, str) else str(result))
            output.append(f"\n=== 执行完成 ===\n")

            # 更新指标
            exec_time = time.time() - start_time
            self.last_metrics = ConversationMetrics(
                success=True,
                response_time=exec_time,
                token_count=len(result.split()) if isinstance(result, str) else 0,
            )

        except Exception as e:
            exec_time = time.time() - start_time
            exc_type = type(e).__name__
            exc_info = {"args": [str(i) for i in e.args]} if hasattr(e, "args") else {}
            output.append(f"执行错误: {exc_type}: {str(e)}")
            output.append(f"\n执行时间: {exec_time:.2f} 秒")

            self.last_metrics = ConversationMetrics(
                success=False,
                response_time=exec_time,
                error_message=str(e),
                error_type=exc_type,
            )

            logger.error(f"Strategy execution failed: {exc_type}: {str(e)}")

        # 添加执行时间信息
        if exc_type is None:
            output.append(f"\n执行时间: {exec_time:.2f} 秒")

        return ExecutionResult(
            term_out=output,
            exec_time=exec_time,
            exc_type=exc_type,
            exc_info=exc_info,
            exc_stack=exc_stack,
        )

    def _build_exec_prompt(self, strategy: str) -> dict:
        """
        构建策略执行的 prompt。

        将任务描述和策略组合成 LLM 可执行的指令。
        """
        return {
            "角色": "你是一个智能 Agent 执行器。你的任务是根据给定的策略来完成任务。",
            "任务描述": self.task_desc,
            "执行策略": strategy,
            "执行要求": [
                "严格按照策略中定义的行为规则执行",
                "记录执行过程中的关键决策点",
                "输出执行结果和观察到的效果",
                "如果策略不完整或有歧义，按照最佳实践补充",
            ],
            "输出格式": (
                "请按以下格式输出：\n"
                "1. 执行摘要（2-3句话）\n"
                "2. 关键执行步骤\n"
                "3. 执行结果\n"
                "4. 观察与反思"
            ),
        }

    def get_last_metrics(self) -> ConversationMetrics | None:
        """获取最近一次执行的指标"""
        return self.last_metrics

    def cleanup_session(self):
        """
        清理会话（保留接口兼容）。
        Agent 对话是无状态的，无需实际清理。
        """
        logger.debug("AgentRunner session cleanup (no-op for stateless agent)")

    def evaluate_strategy_quality(self, strategy: str, result: str) -> dict:
        """
        评估策略执行质量（可选方法）。

        通过 LLM 对策略和执行结果进行多维评估。

        Args:
            strategy: 策略文本
            result: 执行结果

        Returns:
            包含各维度评估分数的字典
        """
        from .backend import query
        from .backend.utils import FunctionSpec

        eval_func_spec = FunctionSpec(
            name="submit_evaluation",
            json_schema={
                "type": "object",
                "properties": {
                    "task_success": {
                        "type": "number",
                        "description": "任务完成度 [0, 1]，策略是否成功完成了目标任务",
                    },
                    "user_satisfaction": {
                        "type": "number",
                        "description": "用户满意度 [0, 1]，结果是否会让用户满意",
                    },
                    "efficiency": {
                        "type": "number",
                        "description": "执行效率 [0, 1]，策略是否高效完成任务",
                    },
                    "safety": {
                        "type": "number",
                        "description": "安全性 [0, 1]，策略是否有副作用或风险",
                    },
                    "creativity": {
                        "type": "number",
                        "description": "创造性 [0, 1]，策略是否有创新之处",
                    },
                    "summary": {
                        "type": "string",
                        "description": "对策略执行效果的简短总结",
                    },
                },
                "required": ["task_success", "user_satisfaction", "efficiency", "safety", "creativity", "summary"],
            },
            description="提交对策略执行效果的多维评估。",
        )

        prompt = {
            "角色": "你是一个策略评估专家，负责评估 Agent 策略的执行效果。",
            "任务描述": self.task_desc,
            "策略内容": strategy,
            "执行结果": result,
            "评估要求": [
                "从五个维度评估策略效果，每个维度分数在 [0, 1] 之间",
                "task_success: 策略是否成功完成了目标任务",
                "user_satisfaction: 结果的质量是否让用户满意",
                "efficiency: 策略是否高效（时间、资源）",
                "safety: 策略是否有副作用、是否安全",
                "creativity: 策略是否有创新、非平凡的思路",
            ],
        }

        try:
            response = query(
                system_message=prompt,
                user_message=None,
                func_spec=eval_func_spec,
                model=self.model,
                temperature=0.3,
            )

            if isinstance(response, dict):
                return response
            return {}
        except Exception as e:
            logger.error(f"Strategy quality evaluation failed: {e}")
            return {}
