"""
进化 Agent：feeling 项目自进化系统的核心决策引擎

从 AIDE 的 agent.py 改造而来：
- 所有 "code" 相关 prompt 改为 "strategy"（策略）
- _draft() → 生成新策略（提示词/行为规则/参数）
- _improve() → 改进现有策略
- _debug() → 修复失败策略
- 新增 _explore() → 探索全新方向
- search_policy() 加入 explore 概率
- plan_and_code_query() → plan_and_strategy_query()
"""

import logging
import random
from enum import Enum
from typing import Any, Callable, cast

from .backend import FunctionSpec, query
from .runner import ExecutionResult
from .journal import EvolutionJournal, EvolutionNode
from .utils.metric import MetricValue, WorstMetricValue, MultiDimensionalMetric
from .utils.response import extract_code, extract_text_up_to_code, wrap_code

logger = logging.getLogger("evolution")


ExecCallbackType = Callable[[str, bool], ExecutionResult]


class SearchAction(Enum):
    """search_policy() 的返回动作类型"""
    DRAFT = "draft"       # 生成新草稿
    EXPLORE = "explore"   # 探索全新方向


# 策略评估的函数调用规范
review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true 如果执行输出显示策略执行失败或存在 bug，否则 false。",
            },
            "summary": {
                "type": "string",
                "description": "如果有 bug，提出修复建议。否则，写一段简短总结（2-3句话）描述策略执行的经验发现。",
            },
            "metric": {
                "type": "number",
                "description": "如果策略执行成功，报告验证指标值。否则留空。",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true 如果指标应最小化（如 MSE），false 如果指标应最大化（如准确率）。",
            },
        },
        "required": ["is_bug", "summary", "metric", "lower_is_better"],
    },
    description="提交对策略执行输出的评估。",
)


class EvolutionAgent:
    """
    进化 Agent：负责生成、改进、调试和探索策略。

    核心能力：
    - _draft(): 生成初始策略
    - _improve(): 改进现有策略
    - _debug(): 修复失败策略
    - _explore(): 探索全新方向（AIDE 没有的能力）
    - search_policy(): 选择下一步行动（含 explore 概率）
    """

    def __init__(
        self,
        task_desc: str,
        journal: EvolutionJournal,
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        feedback_model: str | None = None,
        feedback_temp: float = 0.3,
        num_drafts: int = 3,
        debug_prob: float = 0.2,
        explore_prob: float = 0.15,
        max_debug_depth: int = 3,
    ):
        """
        初始化进化 Agent。

        Args:
            task_desc: 任务描述
            journal: 进化日志
            model: 策略生成使用的模型
            temperature: 策略生成的温度
            feedback_model: 反馈评估使用的模型（默认同 model）
            feedback_temp: 反馈评估的温度
            num_drafts: 初始草稿数量
            debug_prob: 调试概率
            explore_prob: 探索概率
            max_debug_depth: 最大调试深度
        """
        super().__init__()
        self.task_desc = task_desc
        self.journal = journal
        self.model = model
        self.temperature = temperature
        self.feedback_model = feedback_model or model
        self.feedback_temp = feedback_temp
        self.num_drafts = num_drafts
        self.debug_prob = debug_prob
        self.explore_prob = explore_prob
        self.max_debug_depth = max_debug_depth

    def search_policy(self) -> EvolutionNode | SearchAction:
        """
        选择一个节点来工作，或返回 SearchAction 表示生成新节点。

        返回:
            EvolutionNode: 选择的节点（improve/debug）
            SearchAction.DRAFT: 生成新草稿
            SearchAction.EXPLORE: 探索全新方向
        """
        # 初始草稿阶段
        if len(self.journal.draft_nodes) < self.num_drafts:
            logger.debug("[search policy] 草稿不足，生成新草稿")
            return SearchAction.DRAFT

        # 调试：随机概率选择 buggy 节点
        if random.random() < self.debug_prob:
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= self.max_debug_depth)
            ]
            if debuggable_nodes:
                logger.debug("[search policy] 选择调试")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] 调试概率触发但无可调试节点")

        # 探索：随机概率触发全新方向探索
        if random.random() < self.explore_prob:
            logger.debug("[search policy] 触发探索模式")
            return SearchAction.EXPLORE

        # 回退到草稿：如果没有好节点
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.debug("[search policy] 无好节点，生成新草稿")
            return SearchAction.DRAFT

        # 贪婪：选择最佳节点
        greedy_node = self.journal.get_best_node()
        logger.debug("[search policy] 选择贪婪节点")
        return greedy_node

    @property
    def _prompt_environment(self):
        """构建环境描述 prompt"""
        env_prompt = {
            "执行环境": (
                "你的策略将在一个受控环境中执行。"
                "请确保策略中的行为规则和参数是可执行的、明确的。"
                "如果策略包含调用外部工具的指令，请确保工具名称和参数格式正确。"
            ),
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        """构建策略实现指南"""
        impl_guideline = [
            "策略应该**完整实现所提议的方案**，包含所有必要的步骤和参数。",
            "策略应该是自包含的，可以直接被执行器理解和执行。",
            "不要跳过任何步骤，不要在完成之前终止。",
            "你的响应应该只包含一个策略代码块。",
            "注意策略的复杂度，应该在合理时间内完成执行。",
            "如果需要与外部系统交互，请明确指定交互协议和数据格式。",
            "策略应该包含明确的成功/失败判断标准。",
        ]
        return {"策略实现指南": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        """构建响应格式说明"""
        return {
            "响应格式": (
                "你的响应应该包含：\n"
                "1. 一段简短的方案概述（3-5句话），用自然语言描述你的策略思路\n"
                "2. 一个策略代码块（用 ``` 包裹），实现这个方案\n"
                "不应有额外的标题或文本。只需自然语言概述 + 策略代码块。"
            )
        }

    def plan_and_strategy_query(self, prompt, retries=3) -> tuple[str, str]:
        """
        在同一次 LLM 调用中生成自然语言计划 + 策略，然后分离它们。

        如果响应没有代码块，则将整个响应作为策略（plan=摘要, strategy=全文）。
        """
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.model,
                temperature=self.temperature,
            )

            # 尝试从响应中提取策略（代码块形式）
            strategy = extract_code(completion_text)
            plan = extract_text_up_to_code(completion_text)

            if strategy and plan:
                return plan, strategy

            # 如果没有代码块，将整个响应作为策略
            if completion_text and len(completion_text.strip()) > 20:
                # 按句号或换行分割，取前 2-3 句作为 plan
                separators = ['。', '. ', '\n']
                plan_end = len(completion_text)
                sep_count = 0
                for sep in separators:
                    pos = 0
                    while pos < len(completion_text):
                        idx = completion_text.find(sep, pos)
                        if idx < 0:
                            break
                        sep_count += 1
                        if sep_count >= 3:
                            plan_end = idx + len(sep)
                            break
                        pos = idx + len(sep)
                    if sep_count >= 3:
                        break

                plan = completion_text[:plan_end].strip()
                strategy = completion_text
                return plan, strategy

            print("计划 + 策略提取失败，重试...")
        print("最终计划 + 策略提取尝试失败，放弃...")
        return "", completion_text or ""

    def _draft(self) -> EvolutionNode:
        """
        生成初始策略草稿。

        与 AIDE 的 _draft() 对应，但 prompt 面向策略生成而非代码生成。
        """
        prompt: Any = {
            "简介": (
                "你是一个智能策略设计师。"
                "为了完成给定的任务，你需要设计一个优秀的、有创意的执行策略，"
                "然后将其实现为可执行的策略规范。"
                "我们现在将提供任务描述。"
            ),
            "任务描述": self.task_desc,
            "历史记录": self.journal.generate_summary(),
            "指令": {},
        }
        prompt["指令"] |= self._prompt_resp_fmt
        prompt["指令"] |= {
            "策略设计指南": [
                "第一个策略设计应该相对简单，不需要复杂的优化或多策略集成。",
                "参考历史记录中的信息，不要提出重复的策略方案。",
                "策略概述应该是 3-5 句话。",
                "提出合理的评估指标来衡量策略效果。",
                "策略应该直接针对任务目标，不要做无关的探索。",
            ],
        }
        prompt["指令"] |= self._prompt_impl_guideline
        prompt["指令"] |= self._prompt_environment

        plan, strategy = self.plan_and_strategy_query(prompt)
        return EvolutionNode(plan=plan, strategy=strategy)

    def _improve(self, parent_node: EvolutionNode) -> EvolutionNode:
        """
        改进现有策略。

        与 AIDE 的 _improve() 对应，基于父节点策略进行改进。
        """
        prompt: Any = {
            "简介": (
                "你是一个智能策略设计师。下面提供了一个之前开发的策略，"
                "你应该改进它以进一步提升执行效果。"
                "首先用自然语言简述改进计划，然后基于提供的先前策略实现改进。"
            ),
            "任务描述": self.task_desc,
            "历史记录": self.journal.generate_summary(),
            "指令": {},
        }
        prompt["先前策略"] = {
            "策略内容": wrap_code(parent_node.strategy),
        }

        prompt["指令"] |= self._prompt_resp_fmt
        prompt["指令"] |= {
            "策略改进指南": [
                "改进概述应该是对先前策略如何改进的简要自然语言描述。",
                "你应该非常具体，只提出一个可操作的改进点。",
                "这个改进应该是原子性的，以便我们可以实验性地评估其效果。",
                "参考历史记录中的信息来提出改进。",
                "改进概述应该是 3-5 句话。",
            ],
        }
        prompt["指令"] |= self._prompt_impl_guideline

        plan, strategy = self.plan_and_strategy_query(prompt)
        return EvolutionNode(
            plan=plan,
            strategy=strategy,
            parent=parent_node,
        )

    def _debug(self, parent_node: EvolutionNode) -> EvolutionNode:
        """
        修复失败的策略。

        与 AIDE 的 _debug() 对应，基于错误信息修复策略。
        """
        prompt: Any = {
            "简介": (
                "你是一个智能策略设计师。"
                "你之前的策略有 bug，基于下面的信息，你应该修订它以修复这个 bug。"
                "你的响应应该包含自然语言的修复方案概述，"
                "然后是一个策略代码块来实现修复。"
            ),
            "任务描述": self.task_desc,
            "先前（有 bug 的）策略": wrap_code(parent_node.strategy),
            "执行输出": wrap_code(parent_node.term_out, lang=""),
            "指令": {},
        }
        prompt["指令"] |= self._prompt_resp_fmt
        prompt["指令"] |= {
            "Bug 修复指南": [
                "你应该写一段简要的自然语言描述（3-5句话），说明如何修复先前实现中的问题。",
                "专注于修复具体问题，不要引入不相关的改动。",
                "如果错误信息不明确，做出最合理的推断。",
            ],
        }
        prompt["指令"] |= self._prompt_impl_guideline

        plan, strategy = self.plan_and_strategy_query(prompt)
        return EvolutionNode(plan=plan, strategy=strategy, parent=parent_node)

    def _explore(self) -> EvolutionNode:
        """
        探索全新方向（AIDE 没有的能力）。

        生成一个与现有策略完全不同的全新策略，用于跳出局部最优。
        """
        # 收集现有策略的摘要，用于避免重复
        existing_summaries = []
        for n in self.journal.good_nodes[:5]:  # 只看最近 5 个好节点
            existing_summaries.append(f"- {n.plan}")

        existing_str = "\n".join(existing_summaries) if existing_summaries else "无"

        prompt: Any = {
            "简介": (
                "你是一个创新策略设计师。"
                "你需要跳出已有的思路，探索一个全新的、可能非常不同的策略方向。"
                "目标是发现可能被遗漏的全新解法。"
            ),
            "任务描述": self.task_desc,
            "已有策略方向": existing_str,
            "指令": {},
        }
        prompt["指令"] |= self._prompt_resp_fmt
        prompt["指令"] |= {
            "探索指南": [
                "这个策略应该与已有策略方向**完全不同**。",
                "可以考虑完全不同的算法、数据处理方式、或问题分解方法。",
                "不需要追求完美，关键是探索新方向。",
                "概述应该是 3-5 句话，重点说明为什么这个方向值得探索。",
                "策略可以更激进，允许一定程度的冒险。",
            ],
        }
        prompt["指令"] |= self._prompt_impl_guideline
        prompt["指令"] |= self._prompt_environment

        plan, strategy = self.plan_and_strategy_query(prompt)
        return EvolutionNode(plan=plan, strategy=strategy)

    def update_data_preview(self):
        """更新数据预览（保留接口兼容，feeling 项目中可能不需要）"""
        pass

    def step(self, exec_callback: ExecCallbackType):
        """
        执行一步进化。

        根据 search_policy() 的结果决定行动：
        - SearchAction.EXPLORE → 探索新方向
        - SearchAction.DRAFT → 生成新草稿
        - buggy 节点 → 调试
        - 好节点 → 改进
        """
        parent = self.search_policy()

        logger.debug(f"Agent 生成策略，父节点类型: {type(parent)}")

        if isinstance(parent, SearchAction):
            if parent == SearchAction.EXPLORE:
                result_node = self._explore()
            else:
                result_node = self._draft()
        elif parent.is_buggy:
            result_node = self._debug(parent)
        else:
            result_node = self._improve(parent)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.strategy, True),
        )
        self.journal.append(result_node)

    def parse_exec_result(self, node: EvolutionNode, exec_result: ExecutionResult):
        """
        解析策略执行结果。

        与 AIDE 的 parse_exec_result() 对应，使用 LLM 评估策略执行效果。
        """
        logger.info(f"Agent 正在解析节点 {node.id} 的执行结果")

        node.absorb_exec_result(exec_result)

        prompt = {
            "简介": (
                "你是一个智能策略评估专家。"
                "你设计了一个策略来完成任务，现在需要评估策略执行的输出。"
                "你应该判断是否有 bug，并报告经验发现。"
            ),
            "任务描述": self.task_desc,
            "策略内容": wrap_code(node.strategy),
            "执行输出": wrap_code(node.term_out, lang=""),
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.feedback_model,
                temperature=self.feedback_temp,
            ),
        )

        # 如果指标不是浮点数，设为 None
        if not isinstance(response["metric"], float):
            response["metric"] = None

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
        )

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )

        # 计算多维指标
        try:
            exec_time = exec_result.exec_time if exec_result else 0.1
            node.multi_metric = MultiDimensionalMetric(
                task_success=1.0 if not node.is_buggy else 0.0,
                user_satisfaction=node.metric.value if node.metric and node.metric.value else 0.0,
                efficiency=max(0.0, 1.0 - exec_time / 3600.0),
                safety=1.0 if node.exc_type is None else 0.5,
                creativity=0.5,
            )
        except Exception:
            pass  # 多维指标是可选的
