"""
LAAP AGI 认知模块包。

子模块:
- world_model: 统一世界模型（实体/关系建模）
- causal: 因果推理引擎（UnifiedCausalEngine）
- meta_learning: 元学习（学习效率评估）
- curriculum: 课程系统（渐进学习）
- perception: 感知引擎
- safety: 安全引擎
- conscious: 意识模块
- self_model: 自我模型
- memory_system: AGI 记忆系统
"""

from .curriculum import CurriculumEngine
from .meta_learning import MetaLearningEngine
from .perception import UnifiedPerceptionEngine
from .ctm import ContinuousThoughtEngine
from .retnet_router import RetNetRouter
