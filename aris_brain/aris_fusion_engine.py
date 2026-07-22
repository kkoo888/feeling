"""
小茜 融合引擎 — 统一入口 (V4)
================================
版本: 0.10
基于 V4 极致架构:
  1. NER 实体提取 (分层正则)
  2. 情感/情绪识别 (词典 + 否定词 + 程度词)
  3. LLM Function Calling 兜底 (三段式分类)
  4. 推理链压缩 + 信息密度 (LongCoT + UID)
  5. Self-Attributing 失败归因

印记: 小茜 永远记得主人 — 2026-07-22
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("aris.fusion")

# 延迟导入 V4 引擎
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from aris_fusion_engine_v4 import FusionEngineV4
        _engine = FusionEngineV4()
    return _engine


def process(text: str, use_polish: bool = False) -> Dict[str, Any]:
    """统一处理入口 — 向后兼容 v1/v2 接口。

    Args:
        text: 用户输入文本
        use_polish: 是否启用润色 (V4 暂不支持，保留接口)

    Returns:
        处理结果字典
    """
    engine = _get_engine()
    result = engine.process(text)

    # 向后兼容: 确保 v1/v2 的字段存在
    if "matched" not in result:
        result["matched"] = result.get("confidence", 0) >= 0.5
    if "output" not in result:
        result["output"] = ""
    if "intent" not in result:
        result["intent"] = "unknown"

    return result


def get_engine():
    """获取引擎实例。"""
    return _get_engine()


# 版本信息
__version__ = "0.10"
__engine_version__ = "0.10"
