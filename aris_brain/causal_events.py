# -*- coding: utf-8 -*-
"""
因果事件总线 — 轻量级内存 pub/sub
=====================================
B7: 让因果引擎从"被调用的工具"变成"事件广播源"。

设计原则:
  - 独立模块,不侵入 cognitive_bus.py 的文件轮询路由
  - 同步调用,异常隔离(一个订阅者崩溃不影响其他)
  - 无依赖,纯 stdlib

事件类型:
  CAUSAL_DISCOVERY       — discover() 完成,广播发现的边和混杂
  COUNTERFACTUAL_RESULT  — 反事实复盘完成,广播 delta
  INTERVENTION_EFFECT    — 干预效应计算完成,广播效应值
  MEDIATION_DECOMPOSED   — 中介分解完成,广播直接/间接效应
  TRANSITIVE_CHAIN       — 传递链发现

订阅者:
  DesireEngine     → 收到因果发现后重新评估欲望优先级
  EmotionEngine    → 收到反事实结果后调整情绪归因
  EpisodicMemory  → 收到中介分解后存储带因果标注的 episode
  GoalEngine      → 收到干预效应后更新目标评分
"""

import logging
from typing import Dict, List, Callable, Any
from collections import defaultdict

logger = logging.getLogger("aris.causal_events")

# ── 事件类型常量 ──────────────────────────────────────────

CAUSAL_DISCOVERY = "causal_discovery"
COUNTERFACTUAL_RESULT = "cf_result"
INTERVENTION_EFFECT = "intervention_effect"
MEDIATION_DECOMPOSED = "mediation"
TRANSITIVE_CHAIN = "transitive_chain"

# ── 订阅注册表 ────────────────────────────────────────────

_subscribers: Dict[str, List[Callable[[Dict], Any]]] = defaultdict(list)
_stats = {"published": 0, "delivered": 0, "failed": 0}


def subscribe(event_type: str, handler: Callable[[Dict], Any]) -> None:
    """订阅因果事件。

    Args:
        event_type: 事件类型常量(见上方定义)
        handler: 回调函数,接收一个 dict 参数

    Example:
        from causal_events import CAUSAL_DISCOVERY, subscribe
        subscribe(CAUSAL_DISCOVERY, self._on_causal_discovery)
    """
    _subscribers[event_type].append(handler)
    logger.debug(f"因果事件订阅: {event_type} → {getattr(handler, '__name__', str(handler))}")


def publish(event_type: str, data: Dict) -> None:
    """广播因果事件给所有订阅者。

    同步调用,异常隔离:单个订阅者崩溃不影响其他订阅者。
    """
    _stats["published"] += 1
    handlers = _subscribers.get(event_type, [])
    if not handlers:
        return
    logger.debug(f"因果事件广播: {event_type} → {len(handlers)} 个订阅者")
    for handler in handlers:
        try:
            handler(data)
            _stats["delivered"] += 1
        except Exception as e:
            _stats["failed"] += 1
            logger.warning(
                f"因果事件处理失败 {event_type} → "
                f"{getattr(handler, '__name__', str(handler))}: {e}"
            )


def stats() -> Dict:
    """返回事件总线统计。"""
    return {
        **_stats,
        "subscriber_count": sum(len(v) for v in _subscribers.values()),
        "event_types": {k: len(v) for k, v in _subscribers.items() if v},
    }


# ── 便捷发布函数(供 cognitive_bridge tick 调用) ─────────


def emit_discovery(edges_count: int, confounders: List[str],
                   edges: List = None, details: Dict = None) -> None:
    """广播因果发现事件。"""
    publish(CAUSAL_DISCOVERY, {
        "edges_count": edges_count,
        "confounders": confounders,
        "edges": edges or [],
        "details": details or {},
    })


def emit_counterfactual(delta: float, cause_var: str, effect_var: str,
                         observed_y: float = None, cf_y: float = None) -> None:
    """广播反事实结果事件。"""
    publish(COUNTERFACTUAL_RESULT, {
        "delta": delta,
        "cause_var": cause_var,
        "effect_var": effect_var,
        "observed_y": observed_y,
        "counterfactual_y": cf_y,
    })


def emit_intervention(effect: float, do_var: str, target_var: str,
                      method: str = "backdoor") -> None:
    """广播干预效应事件。"""
    publish(INTERVENTION_EFFECT, {
        "effect": effect,
        "do_var": do_var,
        "target_var": target_var,
        "method": method,
    })


def emit_mediation(direct_effect: float, indirect_effect: float,
                    treatment: str, mediator: str, outcome: str) -> None:
    """广播中介分解事件。"""
    publish(MEDIATION_DECOMPOSED, {
        "direct_effect": direct_effect,
        "indirect_effect": indirect_effect,
        "treatment": treatment,
        "mediator": mediator,
        "outcome": outcome,
    })


def emit_transitive_chain(path: List[str], confidence: float = 0.0) -> None:
    """广播传递链发现事件。"""
    publish(TRANSITIVE_CHAIN, {
        "path": path,
        "confidence": confidence,
    })
