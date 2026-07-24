# -*- coding: utf-8 -*-
"""
B2 因果目标评分模块 — 进化版2 (置信度加权)
============================================
改进: 加置信度加权机制, 用 intervene 返回的 p_value 计算统计置信度,
对 bonus 进行加权。p_value 越小(越显著)则置信度越高, bonus 权重越大。

进化策略:
  1. confidence_weight=0.3 — 启用置信度加权
  2. confidence = 1 - p_value (p_value 来自 intervene 结果)
  3. conf_factor = (1-cw) + cw * confidence — 置信度加权因子
  4. bonus *= conf_factor — 高置信度效应获得更高分

这修复了 v2 的 NameError bug (confidence 未初始化), 同时提升 conf 信号 0→0.5。

参考:
  - Pearl 2000: 因果干预效应 (do-calculus)
  - Lattimore et al. 2016: Causal Bandits
"""
import math
from typing import Dict, List, Optional, Any, Tuple


# ── 模块级状态 ──────────────────────────────────────────
_last_engine = None

# 关键词: 用于识别可作为目标(outcome)的变量
_TARGET_KEYWORDS = (
    "emotion", "valence", "engagement", "satisfaction", "bond", "trust",
    "outcome", "result", "response", "feeling", "sentiment", "arousal",
    "responded", "msg_len", "question", "request",
)

# GoalDomain → treatment 变量名映射
_DOMAIN_TO_TREATMENT = {
    "learn": "topic_code",
    "explore": "topic_code",
    "improve": "aris_resp_len",
    "create": "aris_resp_len",
    "connect": "aris_initiative",
    "heal": "aris_initiative",
    "optimize": "aris_keyword_density",
    "secure": "aris_initiative",
}


# ── 可进化参数 ──────────────────────────────────────────

def get_scoring_config() -> dict:
    """返回可进化的目标评分参数 (进化版2: 置信度加权)。"""
    return {
        "min_effect": 0.1,
        "max_bonus": 20.0,
        "max_penalty": 10.0,
        "effect_scale": 10.0,
        "n_samples": 30,
        "default_outcome": "主人_responded",
        "max_targets": 5,
        "negative_penalty": 5.0,
        "confidence_weight": 0.3,
    }


# ── 动态目标发现 ────────────────────────────────────────

def _discover_targets(ce) -> list:
    targets, seen = [], set()
    try:
        from causal_feature_extractor import get_variable_roles
        roles = get_variable_roles()
        for v in roles.get("outcome", []):
            if v not in seen:
                targets.append(v)
                seen.add(v)
    except Exception:
        pass
    if ce is not None and hasattr(ce, 'graph') and ce.graph.edges:
        in_deg = {}
        all_nodes = set()
        for ek in ce.graph.edges:
            parts = ek.split("->")
            if len(parts) == 2:
                s, t = parts[0].strip(), parts[1].strip()
                all_nodes.add(s)
                all_nodes.add(t)
                in_deg[t] = in_deg.get(t, 0) + 1
        for n in all_nodes:
            if in_deg.get(n, 0) > 0 and n not in seen:
                if any(kw in n.lower() for kw in _TARGET_KEYWORDS):
                    targets.append(n)
                    seen.add(n)
        if len(targets) < 3:
            for n in all_nodes:
                if in_deg.get(n, 0) > 0 and n not in seen:
                    targets.append(n)
                    seen.add(n)
    try:
        from causal_feature_extractor import get_variable_roles
        treatments = set(get_variable_roles().get("treatment", []))
        targets = [t for t in targets if t not in treatments]
    except Exception:
        pass
    cfg = get_scoring_config()
    if not targets:
        targets = [cfg["default_outcome"]]
    return targets[:cfg.get("max_targets", 5)]


def get_goal_targets(ce=None) -> List[str]:
    global _last_engine
    engine = ce if ce is not None else _last_engine
    if engine is not None:
        try:
            t = _discover_targets(engine)
            if t:
                return t
        except Exception:
            pass
    return [get_scoring_config()["default_outcome"]]


# ── 核心评分函数 ────────────────────────────────────────

def causal_goal_score(ce, goal_domain: str, goal_description: str = "") -> Dict:
    global _last_engine
    _last_engine = ce
    cfg = get_scoring_config()
    treatment = _DOMAIN_TO_TREATMENT.get(goal_domain, "aris_initiative")
    targets = get_goal_targets(ce)
    if not targets:
        targets = [cfg["default_outcome"]]

    cw = cfg.get("confidence_weight", 0.0)

    all_effects = {}
    best_effect = 0.0
    best_target = targets[0]
    best_confidence = 1.0  # 始终初始化, 避免 NameError

    for target in targets:
        try:
            iv = ce.intervene(treatment, 1.0, target, n_samples=cfg["n_samples"])
            effect = iv.get("intervention_effect", 0)
        except Exception:
            effect = 0.0
            iv = {}

        # 计算置信度: p_value 越小, 置信度越高
        p_value = 1.0
        if isinstance(iv, dict):
            p_value = iv.get("p_value", iv.get("pvalue", 1.0))
        try:
            p_value = float(p_value)
        except (TypeError, ValueError):
            p_value = 1.0
        confidence = max(0.0, min(1.0, 1.0 - p_value))

        all_effects[target] = round(effect, 3)
        if abs(effect) > abs(best_effect):
            best_effect = effect
            best_target = target
            best_confidence = confidence

    # 置信度加权因子: (1-cw) + cw * confidence
    conf_factor = (1.0 - cw) + cw * best_confidence

    bonus = 0.0
    if best_effect > cfg["min_effect"]:
        bonus = min(cfg["max_bonus"], best_effect * cfg["effect_scale"]) * conf_factor
    elif best_effect < -cfg["min_effect"]:
        penalty = abs(best_effect) * cfg.get("negative_penalty", 0) * conf_factor
        bonus = -min(cfg.get("max_penalty", 10.0), penalty)

    return {
        "domain": goal_domain,
        "treatment": treatment,
        "best_target": best_target,
        "best_effect": round(best_effect, 3),
        "bonus": round(bonus, 1),
        "all_effects": all_effects,
        "confidence": round(best_confidence, 3),
    }


def score_multiple_goals(ce, goals: List[Tuple[str, str]]) -> List[Dict]:
    results = []
    for domain, desc in goals:
        result = causal_goal_score(ce, domain, desc)
        results.append(result)
    results.sort(key=lambda r: r.get("bonus", 0), reverse=True)
    return results


def is_causal_ready(ce) -> bool:
    if ce is None:
        return False
    if not hasattr(ce, 'intervene'):
        return False
    if not hasattr(ce, 'observations'):
        return False
    if len(ce.observations) < 10:
        return False
    if not hasattr(ce, 'graph') or not ce.graph.edges:
        return False
    return True