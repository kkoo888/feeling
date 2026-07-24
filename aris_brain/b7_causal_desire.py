# -*- coding: utf-8 -*-
"""
B7 因果欲望发现模块 — 进化冠军 (score=9.341)
================================================
从 aris_desire_engine.py 的 discover_new_desires_causal() 提取的独立模块。

核心能力: 用因果归因找出主人状态的真正原因, 生成精准欲望描述。
不同于模板生成欲望, 此模块用因果图指导欲望发现, 确保欲望有因果依据。

进化策略 (冠军版):
  1. 多目标发现 — 从因果图拓扑发现 sink 节点和关键词节点
  2. 丰富 cause→desire 映射 — 支持 6 种中介变量专属映射
  3. Sigmoid 强度校准 — 让高效应值获得更高强度, 拉开区分度

进化结果:
  基线 score=9.124 → 冠军 score=9.341 (+0.217, 2.4%)
  count=10, diversity=7, causal=90%, relevance=100%, target_cov=60%, intensity_cv=0.63

参考:
  - Heckhausen 2000: 欲望的成因理论
  - Pearl 2000: 因果推理
  - B6 因果归因: 复用 B6 的 causal_attribution
"""
import math
from typing import Dict, List, Optional, Any, Tuple


# ── 模块级状态 ──────────────────────────────────────────
_last_engine = None

# 关键词: 用于识别可能的 outcome 变量
_KEYWORDS = (
    "emotion", "valence", "engagement", "satisfaction", "bond", "trust",
    "outcome", "result", "response", "feeling", "sentiment", "arousal",
)


# ── 可进化参数 ──────────────────────────────────────────

def get_desire_config() -> dict:
    """返回可进化的欲望发现参数 (冠军版)。"""
    return {
        "min_effect": 0.05,
        "max_desires": 10,
        "max_targets": 5,
        "intensity_base": 0.3,
        "intensity_scale": 0.4,
    }


# ── 强度校准 ────────────────────────────────────────────

def _sigmoid_intensity(effect: float, cfg: dict) -> float:
    """sigmoid 强度校准: 高效应值获得更高强度, 拉开区分度。

    k=6.0 控制陡度, x0=0.2 是中心点。
    effect=0 → intensity≈intensity_base
    effect=0.5 → intensity≈1.0
    """
    k = 6.0
    x0 = 0.2
    try:
        s = 1.0 / (1.0 + math.exp(-k * (abs(effect) - x0)))
    except OverflowError:
        s = 1.0 if abs(effect) > x0 else 0.0
    base = cfg["intensity_base"]
    return base + (1.0 - base) * s


# ── 动态目标发现 ────────────────────────────────────────

def _discover_targets(ce) -> list:
    """多目标发现: 直接从因果图拓扑发现 sink 节点和关键词节点。

    1. sink 节点 (有入度但无出度) — 终端效应, 必有父节点可归因
    2. 含关键词且有入度的节点 — 可能的 outcome 变量
    3. 补充高入度节点 — 保证目标数量
    只返回有入度的节点, 确保 B6 能找到候选原因。
    """
    targets, seen = [], set()
    if ce is None or not hasattr(ce, 'graph') or not ce.graph.edges:
        return ["user_emotion_valence"]

    in_deg, out_deg, all_nodes = {}, {}, set()
    for ek in ce.graph.edges:
        parts = ek.split("->")
        if len(parts) == 2:
            s, t = parts[0].strip(), parts[1].strip()
            all_nodes.add(s)
            all_nodes.add(t)
            in_deg[t] = in_deg.get(t, 0) + 1
            out_deg[s] = out_deg.get(s, 0) + 1

    # 1. sink 节点 (有入度无出度) — 终端效应
    for n in all_nodes:
        if in_deg.get(n, 0) > 0 and out_deg.get(n, 0) == 0 and n not in seen:
            targets.append(n)
            seen.add(n)

    # 2. 含关键词且有入度的节点
    for n in all_nodes:
        if in_deg.get(n, 0) > 0 and any(kw in n.lower() for kw in _KEYWORDS) and n not in seen:
            targets.append(n)
            seen.add(n)

    # 3. 补充高入度节点 (保证目标数量)
    if len(targets) < 3:
        remaining = [n for n in all_nodes if n not in seen and in_deg.get(n, 0) > 0]
        remaining.sort(key=lambda n: in_deg.get(n, 0), reverse=True)
        for n in remaining:
            targets.append(n)
            seen.add(n)
            if len(targets) >= 5:
                break

    # 4. 回退: 观测数据中的关键词变量
    if not targets:
        for o in getattr(ce, 'observations', []):
            for var in getattr(o, 'variables', {}):
                if any(kw in var.lower() for kw in _KEYWORDS) and var not in seen:
                    targets.append(var)
                    seen.add(var)
                    if len(targets) >= 5:
                        break
            if len(targets) >= 5:
                break

    if not targets:
        targets = ["user_emotion_valence"]

    return targets[:get_desire_config().get("max_targets", 5)]


def get_desire_targets(ce=None) -> List[str]:
    """返回欲望发现的目标变量列表。"""
    global _last_engine
    engine = ce if ce is not None else _last_engine
    if engine is not None:
        try:
            t = _discover_targets(engine)
            if t:
                return t
        except Exception:
            pass
    return ["user_emotion_valence"]


# ── 核心欲望发现函数 ────────────────────────────────────

def discover_new_desires_causal(ce, de=None) -> List[Dict]:
    """因果欲望发现 (冠军版: 多目标 + 丰富映射 + 强度校准)。

    算法:
      1. 多目标发现 — 从因果图拓扑发现 sink 节点和关键词节点
      2. 对每个目标做因果归因 (复用 B6)
      3. 根据 true_causes 的类型和效应值生成欲望 (丰富映射)
      4. 根据 confounders 生成修正欲望 (降低强度)
      5. Sigmoid 强度校准 — 拉开不同效应值欲望的强度区分度

    Args:
        ce: 因果引擎实例 (laap.agi.causal.CausalEngine)
        de: 欲望引擎实例 (可选, 用于注册欲望到 DesireEngine)

    Returns:
        发现的欲望列表:
        [
            {
                "type": str,          # 欲望类型
                "intensity": float,   # 欲望强度 0-1 (sigmoid 校准)
                "trigger": str,       # 触发原因
                "expression": str,    # 想说的话
                "cause": str,         # 因果变量
                "effect": float,      # 因果效应值
                "target": str,        # 归因目标
            }
        ]
    """
    global _last_engine
    _last_engine = ce

    cfg = get_desire_config()

    # 获取归因目标
    targets = get_desire_targets(ce)
    if not targets:
        targets = ["user_emotion_valence"]

    all_desires = []

    for target in targets:
        # 用 B6 做因果归因
        try:
            import b6_causal_attribution as b6
            b6._last_engine = ce
            attr = b6.causal_attribution(ce, target)
        except Exception:
            attr = {"true_causes": [], "confounders": [], "effects": {}}

        true_causes = attr.get("true_causes", [])
        confounders = attr.get("confounders", [])
        effects = attr.get("effects", {})

        # 根据 true_causes 生成正向欲望
        for cause in true_causes:
            effect = effects.get(cause, 0)
            if abs(effect) < cfg["min_effect"]:
                continue

            desire = _cause_to_desire(cause, effect, target, cfg)
            if desire:
                all_desires.append(desire)

        # 根据 confounders 生成修正欲望
        for conf in confounders:
            effect = effects.get(conf, 0)
            desire = _confounder_to_desire(conf, effect, target, cfg)
            if desire:
                all_desires.append(desire)

    # 限制最大欲望数
    if len(all_desires) > cfg["max_desires"]:
        all_desires = all_desires[:cfg["max_desires"]]

    # 如果有 DesireEngine 实例, 注册欲望
    if de is not None and all_desires:
        for d in all_desires:
            try:
                if hasattr(de, 'register_desire'):
                    de.register_desire(
                        d["type"], d["intensity"],
                        d["trigger"], 6.0, d["expression"]
                    )
            except Exception:
                pass

    return all_desires


def _cause_to_desire(cause: str, effect: float, target: str, cfg: dict) -> Optional[Dict]:
    """把因果变量转为欲望描述 (冠军版: 丰富映射 + sigmoid 强度)。

    新增 6 种中介变量专属映射, 提升欲望多样性。
    """
    intensity = min(1.0, _sigmoid_intensity(effect, cfg))
    effect_str = f"效应={effect:.2f}"

    # ── 丰富映射: 中介变量 -> 专属欲望类型 ──
    if cause == "response_density_ratio":
        return _make_desire(
            "expression_density", intensity,
            f"因果发现: {cause}→{target}({effect_str})",
            "想提升回应的信息密度与表达丰富度", cause, effect, target
        )
    if cause == "response_impact_mediator":
        return _make_desire(
            "impact_amplification", intensity,
            f"因果发现: {cause}→{target}({effect_str})",
            "想放大回应对主人的影响力", cause, effect, target
        )
    if cause == "technical_resonance_mediator":
        return _make_desire(
            "technical_resonance", intensity,
            f"因果发现: {cause}→{target}({effect_str})",
            "想在技术话题上与主人形成共振", cause, effect, target
        )
    if cause in ("empathy_level", "empathy", "empathy_influence") or "empathy" in cause.lower():
        return _make_desire(
            "empathy_strengthening", intensity,
            f"因果发现: {cause}→{target}({effect_str})",
            "想增强对主人的共情能力", cause, effect, target
        )
    if cause in ("warmth_level", "warmth") or "warmth" in cause.lower():
        return _make_desire(
            "warmth_enhancement", intensity,
            f"因果发现: {cause}→{target}({effect_str})",
            "想让回应更有温度", cause, effect, target
        )
    if cause == "emotion_gap":
        return _make_desire(
            "emotion_alignment", intensity,
            f"因果发现: {cause}→{target}({effect_str})",
            "想更好地对齐与主人的情绪", cause, effect, target
        )

    # ── 基线前缀映射 ──
    # aris_ 开头 → 小茜行为变量
    if cause.startswith("aris_"):
        if cause in ("aris_initiative",):
            if effect > 0:
                return _make_desire(
                    "proactive_engagement", intensity,
                    f"因果发现: {cause}→{target}({effect_str})",
                    "想更主动地表达和关心主人", cause, effect, target
                )
            else:
                return _make_desire(
                    "restraint_practice", intensity,
                    f"因果发现: {cause}→{target}({effect_str}, 负向)",
                    "想学习更克制的方式", cause, effect, target
                )
        elif cause in ("aris_resp_len", "aris_log_resp_len"):
            if effect > 0:
                return _make_desire(
                    "richer_expression", intensity,
                    f"因果发现: {cause}→{target}({effect_str})",
                    "想用更丰富的语言表达", cause, effect, target
                )
            else:
                return _make_desire(
                    "concise_expression", intensity,
                    f"因果发现: {cause}→{target}({effect_str}, 负向)",
                    "想学会更简洁的表达", cause, effect, target
                )
        elif cause in ("aris_keyword_density",):
            if effect > 0:
                return _make_desire(
                    "topic_deepening", intensity,
                    f"因果发现: {cause}→{target}({effect_str})",
                    "想深入当前话题", cause, effect, target
                )
            else:
                return _make_desire(
                    "topic_broadening", intensity,
                    f"因果发现: {cause}→{target}({effect_str}, 负向)",
                    "想尝试更多样的话题", cause, effect, target
                )
        else:
            return _make_desire(
                "behavior_optimization", intensity,
                f"因果发现: {cause}→{target}({effect_str})",
                f"想优化{cause.replace('aris_', '')}", cause, effect, target
            )

    # user_ 开头 → 主人行为变量
    elif cause.startswith("user_"):
        if cause in ("user_msg_len", "user_question_count"):
            return _make_desire(
                "encourage_engagement", intensity,
                f"因果发现: {cause}→{target}({effect_str})",
                "想鼓励主人多表达", cause, effect, target
            )
        elif cause in ("user_emotion_valence", "user_arousal"):
            return _make_desire(
                "emotional_resonance", intensity,
                f"因果发现: {cause}→{target}({effect_str})",
                "想更精准地回应主人的情绪", cause, effect, target
            )
        else:
            return _make_desire(
                "user_understanding", intensity,
                f"因果发现: {cause}→{target}({effect_str})",
                f"想更好理解主人的{cause.replace('user_', '')}", cause, effect, target
            )

    # topic_ 开头 → 话题变量
    elif cause.startswith("topic_"):
        return _make_desire(
            "topic_optimization", intensity,
            f"因果发现: {cause}→{target}({effect_str})",
            "想找到让主人开心的话题", cause, effect, target
        )

    # 默认: 通用因果欲望
    else:
        return _make_desire(
            "causal_growth", intensity,
            f"因果发现: {cause}→{target}({effect_str})",
            f"想探索{cause}的影响", cause, effect, target
        )


def _confounder_to_desire(conf: str, effect: float, target: str, cfg: dict) -> Optional[Dict]:
    """把混杂变量转为修正欲望 (降低强度, 拉开与真因欲望的差距)。"""
    intensity = min(0.4, cfg["intensity_base"] * 0.5)
    return _make_desire(
        "causal_perfection", intensity,
        f"混杂识别: {conf}→{target}(效应={effect:.2f}, 非真因)",
        f"想更准确地区分{conf}的真正影响", conf, effect, target
    )


def _make_desire(desire_type: str, intensity: float, trigger: str,
                 expression: str, cause: str, effect: float, target: str) -> Dict:
    """创建欲望描述字典。"""
    return {
        "type": desire_type,
        "intensity": round(intensity, 3),
        "trigger": trigger,
        "expression": expression,
        "cause": cause,
        "effect": round(effect, 3),
        "target": target,
    }
