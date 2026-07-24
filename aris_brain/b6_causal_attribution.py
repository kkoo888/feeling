# -*- coding: utf-8 -*-
"""
B6 因果归因模块 — 进化基线 (相关性回退 + 动态目标)
=====================================================
从 aris_desire_engine.py 的 causal_attribution() 提取的独立模块。

核心能力: 给定一个观察到的效应(如 user_emotion_valence 下降),
找到真正的原因(干预验证有效), 剔除混杂变量(看起来相关但干预无效)。

进化策略:
  1. 动态目标发现 — 从因果图拓扑结构自动发现 sink 节点和关键词匹配节点
  2. 相关性回退 — 当因果图无直接父节点时, 用 Pearson 相关找到候选原因
  3. 干预验证 — 对候选原因做 intervene() 验证因果效应

参考:
  - Kelley 1967 共变法则: 原因和结果共同变化才是真正原因
  - Pearl 1995 backdoor criterion: 用干预效应区分真正原因和混杂
  - DoWhy identify_effect(): 自动识别可估的调整集
"""
import math
from typing import Dict, List, Optional, Any, Tuple


# ── 模块级状态 ──────────────────────────────────────────
_last_engine = None

# 关键词: 用于从因果图节点中识别可能的 outcome 变量
_KEYWORDS = (
    "emotion", "valence", "engagement", "satisfaction", "bond", "trust",
    "outcome", "result", "effect", "response", "feeling", "sentiment",
)


# ── 可进化参数 ──────────────────────────────────────────

def get_attribution_config() -> dict:
    """返回可进化的归因参数。

    进化系统会优化这些参数:
      intervention_threshold: 干预效应阈值, 超过则判为真正原因
      min_observations: 触发归因所需的最少观测数
      n_samples: 干预模拟的采样数
      alpha: 因果发现的显著性水平
      max_candidates: 候选原因的最大数量(防止爆炸)
      confidence_threshold: 归因结果置信度阈值
      correlation_top_k: 相关性回退时取 top-k 个候选
      correlation_min_abs: 相关性回退的最小绝对相关系数
    """
    return {
        "intervention_threshold": 0.15,
        "min_observations": 10,
        "n_samples": 30,
        "alpha": 0.05,
        "max_candidates": 15,
        "confidence_threshold": 0.5,
        "correlation_top_k": 5,
        "correlation_min_abs": 0.08,
    }


# ── 相关性回退辅助 ──────────────────────────────────────

def _pearson(x: list, y: list) -> float:
    """计算 Pearson 相关系数"""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / (dx * dy)))


def _correlation_fallback(ce, target: str, cfg: dict) -> list:
    """当因果图无直接父节点时, 用相关性找到候选原因。

    遍历所有观测变量, 计算与 target 的 Pearson 相关系数,
    取 top-k 个绝对相关系数最大的作为候选原因。
    """
    obs_list = getattr(ce, 'observations', [])
    if len(obs_list) < 3:
        return []

    # 提取 target 的时间序列
    tvals = [
        float(getattr(o, 'variables', {}).get(target, 0.0))
        for o in obs_list
    ]
    if len(tvals) < 3:
        return []

    # 收集所有其他变量
    all_vars = set()
    for o in obs_list:
        all_vars.update(getattr(o, 'variables', {}).keys())
    all_vars.discard(target)

    # 计算每个变量与 target 的相关系数
    scored = []
    for var in all_vars:
        vvals = [
            float(getattr(o, 'variables', {}).get(var, 0.0))
            for o in obs_list
        ]
        if len(vvals) < 3:
            continue
        r = _pearson(vvals, tvals)
        if abs(r) >= cfg.get("correlation_min_abs", 0.08):
            scored.append((var, abs(r)))

    scored.sort(key=lambda t: t[1], reverse=True)
    return [v for v, _ in scored[:cfg.get("correlation_top_k", 5)]]


# ── 动态目标发现 ────────────────────────────────────────

def _discover_outcomes(ce) -> list:
    """从因果图拓扑结构动态发现归因目标。

    策略:
      1. 找 sink 节点 (有入度但无出度) — 这些是终端效应
      2. 找含关键词的节点 — 这些是可能的 outcome 变量
      3. 如果太少, 补充高入度节点
      4. 最后回退到观测数据中的关键词变量
    """
    outcomes, seen = [], set()
    in_deg, out_deg, all_nodes = {}, {}, set()

    for ek in ce.graph.edges:
        parts = ek.split("->")
        if len(parts) == 2:
            s, t = parts[0].strip(), parts[1].strip()
            all_nodes.add(s)
            all_nodes.add(t)
            in_deg[t] = in_deg.get(t, 0) + 1
            out_deg[s] = out_deg.get(s, 0) + 1

    # 1. Sink 节点 (有入度但无出度)
    for n in all_nodes:
        if in_deg.get(n, 0) > 0 and out_deg.get(n, 0) == 0 and n not in seen:
            outcomes.append(n)
            seen.add(n)

    # 2. 含关键词的节点
    for n in all_nodes:
        if any(kw in n.lower() for kw in _KEYWORDS) and n not in seen:
            outcomes.append(n)
            seen.add(n)

    # 3. 补充高入度节点
    if len(outcomes) < 3:
        remaining = [n for n in all_nodes if n not in seen]
        remaining.sort(key=lambda n: in_deg.get(n, 0), reverse=True)
        for n in remaining[:3]:
            if in_deg.get(n, 0) > 0:
                outcomes.append(n)
                seen.add(n)

    # 4. 回退到观测数据中的关键词变量
    if not outcomes:
        for o in getattr(ce, 'observations', []):
            for var in getattr(o, 'variables', {}):
                if any(kw in var.lower() for kw in _KEYWORDS) and var not in seen:
                    outcomes.append(var)
                    seen.add(var)

    return outcomes


def get_attribution_targets() -> List[str]:
    """返回归因目标变量列表 — 动态从因果图拓扑发现。

    如果有 engine 引用 (_last_engine), 从因果图结构自动发现 sink 节点
    和关键词匹配节点。否则回退到硬编码列表。
    """
    global _last_engine
    if _last_engine is not None:
        try:
            t = _discover_outcomes(_last_engine)
            if t:
                return t
        except Exception:
            pass
    return [
        "user_emotion_valence",
        "user_engagement",
        "user_satisfaction",
        "bond_level",
        "trust_level",
    ]


# ── 核心归因函数 ──────────────────────────────────────────

def causal_attribution(ce, observed_effect: str) -> Dict:
    """因果归因: 找到观察效应的真正原因(剔除混杂变量)。

    算法:
      1. discover() 发现因果结构
      2. 找所有指向 observed_effect 的父节点(候选原因)
      3. 如果无父节点, 用相关性回退找候选原因
      4. 对每个候选原因做 intervene() 验证干预效应
      5. 干预效应 > 阈值 → 真正原因; 否则 → 混杂变量

    Args:
        ce: 因果引擎实例 (laap.agi.causal.CausalEngine)
        observed_effect: 观察到的效应变量名(如 "user_emotion_valence")

    Returns:
        {
            "observed_effect": str,
            "true_causes": List[str],    # 干预验证有效的真正原因
            "confounders": List[str],    # 混杂变量(看起来相关但干预无效)
            "effects": Dict[str, float], # 每个原因的干预效应值
            "confidence": float,         # 归因置信度 0-1
        }
    """
    global _last_engine
    _last_engine = ce  # 保存引擎引用供动态目标发现使用

    cfg = get_attribution_config()
    result = {
        "observed_effect": observed_effect,
        "true_causes": [],
        "confounders": [],
        "effects": {},
        "confidence": 0.0,
    }

    # 确保有足够的观测数据
    if len(getattr(ce, 'observations', [])) < cfg["min_observations"]:
        result["note"] = f"观测不足(需≥{cfg['min_observations']}条)"
        return result

    # 确保因果结构已发现
    if not ce.graph.edges:
        try:
            ce.discover(alpha=cfg["alpha"])
        except Exception:
            pass

    if not ce.graph.edges:
        result["note"] = "无法发现因果结构"
        return result

    # 找所有指向 observed_effect 的父节点(候选原因)
    parents = ce.graph.get_parents(observed_effect) or []
    if not parents:
        # 尝试找相关变量: 所有和 observed_effect 有边的变量
        for ek in ce.graph.edges:
            parts = ek.split("->")
            if len(parts) == 2:
                if parts[1] == observed_effect:
                    parents.append(parts[0])
                elif parts[0] == observed_effect:
                    parents.append(parts[1])

    # 相关性回退: 如果因果图无父节点, 用 Pearson 相关找候选
    if not parents:
        parents = _correlation_fallback(ce, observed_effect, cfg)

    if not parents:
        result["note"] = "无候选原因变量"
        return result

    # 限制候选数量
    if len(parents) > cfg["max_candidates"]:
        parents = parents[:cfg["max_candidates"]]

    true_causes = []
    confounders = []
    effects = {}

    for cause_var in parents:
        try:
            iv = ce.intervene(cause_var, 1.0, observed_effect,
                              n_samples=cfg["n_samples"])
            effect = iv.get("intervention_effect", 0)
            effects[cause_var] = round(effect, 3)

            if abs(effect) > cfg["intervention_threshold"]:
                # 干预效应显著 → 真正原因
                true_causes.append(cause_var)
            else:
                # 干预效应微弱 → 混杂变量
                confounders.append(cause_var)
        except Exception:
            confounders.append(cause_var)

    # 计算归因置信度
    total_candidates = len(true_causes) + len(confounders)
    if total_candidates > 0:
        # 置信度 = 真因比例 × 平均效应强度
        cause_ratio = len(true_causes) / total_candidates
        if true_causes:
            avg_effect = sum(abs(effects[c]) for c in true_causes) / len(true_causes)
            avg_effect_norm = min(1.0, avg_effect / 0.5)  # 归一化到 0-1
        else:
            avg_effect_norm = 0.0
        confidence = cause_ratio * 0.6 + avg_effect_norm * 0.4
    else:
        confidence = 0.0

    result["true_causes"] = true_causes
    result["confounders"] = confounders
    result["effects"] = effects
    result["confidence"] = round(confidence, 3)

    return result


def discover_new_desires_causal(ce) -> List[str]:
    """B6: 基于因果归因发现新欲望。

    用因果归因找出主人状态的真正原因, 生成精准欲望描述。

    Args:
        ce: 因果引擎实例

    Returns:
        新发现的欲望描述列表
    """
    new_desires = []

    # 对主人情绪做因果归因
    attribution = causal_attribution(ce, "user_emotion_valence")
    true_causes = attribution.get("true_causes", [])
    effects = attribution.get("effects", {})

    for cause in true_causes:
        effect = effects.get(cause, 0)

        # 根据 cause 变量类型生成不同欲望
        if cause in ("aris_initiative", "aris_resp_len"):
            # 小茜主动性 → 主人情绪好 → 强化主动性欲望
            if effect > 0:
                new_desires.append(
                    f"提升{cause.replace('aris_', '')} — 因果归因确认正向效应={effect:.2f}"
                )
        elif cause in ("empathy_level", "warmth_level"):
            # 共情 → 主人情绪好 → 强化连接欲
            if effect > 0:
                new_desires.append(
                    f"增强{cause.replace('_level', '')} — 因果效应={effect:.2f}"
                )
        elif cause in ("topic_code",):
            # 话题选择 → 主人情绪 → 优化话题
            new_desires.append(
                f"优化话题选择 — 因果效应={effect:.2f}"
            )

    return new_desires


# ── 批量归因 ────────────────────────────────────────────

def batch_attribution(ce, targets: Optional[List[str]] = None) -> Dict[str, Dict]:
    """对多个目标变量批量做因果归因。

    Args:
        ce: 因果引擎实例
        targets: 归因目标列表, None 则用 get_attribution_targets()

    Returns:
        {target: attribution_result}
    """
    if targets is None:
        targets = get_attribution_targets()

    results = {}
    for target in targets:
        results[target] = causal_attribution(ce, target)

    return results
