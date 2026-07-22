"""
UnifiedCausalEngine — 统一因果推理引擎（终极版）
==========================================

基于 Pearl 因果层级理论（关联/干预/反事实）+ 项目设计文档实现。

终极版整合的前沿技术:
1. PC 算法骨架学习 + v-structure 检测
2. Meek 规则 (R1-R4) — 方向传播到不动点
3. HSIC (Hilbert-Schmidt Independence Criterion) — 非线性残差独立性检验,升级 ANM
4. Doubly Robust (DR) 估计 + IPW — 干预效应双重稳健无偏估计
5. GES (Greedy Equivalence Search) — 基于评分的备选因果发现
6. DirectLiNGAM — 非高斯性迭代因果序确定
7. 因果中介分析 — NDE/NIE 分解
8. self_test() — 端到端自检
9. SHA-256 量子投影 — 跨进程可复现
10. 原子持久化 — 防崩溃数据损坏
11. predict() 输入验证 + 错误处理

不依赖外部因果库（DoWhy/CausalNex），纯 Python + 标准库 + networkx 实现。

设计依据:
- docs/feeling-advanced-design.md (CausalDiscovery 设计)
- docs/feeling-frontier-theories.md (Pearl 三层级 + DoWhy/CausalNex/CERMIC)
- docs/feeling-foundation-blocks-v2.md (方块 75 反事实 + 方块 76 因果图)
- aris_brain/agi_subscriber.py (调用契约: predict(query, mode, top_k))
- aris_brain/aris_cognitive_bridge.py (learn_bond / detect_transitive_chains 兼容)

参考:
- Pearl, J. (1995) Causal diagrams for empirical research
- Hoyer et al. (2009) Nonlinear causal discovery with additive noise models
- Shimizu et al. (2006) A linear non-gaussian acyclic model for causal discovery
- Robins et al. (1994) Estimation of regression coefficients when some regressors
  are not always observed (Doubly Robust)
- Chickering (2002) Optimal structure identification with greedy search (GES)
- Meek (1995) Causal inference and causal explanation
"""

import hashlib
import json
import logging
import math
import os
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

logger = logging.getLogger("laap.agi.causal")


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class Observation:
    """单次观测记录"""
    timestamp: float
    variables: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def get(self, key, default=None):
        return self.variables.get(key, self.context.get(key, default))


@dataclass
class CausalEdge:
    """因果边"""
    source: str
    target: str
    confidence: float = 0.0
    effect_size: float = 0.0
    edge_type: str = "unknown"
    n_observations: int = 0
    strength: float = 0.0  # 相关系数绝对值, discover() 自动填充


class CausalGraph:
    """因果图 — 有向无环图（DAG）"""

    def __init__(self):
        self.edges: Dict[str, CausalEdge] = {}
        self.nodes: Set[str] = set()
        if _HAS_NX:
            self._nx = nx.DiGraph()
        else:
            self._nx = None

    def add_node(self, name: str):
        self.nodes.add(name)
        if self._nx is not None:
            self._nx.add_node(name)

    def add_edge(self, parent: str, child: str, confidence: float = 0.5,
                 effect_size: float = 0.0, edge_type: str = "direct",
                 n_observations: int = 1, strength: float = 0.0):
        self.add_node(parent)
        self.add_node(child)
        key = f"{parent}->{child}"
        self.edges[key] = CausalEdge(
            source=parent, target=child, confidence=confidence,
            effect_size=effect_size, edge_type=edge_type,
            n_observations=n_observations, strength=strength,
        )
        if self._nx is not None:
            self._nx.add_edge(parent, child, confidence=confidence)
        if not self.is_dag():
            del self.edges[key]
            if self._nx is not None:
                self._nx.remove_edge(parent, child)
            logger.warning(f"拒绝添加 {parent}->{child}：会形成环")

    def remove_edge(self, parent: str, child: str):
        key = f"{parent}->{child}"
        self.edges.pop(key, None)
        if self._nx is not None:
            if self._nx.has_edge(parent, child):
                self._nx.remove_edge(parent, child)

    def is_dag(self) -> bool:
        if self._nx is not None:
            try:
                return nx.is_directed_acyclic_graph(self._nx)
            except Exception:
                pass
        visited, stack = set(), set()
        adj = defaultdict(list)
        for e in self.edges.values():
            adj[e.source].append(e.target)

        def has_cycle(node):
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for nxt in adj[node]:
                if has_cycle(nxt):
                    return True
            stack.remove(node)
            return False

        for n in self.nodes:
            if has_cycle(n):
                return False
        return True

    def get_parents(self, node: str) -> List[str]:
        return [e.source for e in self.edges.values() if e.target == node]

    def get_children(self, node: str) -> List[str]:
        return [e.target for e in self.edges.values() if e.source == node]

    def get_ancestors(self, node: str) -> List[str]:
        if self._nx is not None:
            try:
                return list(nx.ancestors(self._nx, node))
            except Exception:
                pass
        ancestors, queue = set(), deque(self.get_parents(node))
        while queue:
            n = queue.popleft()
            if n not in ancestors:
                ancestors.add(n)
                queue.extend(self.get_parents(n))
        return list(ancestors)

    def get_descendants(self, node: str) -> List[str]:
        if self._nx is not None:
            try:
                return list(nx.descendants(self._nx, node))
            except Exception:
                pass
        desc, queue = set(), deque(self.get_children(node))
        while queue:
            n = queue.popleft()
            if n not in desc:
                desc.add(n)
                queue.extend(self.get_children(n))
        return list(desc)

    def has_edge(self, a: str, b: str) -> bool:
        """检查 a→b 或 b→a 是否存在"""
        return f"{a}->{b}" in self.edges or f"{b}->{a}" in self.edges

    def to_dict(self) -> Dict:
        return {
            "nodes": sorted(self.nodes),
            "edges": {k: asdict(v) for k, v in self.edges.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CausalGraph":
        g = cls()
        for n in data.get("nodes", []):
            g.add_node(n)
        for k, e in data.get("edges", {}).items():
            g.add_edge(
                e["source"], e["target"], e.get("confidence", 0.5),
                e.get("effect_size", 0.0), e.get("edge_type", "direct"),
                e.get("n_observations", 1),
            )
        return g


# ═══════════════════════════════════════════════════════════════
# 统计工具（不依赖 scipy/numpy）
# ═══════════════════════════════════════════════════════════════

def _safe_corr(xs: List[float], ys: List[float]) -> float:
    """安全的皮尔逊相关系数"""
    if len(xs) < 2 or len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return 0.0
    try:
        return statistics.correlation(xs, ys)
    except (ValueError, statistics.StatisticsError):
        return 0.0


def _skewness(xs: List[float]) -> float:
    """样本偏度（Fisher-Pearson 偏度校正 G1）。

    G1 = sqrt(n*(n-1)) / (n-2) * g1
    其中 g1 = m3 / s^3 是有偏估计。
    匹配 scipy.stats.skew(bias=False) 和 Excel SKEW 函数。
    """
    n = len(xs)
    if n < 3:
        return 0.0
    m = statistics.fmean(xs)
    s = statistics.stdev(xs)
    if s == 0:
        return 0.0
    g1 = statistics.fmean([((x - m) / s) ** 3 for x in xs])
    # Fisher-Pearson 偏度校正
    correction = math.sqrt(n * (n - 1)) / (n - 2)
    return g1 * correction


def _kurtosis(xs: List[float]) -> float:
    """超额峰度（excess kurtosis），衡量分布的尖峰/厚尾程度。"""
    n = len(xs)
    if n < 4:
        return 0.0
    m = statistics.fmean(xs)
    s = statistics.stdev(xs)
    if s == 0:
        return 0.0
    g2 = statistics.fmean([((x - m) / s) ** 4 for x in xs]) - 3.0
    return g2


def _linear_fit(xs: List[float], ys: List[float]) -> Tuple[float, float, List[float]]:
    """拟合 Y = a*X + b，返回 (a, b, residuals)。"""
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    x_var = statistics.variance(xs)
    if x_var == 0:
        return 0.0, y_mean, [y - y_mean for y in ys]
    a = statistics.covariance(xs, ys) / x_var
    b = y_mean - a * x_mean
    residuals = [ys[i] - (a * xs[i] + b) for i in range(len(xs))]
    return a, b, residuals


def _least_squares(X: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """最小二乘法（正规方程 + 高斯消元 + 主元选取）。"""
    n = len(X)
    if n == 0:
        return None
    p = len(X[0])
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        for j in range(p):
            Xty[j] += X[i][j] * y[i]
            for k in range(p):
                XtX[j][k] += X[i][j] * X[i][k]
    aug = [XtX[i] + [Xty[i]] for i in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(col + 1, p):
            factor = aug[r][col] / aug[col][col]
            for c in range(col, p + 1):
                aug[r][c] -= factor * aug[col][c]
    beta = [0.0] * p
    for i in range(p - 1, -1, -1):
        beta[i] = aug[i][p]
        for j in range(i + 1, p):
            beta[i] -= aug[i][j] * beta[j]
        beta[i] /= aug[i][i]
    return beta


def _conditional_independence_test(
    observations: List[Observation],
    x: str, y: str, z_set: List[str],
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """条件独立性测试（偏相关 Fisher z-transform）。

    给定 Z 集合，测试 X 和 Y 是否条件独立。
    使用偏相关：先回归 X~Z 和 Y~Z，取残差，测残差相关性。
    """
    xs = [o.get(x) for o in observations if o.get(x) is not None]
    ys = [o.get(y) for o in observations if o.get(y) is not None]
    if len(xs) < 5 or len(ys) < 5:
        return True, 1.0

    paired_full = [(float(o.get(x)), float(o.get(y))) for o in observations
                   if o.get(x) is not None and o.get(y) is not None]
    if len(paired_full) < 5:
        return True, 1.0

    if not z_set:
        if len(set([p[0] for p in paired_full])) <= 1 or len(set([p[1] for p in paired_full])) <= 1:
            return True, 1.0
        try:
            corr = statistics.correlation([p[0] for p in paired_full], [p[1] for p in paired_full])
            n = len(paired_full)
            z = 0.5 * math.log((1 + corr) / (1 - corr + 1e-10)) * math.sqrt(max(1, n - 3))
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
            return p > alpha, p
        except (ValueError, statistics.StatisticsError):
            return True, 1.0
    else:
        # 偏相关检验：回归 X~Z 和 Y~Z，取残差相关性
        rows, xs_full, ys_full = [], [], []
        for o in observations:
            xv, yv = o.get(x), o.get(y)
            if xv is None or yv is None:
                continue
            zvals = [o.get(z) for z in z_set]
            if any(zv is None for zv in zvals):
                continue
            rows.append([float(zv) for zv in zvals])
            xs_full.append(float(xv))
            ys_full.append(float(yv))

        if len(rows) < len(z_set) + 5:
            # 数据不足，回退到分层中位数检验
            return _ci_test_stratified(observations, x, y, z_set, paired_full, alpha)

        # 回归 X~Z，取残差
        X_z = [[1.0] + row for row in rows]
        beta_x = _least_squares(X_z, xs_full)
        beta_y = _least_squares(X_z, ys_full)
        if beta_x is None or beta_y is None:
            return _ci_test_stratified(observations, x, y, z_set, paired_full, alpha)

        res_x = [xs_full[i] - sum(beta_x[j] * X_z[i][j] for j in range(len(beta_x)))
                 for i in range(len(xs_full))]
        res_y = [ys_full[i] - sum(beta_y[j] * X_z[i][j] for j in range(len(beta_y)))
                 for i in range(len(ys_full))]

        partial_corr = _safe_corr(res_x, res_y)
        n = len(res_x)
        # Fisher z-transform
        if abs(partial_corr) >= 0.999:
            partial_corr = 0.999 * (1 if partial_corr > 0 else -1)
        z_val = 0.5 * math.log((1 + partial_corr) / (1 - partial_corr + 1e-10)) * math.sqrt(max(1, n - 3 - len(z_set)))
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_val) / math.sqrt(2))))
        return p_value > alpha, p_value


def _ci_test_stratified(observations, x, y, z_set, paired_full, alpha):
    """分层中位数条件独立性检验（回退方法）"""
    stratum_corrs = []
    for z_var in z_set:
        z_vals = [o.get(z_var) for o in observations if o.get(z_var) is not None]
        if len(z_vals) < 5:
            continue
        try:
            z_median = statistics.median([float(v) for v in z_vals])
        except (ValueError, TypeError):
            continue
        for is_low in (True, False):
            subset = [o for o in observations
                      if o.get(z_var) is not None and o.get(x) is not None and o.get(y) is not None
                      and (float(o.get(z_var)) <= z_median) == is_low]
            if len(subset) < 5:
                continue
            xs_sub = [float(o.get(x)) for o in subset]
            ys_sub = [float(o.get(y)) for o in subset]
            try:
                if len(set(xs_sub)) > 1 and len(set(ys_sub)) > 1:
                    corr = statistics.correlation(xs_sub, ys_sub)
                    stratum_corrs.append(abs(corr))
            except (ValueError, statistics.StatisticsError):
                pass
    if not stratum_corrs:
        return True, 1.0
    avg_corr = statistics.mean(stratum_corrs)
    full_corr = abs(_safe_corr([p[0] for p in paired_full], [p[1] for p in paired_full]))
    is_independent = avg_corr < 0.15 or avg_corr < full_corr * 0.4
    return is_independent, 1.0 - avg_corr


# ═══════════════════════════════════════════════════════════════
# HSIC (Hilbert-Schmidt Independence Criterion) — 非线性独立性检验
# ═══════════════════════════════════════════════════════════════

def _median_bandwidth(xs: List[float]) -> float:
    """中位数启发式带宽 (median heuristic for RBF kernel)"""
    n = len(xs)
    if n < 2:
        return 1.0
    # 采样以避免 O(n^2) 配对（n>500 时）
    if n > 500:
        rng = random.Random(42)
        sampled = rng.sample(xs, 500)
    else:
        sampled = xs
    diffs = []
    m = len(sampled)
    for i in range(m):
        for j in range(i + 1, m):
            diffs.append(abs(sampled[i] - sampled[j]))
    if not diffs:
        return 1.0
    med = statistics.median(diffs)
    return max(med, 1e-6)


def _rbf_gram(xs: List[float], sigma: float) -> List[List[float]]:
    """RBF 核 Gram 矩阵 K[i][j] = exp(-|x_i - x_j|^2 / (2*sigma^2))"""
    n = len(xs)
    two_sigma2 = 2.0 * sigma * sigma
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        K[i][i] = 1.0
        for j in range(i + 1, n):
            d2 = (xs[i] - xs[j]) ** 2
            val = math.exp(-d2 / two_sigma2) if d2 > 0 else 1.0
            K[i][j] = val
            K[j][i] = val
    return K


def _hsic(xs: List[float], ys: List[float]) -> float:
    """
    HSIC (biased estimator): HSIC = (1/n^2) * trace(K H L H)
    其中 H = I - 11^T/n 是中心化矩阵。
    用于检测任意（线性+非线性）依赖关系。
    """
    n = len(xs)
    if n < 10:
        return 0.0
    # 采样到 n<=500 以控制内存和运行时
    if n > 500:
        rng = random.Random(42)
        idx = sorted(rng.sample(range(n), 500))
        xs = [xs[i] for i in idx]
        ys = [ys[i] for i in idx]
        n = 500

    sigma_x = _median_bandwidth(xs)
    sigma_y = _median_bandwidth(ys)

    K = _rbf_gram(xs, sigma_x)
    L = _rbf_gram(ys, sigma_y)

    # 中心化: K_c = H K H, 其中 H = I - 11^T/n
    # trace(K_c L_c) = trace(H K H L) = trace(K H L H)  (trace cyclic)
    # 直接计算避免显式构造 H
    # (K H L H)[i][j] = K[i][j] - rowmean(K)[j] - colmean(L)[i] + grandmean
    # 简化: trace(K H L H) = sum over i,j of K_c[i][j] * L_c[i][j]
    row_mean_K = [statistics.fmean(K[i]) for i in range(n)]
    col_mean_K = [statistics.fmean([K[i][j] for i in range(n)]) for j in range(n)]
    grand_mean_K = statistics.fmean(row_mean_K)

    row_mean_L = [statistics.fmean(L[i]) for i in range(n)]
    col_mean_L = [statistics.fmean([L[i][j] for i in range(n)]) for j in range(n)]
    grand_mean_L = statistics.fmean(row_mean_L)

    hsic_val = 0.0
    for i in range(n):
        for j in range(n):
            kc = K[i][j] - row_mean_K[i] - col_mean_K[j] + grand_mean_K
            lc = L[i][j] - row_mean_L[i] - col_mean_L[j] + grand_mean_L
            hsic_val += kc * lc
    hsic_val /= (n * n)
    return max(0.0, hsic_val)


# ═══════════════════════════════════════════════════════════════
# 主引擎
# ═══════════════════════════════════════════════════════════════

class UnifiedCausalEngine:
    """
    统一因果推理引擎（终极版）

    实现 Pearl 三层级因果推理:
    - Level 1 关联: P(Y|X)
    - Level 2 干预: P(Y|do(X))
    - Level 3 反事实: P(Y_x|X', Y')

    整合的前沿技术:
    - Meek 规则: PC 算法方向传播到不动点
    - HSIC: 非线性残差独立性检验,升级 ANM 方向评分
    - Doubly Robust: 双重稳健干预效应估计
    - GES: 基于评分的备选因果发现
    - DirectLiNGAM: 非高斯性因果序确定
    - 中介分析: NDE/NIE 分解

    用法:
        engine = UnifiedCausalEngine(quantum_dim=64, name="小茜Causal")
        engine.observe({"sleep": 5, "mood": 0.3})
        engine.discover()
        causes = engine.query_cause("mood")
        cf = engine.counterfactual(5, 0.3, 8)
    """

    def __init__(self, quantum_dim: int = 64, name: str = "CausalEngine",
                 persist_path: Optional[str] = None):
        self.quantum_dim = quantum_dim
        self.name = name
        self.persist_path = Path(persist_path) if persist_path else None

        self.observations: List[Observation] = []
        self._max_observations = 2000

        self.graph = CausalGraph()
        self._confounders: Set[str] = set()

        self._projection_seed = 42
        self._projection_cache: Dict[str, List[float]] = {}

        self._inferences = 0
        self._counterfactuals = 0
        self._discoveries = 0

        if self.persist_path and self.persist_path.exists():
            self._load()

        logger.info(f"[{self.name}] 因果引擎初始化 (quantum_dim={quantum_dim})")

    # ── 观测 ──────────────────────────────────────────────

    def observe(self, event: Union[Dict, Observation]):
        """记录一次观测"""
        if isinstance(event, dict):
            if "variables" in event or "context" in event:
                variables = event.get("variables", {})
                context = event.get("context", {})
            else:
                variables = {k: v for k, v in event.items()
                             if k not in ("weight",) and not isinstance(v, dict)}
                context = {}
            event = Observation(
                timestamp=time.time(),
                variables=variables,
                context=context,
                weight=event.get("weight", 1.0) if isinstance(event.get("weight"), (int, float)) else 1.0,
            )
        self.observations.append(event)
        if len(self.observations) > self._max_observations:
            self.observations = self.observations[-self._max_observations:]
        for var in event.variables:
            if var not in self.graph.nodes:
                self.graph.add_node(var)

    def observe_many(self, events: List[Dict]):
        """批量记录观测"""
        for e in events:
            self.observe(e)

    # ── 因果发现（PC 算法 + Meek 规则 + 可选 GES/LiNGAM）──

    def discover(self, alpha: float = 0.05,
                 temporal_order: Optional[List[str]] = None,
                 use_meek: bool = True,
                 use_ges: bool = False,
                 use_lingam: bool = False) -> Dict:
        """
        从观测数据自动发现因果图

        基于 PC 算法:
        1. 骨架学习：条件独立性测试删除非因果边
        2. v-structure 检测
        3. Meek 规则 (R1-R4) 传播方向到不动点
        4. ANM-HSIC 评分定向剩余无向边
        5. 可选: GES 备选发现 + DirectLiNGAM 因果序
        6. 效应量估计 + 混杂识别

        Args:
            alpha: 条件独立性检验显著性水平
            temporal_order: 时序顺序（早→晚），优先用于定向
            use_meek: 是否应用 Meek 规则（默认 True，建议保持）
            use_ges: 是否运行 GES 备选发现并合并结果（默认 False）
            use_lingam: 是否使用 DirectLiNGAM 确定因果序（默认 False）

        Returns:
            {"edges_discovered", "nodes", "confounders", "v_structures",
             "meek_applied", "direction_method", "details"}
        """
        if len(self.observations) < 10:
            return {"edges_discovered": 0, "reason": "观测不足（< 10）", "confounders": []}

        nodes = list(self.graph.nodes)
        if len(nodes) < 2:
            return {"edges_discovered": 0, "reason": "变量不足", "confounders": []}

        # Step 1+2: 骨架学习
        skeleton = defaultdict(set)
        removed_by = {}

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                x, y = nodes[i], nodes[j]
                is_ind, p = _conditional_independence_test(
                    self.observations, x, y, [], alpha
                )
                if is_ind:
                    continue
                # 无条件相关强度(用于保守性检查)
                paired = [(float(o.get(x)), float(o.get(y))) for o in self.observations
                          if o.get(x) is not None and o.get(y) is not None]
                unc_corr = _safe_corr([a for a, _ in paired], [b for _, b in paired])
                other_nodes = [n for n in nodes if n not in (x, y)]
                confounded = False
                for z in other_nodes[:8]:
                    is_cond_ind, _ = _conditional_independence_test(
                        self.observations, x, y, [z], alpha
                    )
                    if is_cond_ind:
                        # 保守性检查: 无条件强相关时不删除边
                        # 防止 collider 抵消导致的假阴性
                        # (条件化 collider X→Z←Y 会诱发抵消, 见 Spirtes et al. 2000 §3.4)
                        if abs(unc_corr) > 0.5:
                            continue
                        self._confounders.add(z)
                        removed_by[(x, y)] = z
                        confounded = True
                        break
                if not confounded:
                    skeleton[x].add(y)
                    skeleton[y].add(x)

        # Step 3: v-structure 检测
        self.graph.edges.clear()
        if self.graph._nx is not None:
            self.graph._nx.clear_edges()

        directed: Set[Tuple[str, str]] = set()
        v_structures = 0

        for (x, y), z in removed_by.items():
            if y in skeleton[z] and x in skeleton[z]:
                directed.add((x, z))
                directed.add((y, z))
                v_structures += 1

        # Step 3b: Meek 规则传播方向
        meek_applied = 0
        if use_meek:
            directed, meek_count = self._meek_rules(skeleton, directed)
            meek_applied = meek_count

        # Step 4: 剩余无向边定向（ANM-HSIC 或时序先验）
        temporal_rank = None
        if temporal_order:
            temporal_rank = {v: i for i, v in enumerate(temporal_order)}

        direction_method = "anm"
        for x in list(skeleton.keys()):
            for y in list(skeleton[x]):
                if (x, y) in directed or (y, x) in directed:
                    continue
                if temporal_rank is not None and x in temporal_rank and y in temporal_rank:
                    if temporal_rank[x] <= temporal_rank[y]:
                        parent, child = x, y
                    else:
                        parent, child = y, x
                    direction_method = "temporal"
                else:
                    score = self._anm_direction_score(x, y)
                    if score >= 0:
                        parent, child = x, y
                    else:
                        parent, child = y, x
                if (child, parent) not in directed:
                    directed.add((parent, child))

        # Step 4b: 可选 DirectLiNGAM 因果序
        lingam_order = None
        if use_lingam:
            lingam_order = self._directlingam_order(nodes)
            if lingam_order:
                rank = {v: i for i, v in enumerate(lingam_order)}
                for x in list(skeleton.keys()):
                    for y in list(skeleton[x]):
                        if (x, y) in directed or (y, x) in directed:
                            continue
                        if x in rank and y in rank:
                            if rank[x] < rank[y]:
                                directed.add((x, y))
                            else:
                                directed.add((y, x))

        # Step 4c: 可选 GES 备选发现
        ges_edges = None
        if use_ges:
            ges_edges = self._ges_discover(nodes)
            directed = self._merge_pc_ges(directed, ges_edges)

        # 应用定向边
        for (parent, child) in directed:
            effect_size = self._estimate_effect(parent, child)
            confidence = 0.9
            # 填充 strength: 用相关系数绝对值
            _xs = [float(o.get(parent, 0)) for o in self.observations if o.get(parent) is not None]
            _ys = [float(o.get(child, 0)) for o in self.observations if o.get(child) is not None]
            _corr = abs(_safe_corr(_xs, _ys))
            self.graph.add_edge(
                parent, child, confidence=confidence,
                effect_size=effect_size, edge_type="direct",
                n_observations=len(self.observations),
                strength=_corr,
            )

        # Step 5: 混杂变量识别
        node_list = list(self.graph.nodes)
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                x, y = node_list[i], node_list[j]
                px = set(self.graph.get_parents(x))
                py = set(self.graph.get_parents(y))
                common = px & py
                for z in common:
                    if z not in self._confounders:
                        self._confounders.add(z)
                        for e_key in (f"{z}->{x}", f"{z}->{y}"):
                            e = self.graph.edges.get(e_key)
                            if e:
                                e.edge_type = "confounded"

        self._discoveries += 1
        result = {
            "edges_discovered": len(self.graph.edges),
            "nodes": len(self.graph.nodes),
            "confounders": list(self._confounders),
            "v_structures": v_structures,
            "meek_applied": meek_applied,
            "direction_method": direction_method,
            "lingam_order": lingam_order,
            "ges_used": ges_edges is not None,
            "details": [
                {"edge": k, **asdict(v)}
                for k, v in self.graph.edges.items()
            ],
        }
        logger.info(f"[{self.name}] 发现 {len(self.graph.edges)} 条因果关系 "
                    f"(混杂: {len(self._confounders)}, v-struct: {v_structures}, "
                    f"meek: {meek_applied})")
        if self.persist_path:
            self._save()
        return result

    # ── Meek 规则 (R1-R4) ────────────────────────────────

    def _meek_rules(self, skeleton: Dict[str, Set[str]],
                    directed: Set[Tuple[str, str]]
                    ) -> Tuple[Set[Tuple[str, str]], int]:
        """
        应用 Meek 规则 R1-R4 传播方向到不动点。

        R1: X→Y-Z 且 X,Z 不相邻 → Y→Z (避免产生新 collider)
        R2: X→Z→Y 且 X-Y → X→Y (避免环)
        R3: X-Y, X-Z, X-W, Z→Y, W→Y, Z-W(无向) → X→Y
        R4: X-Y, X-Z(或X→Z), X-W, W→Z(或W-Z), W→Y → X→Y

        Args:
            skeleton: 无向骨架 {node: {neighbors}}
            directed: 已定向的边集合 {(parent, child)}

        Returns:
            (closure_directed, rules_applied_count)
        """
        directed = set(directed)
        rules_applied = 0
        max_iterations = 2 * (sum(len(v) for v in skeleton.values()) + 1)

        def is_directed(a, b):
            return (a, b) in directed

        def is_undirected(a, b):
            return (not is_directed(a, b) and not is_directed(b, a)
                    and b in skeleton.get(a, set()))

        def adjacent(a, b):
            return b in skeleton.get(a, set())

        for _ in range(max_iterations):
            changed = False

            # R1: X→Y-Z 且 X,Z 不相邻 → Y→Z
            for (x, y) in list(directed):
                for z in list(skeleton.get(y, set())):
                    if is_undirected(y, z) and not adjacent(x, z):
                        if (y, z) not in directed and (z, y) not in directed:
                            directed.add((y, z))
                            rules_applied += 1
                            changed = True

            # R2: X→Z→Y 且 X-Y → X→Y
            for (x, z) in list(directed):
                for (z2, y) in list(directed):
                    if z == z2 and is_undirected(x, y):
                        if (x, y) not in directed and (y, x) not in directed:
                            directed.add((x, y))
                            rules_applied += 1
                            changed = True

            # R3: X-Y, X-Z, X-W, Z→Y, W→Y, Z-W(无向) → X→Y
            for x in list(skeleton.keys()):
                for y in list(skeleton.get(x, set())):
                    if not is_undirected(x, y):
                        continue
                    # 找 Z, W 满足条件
                    for z in list(skeleton.get(x, set())):
                        if z == y or not is_undirected(x, z):
                            continue
                        if not is_directed(z, y):
                            continue
                        for w in list(skeleton.get(x, set())):
                            if w == y or w == z or not is_undirected(x, w):
                                continue
                            if not is_directed(w, y):
                                continue
                            if is_undirected(z, w):
                                if (x, y) not in directed and (y, x) not in directed:
                                    directed.add((x, y))
                                    rules_applied += 1
                                    changed = True
                                    break

            # R4: X-Y, X-Z 或 X→Z, X-W, W→Z 或 W-Z, W→Y → X→Y
            # 简化版: 检查链 X-Y, X-W, W→Z, Z→Y
            for x in list(skeleton.keys()):
                for y in list(skeleton.get(x, set())):
                    if not is_undirected(x, y):
                        continue
                    for w in list(skeleton.get(x, set())):
                        if w == y:
                            continue
                        if not (is_undirected(x, w) or is_directed(x, w)):
                            continue
                        for z in list(skeleton.get(w, set())):
                            if z == x or z == y:
                                continue
                            if not (is_directed(w, z) or is_undirected(w, z)):
                                continue
                            if is_directed(z, y):
                                if (x, y) not in directed and (y, x) not in directed:
                                    directed.add((x, y))
                                    rules_applied += 1
                                    changed = True
                                    break

            if not changed:
                break

        # 安全检查: 确保无环
        temp_graph = CausalGraph()
        for n in self.graph.nodes:
            temp_graph.add_node(n)
        for (p, c) in directed:
            if not temp_graph.is_dag():
                break
            temp_graph.add_edge(p, c, confidence=0.5)
        if not temp_graph.is_dag():
            # 移除最后添加的可能成环的边
            logger.warning("Meek 规则后检测到环，回退部分方向")
            valid_directed = set()
            temp_graph2 = CausalGraph()
            for n in self.graph.nodes:
                temp_graph2.add_node(n)
            for (p, c) in directed:
                temp_graph2.add_edge(p, c, confidence=0.5)
                if temp_graph2.is_dag():
                    valid_directed.add((p, c))
                else:
                    temp_graph2.remove_edge(p, c)
            directed = valid_directed

        return directed, rules_applied

    # ── HSIC-ANM 方向评分 ────────────────────────────────

    def _hsic_anm_score(self, x: str, y: str) -> float:
        """
        基于 HSIC 的 ANM 方向评分。

        正值 → X→Y (X→Y 方向残差更独立)
        负值 → Y→X

        比较:
        - HSIC(X, res_Y|X): X→Y 方向，Y 对 X 回归的残差与 X 的依赖
        - HSIC(Y, res_X|Y): Y→X 方向
        正确方向残差更独立 → HSIC 更小
        """
        paired = [
            (float(o.get(x)), float(o.get(y)))
            for o in self.observations
            if o.get(x) is not None and o.get(y) is not None
        ]
        if len(paired) < 30:
            return 0.0  # HSIC 在小样本上不稳定

        xs = [p[0] for p in paired]
        ys = [p[1] for p in paired]

        # X→Y: 残差 res = Y - (aX + b)
        _, _, res_y = _linear_fit(xs, ys)
        # Y→X: 残差 res = X - (aY + b)
        _, _, res_x = _linear_fit(ys, xs)

        hsic_xy = _hsic(xs, res_y)  # X→Y 方向依赖
        hsic_yx = _hsic(ys, res_x)  # Y→X 方向依赖

        # 正确方向 HSIC 更小 → 正值表示倾向 X→Y
        return hsic_yx - hsic_xy

    def _anm_direction_score(self, x: str, y: str) -> float:
        """
        ANM 方向评分：正值倾向于 X→Y，负值倾向于 Y→X。

        综合信号:
        1. HSIC-ANM (n>=30) 或线性残差独立性 (n<30)
        2. 非高斯性差异 (LiNGAM)
        3. 方差比启发式
        """
        paired = [
            (float(o.get(x)), float(o.get(y)))
            for o in self.observations
            if o.get(x) is not None and o.get(y) is not None
        ]
        if len(paired) < 5:
            return 0.0
        xs = [p[0] for p in paired]
        ys = [p[1] for p in paired]

        x_var = statistics.variance(xs)
        y_var = statistics.variance(ys)
        if x_var == 0 or y_var == 0:
            return 0.0

        # 1. ANM 残差独立性（HSIC 或线性回退）
        if len(paired) >= 30:
            anm_score = self._hsic_anm_score(x, y)
            # 归一化 HSIC 到与线性评分相近的尺度
            total_var = x_var + y_var
            anm_score = anm_score / (total_var * 0.01 + 1e-10)
        else:
            # 小样本: 线性残差独立性
            _, _, res_y = _linear_fit(xs, ys)
            indep_xy = abs(_safe_corr(xs, res_y))
            _, _, res_x = _linear_fit(ys, xs)
            indep_yx = abs(_safe_corr(ys, res_x))
            anm_score = indep_yx - indep_xy

        # 2. 非高斯性：原因偏度更大
        skew_x = abs(_skewness(xs))
        skew_y = abs(_skewness(ys))
        nongauss_score = skew_x - skew_y

        # 3. 方差比
        var_score = math.log(x_var / y_var + 1e-10)

        total = anm_score * 2.0 + nongauss_score * 0.5 + var_score * 0.2
        return total

    # ── GES (Greedy Equivalence Search) ──────────────────

    def _ges_discover(self, nodes: List[str]) -> Set[Tuple[str, str]]:
        """
        GES 因果发现: 基于评分的贪心搜索。

        1. 前向搜索: 从空图开始，逐步添加最大化 BIC 改善的边
        2. 后向搜索: 从前向结果开始，逐步删除改善 BIC 的边

        Returns:
            定向边集合 {(parent, child)}
        """
        if len(nodes) < 2 or len(self.observations) < 20:
            return set()

        # 限制候选父节点以控制复杂度
        max_parents = min(8, len(nodes) - 1)

        # 当前图的父节点集合
        parents: Dict[str, Set[str]] = {n: set() for n in nodes}
        edges: Set[Tuple[str, str]] = set()

        # 缓存 BIC 评分
        bic_cache: Dict[Tuple[str, frozenset], float] = {}

        def get_bic(target: str, par: Set[str]) -> float:
            key = (target, frozenset(par))
            if key not in bic_cache:
                bic_cache[key] = self._bic_score(par, target)
            return bic_cache[key]

        # 前向搜索
        improved = True
        while improved and len(edges) < len(nodes) * max_parents:
            improved = False
            best_delta = 1e-3
            best_edge = None

            for target in nodes:
                current_parents = parents[target]
                if len(current_parents) >= max_parents:
                    continue
                current_bic = get_bic(target, current_parents)

                for candidate in nodes:
                    if candidate == target or candidate in current_parents:
                        continue
                    # 检查是否成环
                    if self._would_create_cycle(edges, candidate, target):
                        continue
                    new_parents = current_parents | {candidate}
                    new_bic = get_bic(target, new_parents)
                    delta = new_bic - current_bic
                    if delta > best_delta:
                        best_delta = delta
                        best_edge = (candidate, target)

            if best_edge:
                edges.add(best_edge)
                parents[best_edge[1]].add(best_edge[0])
                improved = True

        # 后向搜索
        improved = True
        while improved:
            improved = False
            best_delta = 1e-3
            best_remove = None

            for (parent, target) in list(edges):
                current_parents = parents[target]
                current_bic = get_bic(target, current_parents)
                new_parents = current_parents - {parent}
                new_bic = get_bic(target, new_parents)
                delta = new_bic - current_bic
                if delta > best_delta:
                    best_delta = delta
                    best_remove = (parent, target)

            if best_remove:
                edges.discard(best_remove)
                parents[best_remove[1]].discard(best_remove[0])
                improved = True

        return edges

    def _would_create_cycle(self, edges: Set[Tuple[str, str]],
                           parent: str, child: str) -> bool:
        """检查添加 parent→child 是否会形成环"""
        # BFS: 从 child 出发能否到达 parent
        visited = {child}
        queue = deque([child])
        adj = defaultdict(list)
        for (p, c) in edges:
            adj[p].append(c)
        while queue:
            node = queue.popleft()
            if node == parent:
                return True
            for nxt in adj[node]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def _merge_pc_ges(self, pc_edges: Set[Tuple[str, str]],
                      ges_edges: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
        """
        合并 PC 和 GES 发现的边。

        - 两者都发现的边: 高置信度 (0.95)
        - 仅一方发现的边: 低置信度 (0.7)
        - 方向冲突 (X→Y vs Y→X): ANM 评分决定
        """
        merged = set(pc_edges)
        for (p, c) in ges_edges:
            if (c, p) in merged:
                # 方向冲突，用 ANM 评分决定
                score = self._anm_direction_score(p, c)
                if score < 0:
                    merged.discard((c, p))
                    merged.add((p, c))
                # 否则保持 PC 的方向
            elif (p, c) not in merged:
                merged.add((p, c))
        return merged

    # ── DirectLiNGAM ─────────────────────────────────────

    def _directlingam_order(self, nodes: List[str]) -> List[str]:
        """
        DirectLiNGAM 因果序确定。

        迭代地找到最外生的变量（与其他变量残差最独立），
        然后从其他变量中去除它的影响（残差化），重复。

        适用于线性非高斯数据。若数据接近高斯，返回空列表。

        Returns:
            因果序（从因到果），若数据不适合返回 []
        """
        if len(nodes) < 2 or len(self.observations) < 30:
            return []

        # 检查非高斯性（LiNGAM 要求非高斯噪声）
        nongauss_count = 0
        for n in nodes:
            vals = [float(o.get(n)) for o in self.observations if o.get(n) is not None]
            if len(vals) < 30:
                continue
            # 用偏度+峰度判断非高斯性
            sk = abs(_skewness(vals))
            kt = abs(_kurtosis(vals))
            if sk > 0.3 or kt > 1.0:
                nongauss_count += 1

        if nongauss_count < len(nodes) * 0.3:
            # 数据过于接近高斯，LiNGAM 不适用
            return []

        # 提取数据矩阵
        data = {}
        for n in nodes:
            data[n] = [float(o.get(n)) if o.get(n) is not None else 0.0
                       for o in self.observations]

        remaining = list(nodes)
        order = []

        for _ in range(len(nodes)):
            if len(remaining) <= 1:
                order.extend(remaining)
                break

            # 对每个候选变量 x，计算它与其他变量残差的独立性
            scores = {}
            for x in remaining:
                total_dep = 0.0
                for y in remaining:
                    if x == y:
                        continue
                    xs = data[x]
                    ys = data[y]
                    # 回归 y ~ x，取残差
                    _, _, res = _linear_fit(xs, ys)
                    # 残差与 x 的相关性（越小越独立）
                    dep = abs(_safe_corr(xs, res))
                    total_dep += dep
                scores[x] = total_dep

            # 选择最外生（独立性最高，dep 最低）的变量
            best = min(scores, key=lambda k: scores[k])
            order.append(best)
            remaining.remove(best)

            # 残差化: 从其他变量中去除 best 的影响
            for y in remaining:
                a, b, _ = _linear_fit(data[best], data[y])
                data[y] = [data[y][i] - (a * data[best][i] + b)
                           for i in range(len(data[y]))]

        return order

    # ── 效应估计 ────────────────────────────────────────

    def _estimate_effect(self, cause: str, effect: str) -> float:
        """估计 cause 对 effect 的效应量（皮尔逊相关近似）"""
        paired = [
            (o.get(cause), o.get(effect))
            for o in self.observations
            if o.get(cause) is not None and o.get(effect) is not None
        ]
        if len(paired) < 5:
            return 0.0
        try:
            return statistics.correlation(
                [float(p[0]) for p in paired],
                [float(p[1]) for p in paired],
            )
        except (ValueError, statistics.StatisticsError):
            return 0.0

    def _r_squared(self, cause: str, effect: str) -> float:
        """计算 cause 对 effect 的 R^2 (决定系数)"""
        paired = [(float(o.get(cause)), float(o.get(effect)))
                  for o in self.observations
                  if o.get(cause) is not None and o.get(effect) is not None]
        if len(paired) < 3:
            return 0.0
        xs = [p[0] for p in paired]
        ys = [p[1] for p in paired]
        _, _, residuals = _linear_fit(xs, ys)
        ss_res = sum(r ** 2 for r in residuals)
        y_mean = statistics.mean(ys)
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        if ss_tot == 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

    def _total_effect_via_paths(self, do_var: str, target_var: str,
                                max_depth: int = 6) -> float:
        """沿所有 do_var→target_var 的有向路径累加效应量。"""
        if do_var == target_var:
            return 1.0
        total = 0.0
        stack = [(do_var, 1.0, 0, {do_var})]
        while stack:
            node, eff, depth, visited = stack.pop()
            if depth >= max_depth:
                continue
            for child in self.graph.get_children(node):
                if child in visited:
                    continue
                edge = self.graph.edges.get(f"{node}->{child}")
                if not edge or edge.effect_size == 0:
                    continue
                new_eff = eff * edge.effect_size
                if child == target_var:
                    total += new_eff
                else:
                    stack.append((child, new_eff, depth + 1, visited | {child}))
        return total

    def _backdoor_adjusted_effect(self, do_var: str, target_var: str
                                  ) -> Tuple[float, List[str]]:
        """
        后门调整估计 do_var 对 target_var 的因果效应。

        P(Y | do(X)) = Σ_z P(Y | X, Z=z) P(Z=z)
        线性模型下: 回归 Y ~ X + Z，X 的系数即为无偏因果效应。

        Returns:
            (adjusted_effect, backdoor_vars)
        """
        do_parents = set(self.graph.get_parents(do_var))
        target_ancestors = set(self.graph.get_ancestors(target_var))

        backdoor_vars = list((do_parents & target_ancestors) - {target_var})

        for z in self._confounders:
            if z == do_var or z == target_var or z in backdoor_vars:
                continue
            z_children = set(self.graph.get_children(z))
            if do_var in z_children and (
                target_var in z_children or z in target_ancestors
            ):
                backdoor_vars.append(z)

        backdoor_vars = [v for v in dict.fromkeys(backdoor_vars)
                         if v not in (do_var, target_var)]

        if not backdoor_vars:
            return self._total_effect_via_paths(do_var, target_var), []

        feature_vars = [do_var] + backdoor_vars
        rows = []
        target_vals = []
        for o in self.observations:
            t = o.get(target_var)
            if t is None:
                continue
            row = [o.get(v) for v in feature_vars]
            if any(r is None for r in row):
                continue
            rows.append([float(r) for r in row])
            target_vals.append(float(t))

        if len(rows) < len(feature_vars) + 3:
            return self._total_effect_via_paths(do_var, target_var), backdoor_vars

        X = [[1.0] + row for row in rows]
        coeffs = _least_squares(X, target_vals)
        if coeffs is None:
            return self._total_effect_via_paths(do_var, target_var), backdoor_vars

        adjusted_effect = coeffs[1]
        if not math.isfinite(adjusted_effect):
            return self._total_effect_via_paths(do_var, target_var), backdoor_vars

        return adjusted_effect, backdoor_vars

    # ── Doubly Robust 估计 ───────────────────────────────

    @staticmethod
    def _logistic_sigmoid(z: float) -> float:
        """数值稳定的 sigmoid 函数"""
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def _propensity(self, treated_var: str, covariates: List[str]) -> List[float]:
        """
        倾向得分估计（逻辑回归 + 梯度下降）。

        将 treated_var 按中位数二值化，用 covariates 预测处理概率。

        Returns:
            倾向得分列表 [0.0001, 0.9999]
        """
        # 提取数据
        rows, treated = [], []
        for o in self.observations:
            tv = o.get(treated_var)
            if tv is None:
                continue
            row = [o.get(c) for c in covariates]
            if any(r is None for r in row):
                continue
            rows.append([float(r) for r in row])
            treated.append(float(tv))

        if len(rows) < 20:
            return []

        # 二值化处理（中位数分割）
        t_median = statistics.median(treated)
        t_binary = [1.0 if t > t_median else 0.0 for t in treated]

        # 标准化协变量
        n, p = len(rows), len(covariates)
        col_means = [statistics.mean([rows[i][j] for i in range(n)]) for j in range(p)]
        col_stds = []
        for j in range(p):
            vals = [rows[i][j] for i in range(n)]
            std = statistics.stdev(vals) if len(set(vals)) > 1 else 1.0
            col_stds.append(max(std, 1e-6))

        X_std = [[(rows[i][j] - col_means[j]) / col_stds[j] for j in range(p)]
                 for i in range(n)]

        # 逻辑回归梯度下降
        beta = [0.0] * p
        lr = 0.1
        max_iter = 500
        prev_loss = float('inf')

        for _ in range(max_iter):
            grads = [0.0] * p
            loss = 0.0
            for i in range(n):
                logit = sum(beta[j] * X_std[i][j] for j in range(p))
                logit = max(-30, min(30, logit))
                pred = self._logistic_sigmoid(logit)
                err = pred - t_binary[i]
                for j in range(p):
                    grads[j] += err * X_std[i][j]
                # BCE loss
                if t_binary[i] == 1:
                    loss -= math.log(max(pred, 1e-10))
                else:
                    loss -= math.log(max(1 - pred, 1e-10))
            loss /= n
            for j in range(p):
                beta[j] -= lr * grads[j] / n
            if abs(prev_loss - loss) < 1e-6:
                break
            prev_loss = loss

        # 计算倾向得分
        propensities = []
        for i in range(n):
            logit = sum(beta[j] * X_std[i][j] for j in range(p))
            logit = max(-30, min(30, logit))
            p_score = self._logistic_sigmoid(logit)
            p_score = max(1e-4, min(1 - 1e-4, p_score))
            propensities.append(p_score)

        return propensities

    def _doubly_robust_effect(self, do_var: str, target_var: str
                              ) -> Tuple[float, List[str], str]:
        """
        Doubly Robust 估计: 结合倾向得分和结果回归模型。

        E[Y(1)] = (1/N) Σ [T_i(Y_i - μ₁(X_i))/e_i + μ₁(X_i)]
        E[Y(0)] = (1/N) Σ [(1-T_i)(Y_i - μ₀(X_i))/(1-e_i) + μ₀(X_i)]
        ATE = E[Y(1)] - E[Y(0)]

        只要倾向得分模型或结果模型之一正确，ATE 无偏。

        Returns:
            (ate, covariates, estimator_name)
        """
        # 后门变量作为协变量
        _, backdoor_vars = self._backdoor_adjusted_effect(do_var, target_var)
        covariates = backdoor_vars if backdoor_vars else [
            n for n in self.graph.nodes if n not in (do_var, target_var)
        ][:5]

        # 提取数据
        rows, treated_vals, target_vals = [], [], []
        for o in self.observations:
            tv = o.get(target_var)
            dv = o.get(do_var)
            if tv is None or dv is None:
                continue
            row = [o.get(c) for c in covariates]
            if any(r is None for r in row):
                continue
            rows.append([float(r) for r in row])
            treated_vals.append(float(dv))
            target_vals.append(float(tv))

        n = len(rows)
        if n < 20:
            return self._backdoor_adjusted_effect(do_var, target_var)[0], covariates, "backdoor_fallback"

        # 二值化处理
        t_median = statistics.median(treated_vals)
        t_binary = [1.0 if t > t_median else 0.0 for t in treated_vals]
        n_treated = sum(t_binary)
        n_control = n - n_treated

        if n_treated < 8 or n_control < 8:
            return self._backdoor_adjusted_effect(do_var, target_var)[0], covariates, "backdoor_fallback"

        # 倾向得分
        propensities = self._propensity(do_var, covariates)
        if not propensities or len(propensities) != n:
            return self._backdoor_adjusted_effect(do_var, target_var)[0], covariates, "backdoor_fallback"

        # 检查倾向得分方差
        prop_var = statistics.variance(propensities)
        if prop_var < 1e-6:
            return self._backdoor_adjusted_effect(do_var, target_var)[0], covariates, "backdoor_fallback"

        # 结果回归模型: Y ~ T + covariates
        X_outcome = [[1.0, t_binary[i]] + rows[i] for i in range(n)]
        coeffs = _least_squares(X_outcome, target_vals)
        if coeffs is None:
            return self._backdoor_adjusted_effect(do_var, target_var)[0], covariates, "backdoor_fallback"

        # μ₁(X) = 预测 T=1 时的 Y
        # μ₀(X) = 预测 T=0 时的 Y
        mu1 = [sum(coeffs[j] * X_outcome[i][j] for j in range(len(coeffs)))
               for i in range(n)]
        # 修改 T=1 重算
        mu1_vals = []
        mu0_vals = []
        for i in range(n):
            x1 = [1.0, 1.0] + rows[i]
            x0 = [1.0, 0.0] + rows[i]
            mu1_vals.append(sum(coeffs[j] * x1[j] for j in range(len(coeffs))))
            mu0_vals.append(sum(coeffs[j] * x0[j] for j in range(len(coeffs))))

        # DR 估计
        y1_dr = 0.0
        y0_dr = 0.0
        for i in range(n):
            e_i = propensities[i]
            y1_dr += t_binary[i] * (target_vals[i] - mu1_vals[i]) / e_i + mu1_vals[i]
            y0_dr += (1 - t_binary[i]) * (target_vals[i] - mu0_vals[i]) / (1 - e_i) + mu0_vals[i]

        y1_dr /= n
        y0_dr /= n
        ate = y1_dr - y0_dr

        if not math.isfinite(ate):
            return self._backdoor_adjusted_effect(do_var, target_var)[0], covariates, "backdoor_fallback"

        # 将二值化 ATE 缩放回连续变量尺度
        do_std = statistics.stdev(treated_vals) if len(set(treated_vals)) > 1 else 1.0
        scaled_effect = ate / max(do_std * 0.5, 1e-6)

        return scaled_effect, covariates, "doubly_robust"

    def _ipw_effect(self, do_var: str, target_var: str) -> Tuple[float, List[str]]:
        """
        IPW (Inverse Propensity Weighting) 估计。

        E[Y(1)] = (1/N) Σ T_i Y_i / e_i
        E[Y(0)] = (1/N) Σ (1-T_i) Y_i / (1-e_i)
        """
        _, backdoor_vars = self._backdoor_adjusted_effect(do_var, target_var)
        covariates = backdoor_vars if backdoor_vars else [
            n for n in self.graph.nodes if n not in (do_var, target_var)
        ][:5]

        rows, treated_vals, target_vals = [], [], []
        for o in self.observations:
            tv = o.get(target_var)
            dv = o.get(do_var)
            if tv is None or dv is None:
                continue
            row = [o.get(c) for c in covariates]
            if any(r is None for r in row):
                continue
            rows.append([float(r) for r in row])
            treated_vals.append(float(dv))
            target_vals.append(float(tv))

        n = len(rows)
        if n < 20:
            return self._backdoor_adjusted_effect(do_var, target_var)

        t_median = statistics.median(treated_vals)
        t_binary = [1.0 if t > t_median else 0.0 for t in treated_vals]

        propensities = self._propensity(do_var, covariates)
        if not propensities or len(propensities) != n:
            return self._backdoor_adjusted_effect(do_var, target_var)

        y1_ipw = sum(t_binary[i] * target_vals[i] / propensities[i] for i in range(n)) / n
        y0_ipw = sum((1 - t_binary[i]) * target_vals[i] / (1 - propensities[i]) for i in range(n)) / n
        ate = y1_ipw - y0_ipw

        if not math.isfinite(ate):
            return self._backdoor_adjusted_effect(do_var, target_var)

        do_std = statistics.stdev(treated_vals) if len(set(treated_vals)) > 1 else 1.0
        scaled_effect = ate / max(do_std * 0.5, 1e-6)

        return scaled_effect, covariates

    # ── Level 1: 关联层 ───────────────────────────────────

    def query_cause(self, effect: str, top_k: int = 5) -> List[Dict]:
        """查询某结果的可能原因"""
        parents = self.graph.get_parents(effect)
        ancestors = self.graph.get_ancestors(effect)
        causes = []
        for p in parents:
            edge = self.graph.edges.get(f"{p}->{effect}")
            if edge:
                causes.append({
                    "cause": p,
                    "confidence": edge.confidence,
                    "effect_size": edge.effect_size,
                    "relation": "direct",
                })
        direct_set = set(parents)
        for a in ancestors:
            if a not in direct_set:
                causes.append({
                    "cause": a,
                    "confidence": 0.3,
                    "effect_size": 0.0,
                    "relation": "indirect",
                })
        causes.sort(key=lambda c: abs(c["effect_size"]), reverse=True)
        self._inferences += 1
        return causes[:top_k]

    def query_effect(self, cause: str, top_k: int = 5) -> List[Dict]:
        """查询某原因可能导致的结果"""
        children = self.graph.get_children(cause)
        effects = []
        for c in children:
            edge = self.graph.edges.get(f"{cause}->{c}")
            if edge:
                effects.append({
                    "effect": c,
                    "confidence": edge.confidence,
                    "effect_size": edge.effect_size,
                })
        effects.sort(key=lambda e: abs(e["effect_size"]), reverse=True)
        self._inferences += 1
        return effects[:top_k]

    # ── Level 2: 干预层（do-calculus + DR 估计）──────────

    def intervene(self, do_var: str, do_value: float,
                  target_var: str, n_samples: int = 100,
                  estimator: str = "auto") -> Dict:
        """
        干预推理: P(target | do(do_var=do_value))

        估计器选择 (estimator):
        - "auto": 优先 DR → 后门调整 → 路径传播
        - "dr": Doubly Robust
        - "backdoor": 后门调整
        - "ipw": IPW
        - "path": 路径传播

        Returns:
            {expected, std, samples, intervention_effect,
             estimator_used, backdoor_adjusted, backdoor_vars}
        """
        if not self.graph.edges:
            return self._intervene_correlation(do_var, do_value, target_var)

        target_obs = [
            o.get(target_var) for o in self.observations
            if o.get(target_var) is not None
        ]
        if not target_obs:
            return {"expected": None, "reason": "无目标变量数据"}

        target_mean = statistics.mean([float(t) for t in target_obs])
        target_std = statistics.stdev([float(t) for t in target_obs]) if len(target_obs) > 1 else 1.0

        # 估计器选择
        effect = 0.0
        backdoor_vars = []
        estimator_used = "path"

        if estimator in ("auto", "dr"):
            effect, covariates, estimator_used = self._doubly_robust_effect(do_var, target_var)
            if estimator_used == "backdoor_fallback":
                effect, backdoor_vars = self._backdoor_adjusted_effect(do_var, target_var)
                estimator_used = "backdoor" if backdoor_vars else "path"
        elif estimator == "ipw":
            effect, backdoor_vars = self._ipw_effect(do_var, target_var)
            estimator_used = "ipw"
        elif estimator == "backdoor":
            effect, backdoor_vars = self._backdoor_adjusted_effect(do_var, target_var)
            estimator_used = "backdoor" if backdoor_vars else "path"
        else:
            effect, backdoor_vars = self._backdoor_adjusted_effect(do_var, target_var)
            estimator_used = "backdoor" if backdoor_vars else "path"

        backdoor_adjusted = len(backdoor_vars) > 0 or estimator_used in ("doubly_robust", "ipw")

        # 干预效应
        do_obs = [o.get(do_var) for o in self.observations if o.get(do_var) is not None]
        do_mean = statistics.mean([float(d) for d in do_obs]) if do_obs else 0.0
        delta = (do_value - do_mean) * effect

        # 蒙特卡洛采样
        rng = random.Random(self._projection_seed)
        samples = []
        for _ in range(n_samples):
            noise = rng.gauss(0, target_std * 0.3)
            samples.append(target_mean + delta + noise)

        self._inferences += 1
        return {
            "expected": statistics.mean(samples),
            "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "samples_count": n_samples,
            "intervention_effect": effect,
            "estimator_used": estimator_used,
            "backdoor_adjusted": backdoor_adjusted,
            "backdoor_vars": backdoor_vars if backdoor_vars else covariates if 'covariates' in dir() else [],
            "do_value": do_value,
            "baseline_mean": target_mean,
            "delta": delta,
        }

    def _intervene_correlation(self, do_var, do_value, target_var) -> Dict:
        """无因果图时的相关性近似干预"""
        corr = self._estimate_effect(do_var, target_var)
        target_obs = [o.get(target_var) for o in self.observations if o.get(target_var) is not None]
        do_obs = [o.get(do_var) for o in self.observations if o.get(do_var) is not None]
        if not target_obs or not do_obs:
            return {"expected": None, "reason": "数据不足"}
        t_mean = statistics.mean([float(t) for t in target_obs])
        t_std = statistics.stdev([float(t) for t in target_obs]) if len(target_obs) > 1 else 1.0
        d_mean = statistics.mean([float(d) for d in do_obs])
        d_std = statistics.stdev([float(d) for d in do_obs]) if len(do_obs) > 1 else 1.0
        if d_std == 0:
            return {"expected": t_mean, "reason": "do_var 无变异"}
        delta = (do_value - d_mean) / d_std * corr * t_std
        return {
            "expected": t_mean + delta,
            "std": t_std * math.sqrt(1 - corr**2),
            "intervention_effect": corr,
            "estimator_used": "correlation",
            "note": "基于相关性近似（无因果图）",
        }

    # ── Level 3: 反事实层 ──────────────────────────────────

    def counterfactual(self, observed_x: float, observed_y: float,
                       counterfactual_x: float,
                       cause_var: str = "", effect_var: str = "",
                       noise_samples: int = 50) -> Dict:
        """
        反事实推理: 如果当初 X 是 cf_x 而非 obs_x，Y 会是多少？

        Pearl 三步:
        1. Abduction: U = Y - f(X)
        2. Action: 修改 X = cf_x
        3. Prediction: Y_cf = f(cf_x, U)

        改进: 噪声模型使用经验残差标准差（而非任意 0.05*|u|+0.01）。
        """
        if not cause_var or not effect_var:
            best_eff, best_cause, best_effect = 0.0, "", ""
            for e in self.graph.edges.values():
                if abs(e.effect_size) > abs(best_eff):
                    best_eff, best_cause, best_effect = e.effect_size, e.source, e.target
            if best_cause and best_effect:
                cause_var, effect_var = best_cause, best_effect
            else:
                cause_var, effect_var = "x", "y"

        paired = [
            (o.get(cause_var), o.get(effect_var))
            for o in self.observations
            if o.get(cause_var) is not None and o.get(effect_var) is not None
        ]
        if len(paired) < 5:
            delta = (counterfactual_x - observed_x) * 0.5
            return {
                "counterfactual_y": observed_y + delta,
                "original_y": observed_y,
                "delta": delta,
                "confidence": 0.2,
                "note": "数据不足，低置信度",
            }

        xs = [float(p[0]) for p in paired]
        ys = [float(p[1]) for p in paired]
        a, b, residuals = _linear_fit(xs, ys)

        # Step 1: Abduction
        predicted_y = a * observed_x + b
        u = observed_y - predicted_y

        # 经验残差标准差（改进的噪声模型）
        if len(residuals) >= 20:
            residual_std = statistics.stdev(residuals)
        else:
            residual_std = 0.05 * abs(u) + 0.01  # 回退到原模型

        # R² 置信度
        ss_res = sum(r ** 2 for r in residuals)
        y_mean = statistics.mean(ys)
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        r_squared = max(0.0, min(1.0, 1.0 - ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # Step 2+3: Action + Prediction
        rng = random.Random(self._projection_seed + 1)
        cf_samples = []
        for _ in range(noise_samples):
            u_sample = u + rng.gauss(0, residual_std)
            cf_y = a * counterfactual_x + b + u_sample
            cf_samples.append(cf_y)

        cf_y = statistics.mean(cf_samples)
        cf_std = statistics.stdev(cf_samples) if len(cf_samples) > 1 else 0.0
        delta = cf_y - observed_y

        confidence = min(1.0, len(paired) / 50.0) * 0.5 + r_squared * 0.5

        self._counterfactuals += 1
        return {
            "counterfactual_y": cf_y,
            "counterfactual_y_std": cf_std,
            "original_y": observed_y,
            "delta": delta,
            "cause_var": cause_var,
            "effect_var": effect_var,
            "model": {"slope": a, "intercept": b, "residual_u": u},
            "r_squared": r_squared,
            "confidence": confidence,
        }

    def counterfactual_multi(self, observed: Dict[str, float],
                             counterfactual: Dict[str, float],
                             target: str,
                             noise_samples: int = 50) -> Dict:
        """
        多变量反事实推理: 如果当初多个 X 是 cf 值，target 会是多少？

        改进: 噪声模型使用训练残差的经验标准差。
        """
        if target not in observed:
            return {"error": f"target '{target}' 不在 observed 中"}

        parents = self.graph.get_parents(target)
        if not parents:
            parents = [v for v in observed if v != target]

        cause_vars = [v for v in parents if v in observed or v in counterfactual]
        if not cause_vars:
            return {"error": "无可用原因变量"}

        n_feats = len(cause_vars)
        rows = []
        target_vals = []
        for o in self.observations:
            if o.get(target) is None:
                continue
            row = [o.get(v) for v in cause_vars]
            if any(r is None for r in row):
                continue
            rows.append([float(r) for r in row])
            target_vals.append(float(o.get(target)))

        if len(rows) < n_feats + 2:
            return {"error": "数据不足以拟合多元模型"}

        X = [[1.0] + row for row in rows]
        y = target_vals
        coeffs = _least_squares(X, y)
        if coeffs is None:
            return {"error": "多元回归求解失败"}

        b = coeffs[0]
        weights = coeffs[1:]

        # 训练残差
        train_residuals = [y[i] - sum(coeffs[j] * X[i][j] for j in range(len(coeffs)))
                           for i in range(len(y))]

        # Step 1: Abduction
        pred_obs = b + sum(weights[i] * float(observed.get(cause_vars[i], 0))
                          for i in range(n_feats))
        u = float(observed[target]) - pred_obs

        # 经验残差标准差
        if len(train_residuals) >= 20:
            residual_std = statistics.stdev(train_residuals)
        else:
            residual_std = 0.05 * abs(u) + 0.01

        # R² 置信度
        y_mean = statistics.mean(y)
        ss_res = sum(r ** 2 for r in train_residuals)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        r_squared = max(0.0, min(1.0, 1.0 - ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # Step 2+3: Action + Prediction
        pred_cf = b + sum(
            weights[i] * float(counterfactual.get(cause_vars[i], observed.get(cause_vars[i], 0)))
            for i in range(n_feats)
        )
        rng = random.Random(self._projection_seed + 2)
        cf_samples = []
        for _ in range(noise_samples):
            u_sample = u + rng.gauss(0, residual_std)
            cf_samples.append(pred_cf + u_sample)

        cf_y = statistics.mean(cf_samples)
        cf_std = statistics.stdev(cf_samples) if len(cf_samples) > 1 else 0.0
        delta = cf_y - float(observed[target])
        confidence = min(1.0, len(rows) / (n_feats * 10 + 10)) * 0.5 + r_squared * 0.5

        self._counterfactuals += 1
        return {
            "counterfactual_y": cf_y,
            "counterfactual_y_std": cf_std,
            "original_y": float(observed[target]),
            "delta": delta,
            "target": target,
            "cause_vars": cause_vars,
            "coefficients": {"intercept": b,
                             "weights": dict(zip(cause_vars, weights))},
            "residual_u": u,
            "r_squared": r_squared,
            "confidence": confidence,
        }

    # ── 因果中介分析 ────────────────────────────────────

    def mediate(self, treatment: str, mediator: str, outcome: str,
                treatment_value: float = 1.0, control_value: float = 0.0,
                n_samples: int = 100) -> Dict:
        """
        因果中介分析: treatment → mediator → outcome 路径分解。

        分解总效应:
        - NDE (Natural Direct Effect): 不经过 mediator 的直接效应
        - NIE (Natural Indirect Effect): 经过 mediator 的间接效应
        - Total Effect = NDE + NIE

        使用 Baron & Kenny 三步法:
        1. M ~ X → 系数 a
        2. Y ~ X + M → 系数 c' (X 直接效应), b (M 效应)
        3. NDE = c' * ΔX, NIE = a * b * ΔX

        Args:
            treatment: 处理变量
            mediator: 中介变量
            outcome: 结果变量
            treatment_value: 处理值
            control_value: 对照值
            n_samples: Bootstrap 样本数

        Returns:
            {nde, nie, total_effect, a, b, c_prime,
             mediation_ratio, consistency_check}
        """
        # 提取数据
        rows = []
        for o in self.observations:
            x, m, y = o.get(treatment), o.get(mediator), o.get(outcome)
            if x is not None and m is not None and y is not None:
                rows.append((float(x), float(m), float(y)))

        if len(rows) < 10:
            return {"error": "数据不足（< 10 个完整观测）"}

        xs = [r[0] for r in rows]
        ms = [r[1] for r in rows]
        ys = [r[2] for r in rows]

        # Step 1: M ~ X
        a, _, _ = _linear_fit(xs, ms)

        # Step 2: Y ~ X + M
        X_reg = [[1.0, xs[i], ms[i]] for i in range(len(rows))]
        coeffs = _least_squares(X_reg, ys)
        if coeffs is None:
            return {"error": "多元回归失败"}

        c_prime = coeffs[1]  # X 的直接效应
        b = coeffs[2]        # M 的效应

        delta_x = treatment_value - control_value
        nde = c_prime * delta_x
        nie = a * b * delta_x
        total = nde + nie

        # 一致性检查: 与后门调整估计对比
        backdoor_effect, _ = self._backdoor_adjusted_effect(treatment, outcome)
        backdoor_total = backdoor_effect * delta_x
        if abs(backdoor_total) > 1e-6:
            rel_diff = abs(total - backdoor_total) / abs(backdoor_total)
            consistent = rel_diff < 0.3
        else:
            consistent = abs(total) < 0.1

        # Bootstrap 置信区间
        rng = random.Random(self._projection_seed + 3)
        nde_samples, nie_samples = [], []
        n = len(rows)
        for _ in range(min(n_samples, 100)):
            sampled = [rng.choice(rows) for _ in range(n)]
            sxs = [r[0] for r in sampled]
            sms = [r[1] for r in sampled]
            sys_ = [r[2] for r in sampled]
            sa, _, _ = _linear_fit(sxs, sms)
            sX = [[1.0, sxs[i], sms[i]] for i in range(n)]
            sc = _least_squares(sX, sys_)
            if sc is not None:
                nde_samples.append(sc[1] * delta_x)
                nie_samples.append(sa * sc[2] * delta_x)

        nde_std = statistics.stdev(nde_samples) if len(nde_samples) > 1 else 0.0
        nie_std = statistics.stdev(nie_samples) if len(nie_samples) > 1 else 0.0

        mediation_ratio = nie / total if abs(total) > 1e-6 else 0.0

        return {
            "nde": nde,
            "nie": nie,
            "total_effect": total,
            "a": a,
            "b": b,
            "c_prime": c_prime,
            "mediation_ratio": mediation_ratio,
            "nde_std": nde_std,
            "nie_std": nie_std,
            "consistency_check": consistent,
            "backdoor_estimate": backdoor_total,
            "treatment": treatment,
            "mediator": mediator,
            "outcome": outcome,
        }

    # ── 量子核相似度（SHA-256 确定性投影）──────────────

    def _quantum_project(self, var: str) -> List[float]:
        """把变量名投影到 quantum_dim 维量子态空间。

        使用 SHA-256 派生种子，保证跨进程可复现（Python hash() 被随机化）。
        """
        cache_key = f"{self.quantum_dim}:{var}"
        if cache_key in self._projection_cache:
            return self._projection_cache[cache_key]
        # SHA-256 派生确定性种子
        seed_str = f"{self._projection_seed}:{var}"
        hash_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
        seed = int.from_bytes(hash_bytes[:8], "big") & 0x7FFFFFFF
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(self.quantum_dim)]
        norm = math.sqrt(sum(v*v for v in vec))
        vec = [v / norm for v in vec]
        self._projection_cache[cache_key] = vec
        return vec

    def quantum_similarity(self, var_a: str, var_b: str) -> float:
        """量子核相似度：两个变量在量子态空间的余弦相似度"""
        va = self._quantum_project(var_a)
        vb = self._quantum_project(var_b)
        return sum(a*b for a, b in zip(va, vb))

    # ── 统一入口（输入验证 + 错误处理）──────────────────

    @staticmethod
    def _safe_float(val, default=0.0):
        """安全转 float"""
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def predict(self, query: Dict, mode: str = "auto", top_k: int = 3) -> Dict:
        """
        统一推理入口

        Args:
            query: {
                "type": "state_analysis" | "cause_query" | "effect_query" |
                        "intervention" | "counterfactual",
                ...
            }
            mode: "auto" | "causal" | "statistical"
            top_k: 返回结果数

        Returns:
            推理结果字典
        """
        # 输入验证
        if not isinstance(query, dict):
            return {"type": "error", "error": "query 必须是字典"}

        qtype = query.get("type", "state_analysis")
        valid_types = {"state_analysis", "cause_query", "effect_query",
                       "intervention", "counterfactual"}
        if qtype not in valid_types:
            return {"type": "error", "error": f"未知类型: {qtype}",
                    "valid_types": list(valid_types)}

        # 自动发现
        if mode == "auto" and not self.graph.edges and len(self.observations) >= 10:
            try:
                self.discover()
            except Exception as e:
                logger.warning(f"自动发现失败: {e}")

        # 分发处理（带错误处理）
        try:
            if qtype == "cause_query":
                effect = query.get("effect", "")
                if not effect:
                    return {"type": "error", "error": "缺少 effect 参数"}
                causes = self.query_cause(effect, top_k)
                return {"type": "causes", "causes": causes, "effect": effect}

            elif qtype == "effect_query":
                cause = query.get("cause", "")
                if not cause:
                    return {"type": "error", "error": "缺少 cause 参数"}
                effects = self.query_effect(cause, top_k)
                return {"type": "effects", "effects": effects, "cause": cause}

            elif qtype == "intervention":
                do_var = query.get("do_var", "")
                target_var = query.get("target_var", "")
                if not do_var:
                    return {"type": "error", "error": "缺少 do_var 参数"}
                if not target_var:
                    return {"type": "error", "error": "缺少 target_var 参数"}
                do_value = self._safe_float(query.get("do_value", 0))
                result = self.intervene(
                    do_var, do_value, target_var,
                    query.get("n_samples", 100),
                    estimator=query.get("estimator", "auto"),
                )
                return {"type": "intervention", "result": result}

            elif qtype == "counterfactual":
                result = self.counterfactual(
                    self._safe_float(query.get("observed_x", 0)),
                    self._safe_float(query.get("observed_y", 0)),
                    self._safe_float(query.get("counterfactual_x", 0)),
                    query.get("cause_var", ""),
                    query.get("effect_var", ""),
                    query.get("noise_samples", 50),
                )
                return {"type": "counterfactual", "result": result}

            # 默认: state_analysis
            return self._state_analysis(query, top_k)

        except Exception as e:
            logger.warning(f"predict() 处理 {qtype} 时出错: {e}", exc_info=True)
            return {"type": "error", "error": str(e),
                    "query_type": qtype, "fallback": True}

    def _state_analysis(self, query: Dict, top_k: int) -> Dict:
        """状态分析"""
        needs = query.get("needs", {})
        emotion = query.get("emotion", "")
        snapshot = {"self_presence": query.get("self_presence", 0.5)}

        if needs:
            dominant_need = max(needs, key=lambda k: needs.get(k, 0))
            dominant_val = needs.get(dominant_need, 0)
        else:
            dominant_need = "unknown"
            dominant_val = 0.0

        emotion_causes = self.query_cause(emotion, top_k) if emotion else []

        related = []
        for node in list(self.graph.nodes)[:top_k * 2]:
            if node != emotion and node != dominant_need:
                sim = self.quantum_similarity(emotion or dominant_need, node)
                if abs(sim) > 0.1:
                    related.append({"variable": node, "quantum_similarity": round(sim, 3)})

        related.sort(key=lambda r: abs(r["quantum_similarity"]), reverse=True)

        self._inferences += 1
        return {
            "type": "state_analysis",
            "dominant_need": dominant_need,
            "dominant_value": dominant_val,
            "emotion": emotion,
            "emotion_causes": emotion_causes[:top_k],
            "quantum_related": related[:top_k],
            "self_presence": snapshot["self_presence"],
            "n_observations": len(self.observations),
            "n_causal_edges": len(self.graph.edges),
            "inferences_count": self._inferences,
        }

    # ── 接口兼容方法 (aris_cognitive_bridge.py) ──────────

    def learn_bond(self, cause: str, target: str = None, *,
                   effect: str = None, matched: bool = True,
                   domain: str = "general", **kwargs) -> Dict:
        """
        学习/记录一条因果绑定。

        兼容 aris_cognitive_bridge.py 的调用:
        ce.learn_bond("aris_said", topic, effect="主人_responded", matched=True, domain="social")

        若 matched=True，直接在因果图中添加一条边。
        """
        # 统一 target / effect 参数
        child = target if target is not None else effect
        if not cause or not child:
            return {"error": "需要 cause 和 target/effect"}

        confidence = 0.8 if matched else 0.4
        effect_size = self._estimate_effect(cause, child) if len(self.observations) >= 5 else 0.0

        self.graph.add_edge(
            cause, child,
            confidence=confidence,
            effect_size=effect_size,
            edge_type="learned",
            n_observations=len(self.observations),
        )

        if self.persist_path:
            self._save()

        return {
            "bond": f"{cause}->{child}",
            "confidence": confidence,
            "matched": matched,
            "domain": domain,
        }

    def detect_transitive_chains(self, min_confidence: float = 0.5) -> List[Dict]:
        """
        检测因果图中的传递链。

        查找 A→B→C 链，其中 A→C 可能缺失或低置信度。
        用于发现间接因果路径。

        Returns:
            传递链列表 [{chain, confidence, gap}]
        """
        chains = []
        for a in self.graph.nodes:
            children_a = self.graph.get_children(a)
            for b in children_a:
                children_b = self.graph.get_children(b)
                for c in children_b:
                    if c == a:
                        continue
                    # 检查 A→C 是否直接存在
                    direct_edge = self.graph.edges.get(f"{a}->{c}")
                    has_direct = direct_edge is not None
                    direct_conf = direct_edge.confidence if direct_edge else 0.0

                    if not has_direct or direct_conf < min_confidence:
                        edge_ab = self.graph.edges.get(f"{a}->{b}")
                        edge_bc = self.graph.edges.get(f"{b}->{c}")
                        if edge_ab and edge_bc:
                            chain_conf = edge_ab.confidence * edge_bc.confidence
                            if chain_conf >= min_confidence:
                                chains.append({
                                    "chain": [a, b, c],
                                    "confidence": round(chain_conf, 3),
                                    "gap": "no_direct" if not has_direct else "low_confidence",
                                    "direct_confidence": round(direct_conf, 3),
                                })

        chains.sort(key=lambda c: c["confidence"], reverse=True)
        return chains

    # ── BIC 评分 ────────────────────────────────────────

    def _bic_score(self, parent_set: Set[str], target: str) -> float:
        """贝叶斯信息准则 (BIC) 评分。越高越好。"""
        target_vals = []
        feature_rows = []
        for o in self.observations:
            t = o.get(target)
            if t is None:
                continue
            row = [o.get(p) for p in parent_set]
            if any(r is None for r in row):
                continue
            target_vals.append(float(t))
            feature_rows.append([float(r) for r in row])
        n = len(target_vals)
        if n < len(parent_set) + 2:
            return -1e10
        if parent_set:
            X = [[1.0] + row for row in feature_rows]
            coeffs = _least_squares(X, target_vals)
            if coeffs is None:
                return -1e10
            preds = [sum(coeffs[j] * X[i][j] for j in range(len(coeffs)))
                     for i in range(n)]
        else:
            mean_t = statistics.mean(target_vals)
            preds = [mean_t] * n
        rss = sum((target_vals[i] - preds[i]) ** 2 for i in range(n))
        sigma2 = rss / n if rss > 0 else 1e-10
        log_lik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1)
        k = len(parent_set) + 1
        bic = log_lik - 0.5 * k * math.log(n)
        return bic

    # ── 持久化（原子写入）────────────────────────────────

    def _atomic_write(self, path: Path, text: str):
        """原子写入: 先写临时文件，再 os.replace 原子替换"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp_path.write_text(text, encoding="utf-8")
            os.replace(str(tmp_path), str(path))
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def save(self):
        """公开持久化接口（aris_cognitive_bridge 等外部调用入口）。

        需先通过 set_persist_path() 指定路径，否则静默跳过。
        """
        self._save()

    def _save(self):
        """保存因果图和观测到文件（原子写入）"""
        if not self.persist_path:
            return
        data = {
            "name": self.name,
            "quantum_dim": self.quantum_dim,
            "graph": self.graph.to_dict(),
            "confounders": list(self._confounders),
            "observations_count": len(self.observations),
            "stats": {
                "inferences": self._inferences,
                "counterfactuals": self._counterfactuals,
                "discoveries": self._discoveries,
            },
            "schema_version": "ultimate_v1",
        }
        try:
            self._atomic_write(
                self.persist_path,
                json.dumps(data, ensure_ascii=False, indent=2),
            )
        except Exception as e:
            logger.warning(f"保存失败: {e}")

    def _load(self):
        """从文件恢复"""
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
            self.name = data.get("name", self.name)
            self.graph = CausalGraph.from_dict(data.get("graph", {}))
            self._confounders = set(data.get("confounders", []))
            self._inferences = data.get("stats", {}).get("inferences", 0)
            self._counterfactuals = data.get("stats", {}).get("counterfactuals", 0)
            self._discoveries = data.get("stats", {}).get("discoveries", 0)
            logger.info(f"[{self.name}] 恢复因果图: {len(self.graph.edges)} 条边")
        except Exception as e:
            logger.warning(f"加载失败: {e}")

    # ── 自检 ────────────────────────────────────────────

    def self_test(self) -> Dict:
        """
        端到端自检: 生成合成数据验证全流程。

        测试:
        1. 因果发现: 验证 discover() 能恢复已知因果结构
        2. 干预推理: 验证 intervene() 效应方向正确
        3. 反事实: 验证 counterfactual() 结果合理
        4. 中介分析: 验证 NDE/NIE 分解
        5. predict(): 验证统一入口不报错
        6. 量子投影: 验证跨实例可复现
        7. 原子持久化: 验证 save/load 往返

        Returns:
            {overall_pass, checks, failures, summary}
        """
        # 保存原始状态
        orig_observations = self.observations
        orig_graph = self.graph
        orig_confounders = self._confounders
        orig_inferences = self._inferences
        orig_cf = self._counterfactuals
        orig_disc = self._discoveries
        orig_cache = self._projection_cache

        checks = []
        failures = []

        def record(name, passed, detail=""):
            checks.append({"name": name, "passed": passed, "detail": detail})
            if not passed:
                failures.append(name)

        try:
            # 生成合成数据: X → M → Y, X → Y (直接)
            # Y = 0.7*X + 1.5*M + noise
            # M = 1.2*X + noise
            # 总效应 = 0.7 + 1.2*1.5 = 0.7 + 1.8 = 2.5
            rng = random.Random(2024)
            test_obs = []
            for _ in range(150):
                x = rng.gauss(0, 1)
                m = 1.2 * x + rng.gauss(0, 0.5)
                y = 0.7 * x + 1.5 * m + rng.gauss(0, 0.5)
                test_obs.append({"X": x, "M": m, "Y": y})

            # 用独立引擎实例测试
            test_engine = UnifiedCausalEngine(quantum_dim=32, name="TestEngine")
            test_engine.observe_many(test_obs)

            # Test 1: 因果发现
            try:
                result = test_engine.discover(alpha=0.05)
                edges = set()
                for e in test_engine.graph.edges:
                    s, t = e.split("->")
                    edges.add((s, t))

                # 期望至少恢复 X→M 和 M→Y
                has_xm = ("X", "M") in edges or ("M", "X") in edges
                has_my = ("M", "Y") in edges or ("Y", "M") in edges
                passed = has_xm and has_my
                record("discover_structure", passed,
                       f"edges={edges}, has_XM={has_xm}, has_MY={has_my}")
            except Exception as e:
                record("discover_structure", False, f"exception: {e}")

            # Test 2: 干预推理
            try:
                iv = test_engine.intervene("X", 2.0, "Y", n_samples=50)
                effect = iv.get("intervention_effect", 0)
                expected = iv.get("expected", 0)
                # X=2 时 Y 应该显著大于均值
                passed = effect > 0.3 and abs(expected) > 0.5
                record("intervention_direction", passed,
                       f"effect={effect:.3f}, expected={expected:.3f}")
            except Exception as e:
                record("intervention_direction", False, f"exception: {e}")

            # Test 3: 反事实
            try:
                cf = test_engine.counterfactual(
                    observed_x=0.0, observed_y=0.0,
                    counterfactual_x=2.0,
                    cause_var="X", effect_var="Y",
                )
                delta = cf.get("delta", 0)
                # X 从 0→2，Y 应该增加约 2.5*2=5... 但线性模型 a≈2.5
                # delta 应该为正且显著
                passed = delta > 0.5
                record("counterfactual_reasonable", passed,
                       f"delta={delta:.3f}")
            except Exception as e:
                record("counterfactual_reasonable", False, f"exception: {e}")

            # Test 4: 中介分析
            try:
                med = test_engine.mediate("X", "M", "Y", 1.0, 0.0)
                nie = med.get("nie", 0)
                nde = med.get("nde", 0)
                total = med.get("total_effect", 0)
                # NIE 应该为正（间接效应 a*b ≈ 1.2*1.5=1.8）
                # NDE 应该为正（直接效应 ≈ 0.7）
                passed = nie > 0.3 and nde > 0.1
                record("mediation_analysis", passed,
                       f"NDE={nde:.3f}, NIE={nie:.3f}, total={total:.3f}")
            except Exception as e:
                record("mediation_analysis", False, f"exception: {e}")

            # Test 5: predict() 统一入口
            try:
                result = test_engine.predict({
                    "type": "intervention",
                    "do_var": "X", "do_value": 1.0,
                    "target_var": "Y",
                })
                passed = isinstance(result, dict) and "result" in result
                record("predict_entry", passed, f"type={result.get('type')}")
            except Exception as e:
                record("predict_entry", False, f"exception: {e}")

            # Test 6: 量子投影可复现
            try:
                e2 = UnifiedCausalEngine(quantum_dim=32, name="TestEngine2")
                v1 = test_engine._quantum_project("test_var")
                v2 = e2._quantum_project("test_var")
                # 应该完全相同（SHA-256 确定性）
                diff = sum((a - b) ** 2 for a, b in zip(v1, v2))
                passed = diff < 1e-10
                record("quantum_reproducible", passed, f"diff={diff:.2e}")
            except Exception as e:
                record("quantum_reproducible", False, f"exception: {e}")

            # Test 7: 原子持久化
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                    tmp_path = tf.name
                os.unlink(tmp_path)  # 删除，让引擎创建

                persist_engine = UnifiedCausalEngine(
                    name="PersistTest", persist_path=tmp_path
                )
                persist_engine.observe_many(test_obs[:20])
                persist_engine.discover()
                persist_engine._save()

                # 重新加载
                reload_engine = UnifiedCausalEngine(
                    name="ReloadTest", persist_path=tmp_path
                )
                passed = len(reload_engine.graph.edges) > 0
                record("atomic_persistence", passed,
                       f"edges_after_reload={len(reload_engine.graph.edges)}")

                os.unlink(tmp_path)
            except Exception as e:
                record("atomic_persistence", False, f"exception: {e}")

            # Test 8: predict() 错误处理
            try:
                result = test_engine.predict({"type": "unknown_type"})
                passed = isinstance(result, dict) and result.get("type") == "error"
                record("error_handling", passed, f"result={result}")
            except Exception as e:
                record("error_handling", False, f"exception: {e}")

            # Test 9: detect_transitive_chains
            try:
                chains = test_engine.detect_transitive_chains()
                passed = isinstance(chains, list)
                record("transitive_chains", passed, f"chains={len(chains)}")
            except Exception as e:
                record("transitive_chains", False, f"exception: {e}")

            # Test 10: learn_bond
            try:
                bond_result = test_engine.learn_bond("A", "B", matched=True)
                passed = isinstance(bond_result, dict) and "bond" in bond_result
                record("learn_bond", passed, f"bond={bond_result.get('bond')}")
            except Exception as e:
                record("learn_bond", False, f"exception: {e}")

        finally:
            # 恢复原始状态
            self.observations = orig_observations
            self.graph = orig_graph
            self._confounders = orig_confounders
            self._inferences = orig_inferences
            self._counterfactuals = orig_cf
            self._discoveries = orig_disc
            self._projection_cache = orig_cache

        n_passed = len([c for c in checks if c["passed"]])
        n_total = len(checks)
        overall = n_passed == n_total

        return {
            "overall_pass": overall,
            "passed": n_passed,
            "total": n_total,
            "checks": checks,
            "failures": failures,
            "summary": f"{n_passed}/{n_total} 项通过",
        }

    # ── ANM 辅助 ────────────────────────────────────────

    def _normalize_anm_score(self, score: float) -> float:
        """ANM 评分归一化到 [-1, 1]"""
        return math.tanh(score)

    def _anm_direction_normalized(self, x: str, y: str) -> float:
        """归一化的 ANM 方向评分"""
        raw = self._anm_direction_score(x, y)
        return self._normalize_anm_score(raw)

    def status(self) -> Dict:
        """引擎状态"""
        return {
            "name": self.name,
            "version": "ultimate",
            "quantum_dim": self.quantum_dim,
            "observations": len(self.observations),
            "causal_edges": len(self.graph.edges),
            "confounders": len(self._confounders),
            "inferences": self._inferences,
            "counterfactuals": self._counterfactuals,
            "discoveries": self._discoveries,
            "persisted": self.persist_path.exists() if self.persist_path else False,
            "features": [
                "meek_rules", "hsic_anm", "doubly_robust",
                "ges", "directlingam", "mediation",
                "self_test", "atomic_save", "sha256_projection",
            ],
        }


# ── 全局单例工厂 ──────────────────────────────────────

_global_engine: Optional["UnifiedCausalEngine"] = None


def get_causal_engine(quantum_dim: int = 64,
                      name: str = "小茜Causal",
                      persist_path: Optional[str] = None
                      ) -> "UnifiedCausalEngine":
    """获取全局唯一的 UnifiedCausalEngine 实例。

    首次调用时创建并持久化加载;后续调用返回同一实例。
    所有模块(agi_subscriber / cognitive_bridge / goal_engine)
    应通过此函数获取引擎,避免多实例导致 observations 分裂
    和 save 覆盖问题。

    Args:
        quantum_dim: 量子投影维度(仅首次创建生效)
        name: 引擎名称(仅首次创建生效)
        persist_path: 持久化路径(仅首次创建生效,默认 None)
                      若已有实例则忽略此参数

    Returns:
        全局唯一的 UnifiedCausalEngine 实例
    """
    global _global_engine
    if _global_engine is None:
        _global_engine = UnifiedCausalEngine(
            quantum_dim=quantum_dim, name=name,
            persist_path=persist_path,
        )
    return _global_engine
