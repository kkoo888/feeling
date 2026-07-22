"""
UnifiedCausalEngine — 统一因果推理引擎
==========================================

基于 Pearl 因果层级理论（关联/干预/反事实）+ 项目设计文档实现。

设计依据:
- docs/feeling-advanced-design.md (CausalDiscovery 设计)
- docs/feeling-frontier-theories.md (Pearl 三层级 + DoWhy/CausalNex/CERMIC)
- docs/feeling-foundation-blocks-v2.md (方块 75 反事实 + 方块 76 因果图)
- aris_brain/agi_subscriber.py (调用契约: predict(query, mode, top_k))

核心能力:
1. observe(event)        - 记录观测
2. discover()            - PC 算法自动发现因果图
3. query_cause(effect)   - 查询结果的可能原因
4. intervene(do_x)       - 干预推理（do-calculus）
5. counterfactual(...)   - 反事实推理（Abduction-Action-Prediction）
6. predict(query, mode)  - 统一入口（供 agi_subscriber 调用）

不依赖外部因果库（DoWhy/CausalNex），纯 Python + 标准库 + networkx 实现。
量子核需求（quantum_dim）用随机投影模拟。
"""

import json
import logging
import math
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
    variables: Dict[str, Any]          # 变量名 -> 值（数值 or 分类）
    context: Dict[str, Any] = field(default_factory=dict)  # 上下文（时间/场景等）
    weight: float = 1.0                # 观测权重（近期观测权重高）

    def get(self, key, default=None):
        return self.variables.get(key, self.context.get(key, default))


@dataclass
class CausalEdge:
    """因果边"""
    source: str                         # 因（parent）
    target: str                         # 果（child）
    confidence: float = 0.0             # 置信度 [0, 1]
    effect_size: float = 0.0            # 效应量（正=促进，负=抑制）
    edge_type: str = "unknown"          # direct / indirect / confounded
    n_observations: int = 0             # 支持该边的观测数


class CausalGraph:
    """
    因果图 — 有向无环图（DAG）

    节点 = 变量，边 = 因果关系（X → Y 表示 X 导致 Y）
    """

    def __init__(self):
        self.edges: Dict[str, CausalEdge] = {}  # key = "source->target"
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
                 n_observations: int = 1):
        """添加因果边"""
        self.add_node(parent)
        self.add_node(child)
        key = f"{parent}->{child}"
        self.edges[key] = CausalEdge(
            source=parent, target=child, confidence=confidence,
            effect_size=effect_size, edge_type=edge_type,
            n_observations=n_observations,
        )
        if self._nx is not None:
            self._nx.add_edge(parent, child, confidence=confidence)
        # 检查环
        if not self.is_dag():
            # 添加后成环，撤销
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
        """检查是否为 DAG（无环）"""
        if self._nx is not None:
            try:
                return nx.is_directed_acyclic_graph(self._nx)
            except Exception:
                pass
        # fallback: 简单环检测
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
        """获取某变量的直接原因（父节点）"""
        return [e.source for e in self.edges.values() if e.target == node]

    def get_children(self, node: str) -> List[str]:
        """获取某变量的直接结果（子节点）"""
        return [e.target for e in self.edges.values() if e.source == node]

    def get_ancestors(self, node: str) -> List[str]:
        """获取所有祖先（间接原因）"""
        if self._nx is not None:
            try:
                return list(nx.ancestors(self._nx, node))
            except Exception:
                pass
        # BFS fallback
        ancestors, queue = set(), deque(self.get_parents(node))
        while queue:
            n = queue.popleft()
            if n not in ancestors:
                ancestors.add(n)
                queue.extend(self.get_parents(n))
        return list(ancestors)

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
# 统计工具（不依赖 scipy）
# ═══════════════════════════════════════════════════════════════

def _chi2_test(observed: List[List[float]], expected: List[List[float]]) -> float:
    """卡方检验，返回 p-value（近似）"""
    chi2 = 0.0
    for i in range(len(observed)):
        for j in range(len(observed[i])):
            exp = expected[i][j]
            if exp > 0:
                chi2 += ((observed[i][j] - exp) ** 2) / exp
    # 自由度 = (rows-1)*(cols-1)
    df = max(1, (len(observed) - 1) * (len(observed[0]) - 1))
    # 用 Wilson-Hilferty 近似从卡方值算 p-value
    z = ((chi2 / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
    # 标准正态 CDF
    p = 0.5 * (1 + math.erf(-z / math.sqrt(2)))
    return max(0.0, min(1.0, p))


def _safe_corr(xs: List[float], ys: List[float]) -> float:
    """安全的皮尔逊相关系数，处理边界情况"""
    if len(xs) < 2 or len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return 0.0
    try:
        return statistics.correlation(xs, ys)
    except (ValueError, statistics.StatisticsError):
        return 0.0


def _skewness(xs: List[float]) -> float:
    """样本偏度（Fisher-Pearson），衡量非高斯性。LiNGAM 用非高斯性判方向。"""
    n = len(xs)
    if n < 3:
        return 0.0
    m = statistics.fmean(xs)
    s = statistics.stdev(xs)
    if s == 0:
        return 0.0
    return statistics.fmean([((x - m) / s) ** 3 for x in xs])


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
    """
    最小二乘法求解 beta 使得 X*beta ≈ y（纯标准库实现）。

    用正规方程: beta = (X^T X)^{-1} X^T y
    通过高斯消元 + 主元选取解线性方程组。
    """
    n = len(X)
    if n == 0:
        return None
    p = len(X[0])
    # X^T X (p x p) 和 X^T y (p)
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        for j in range(p):
            Xty[j] += X[i][j] * y[i]
            for k in range(p):
                XtX[j][k] += X[i][j] * X[i][k]
    # 增广矩阵 [XtX | Xty]，高斯消元
    aug = [XtX[i] + [Xty[i]] for i in range(p)]
    for col in range(p):
        # 选主元（部分主元法）
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None  # 奇异矩阵
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(col + 1, p):
            factor = aug[r][col] / aug[col][col]
            for c in range(col, p + 1):
                aug[r][c] -= factor * aug[col][c]
    # 回代
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
    """
    条件独立性测试：给定 Z，X 和 Y 是否独立？

    返回 (is_independent, p_value)
    - is_independent=True 表示条件独立（无直接因果）
    - is_independent=False 表示可能存在因果
    """
    # 提取 X, Y, Z 的值
    xs = [o.get(x) for o in observations if o.get(x) is not None]
    ys = [o.get(y) for o in observations if o.get(y) is not None]
    if len(xs) < 5 or len(ys) < 5:
        return True, 1.0  # 数据不足，假设独立

    # 保留完整的配对数据（无条件测试和条件测试都会用）
    paired_full = [(float(o.get(x)), float(o.get(y))) for o in observations
                   if o.get(x) is not None and o.get(y) is not None]
    if len(paired_full) < 5:
        return True, 1.0

    # 简化：用相关系数做独立性测试（连续变量）
    # 如果 Z 非空，按 Z 分层后测 X-Y 相关性
    if not z_set:
        # 无条件：直接算 X-Y 相关
        if len(set([p[0] for p in paired_full])) <= 1 or len(set([p[1] for p in paired_full])) <= 1:
            return True, 1.0
        try:
            corr = statistics.correlation([p[0] for p in paired_full], [p[1] for p in paired_full])
            # Fisher z 变换算 p-value
            n = len(paired_full)
            z = 0.5 * math.log((1 + corr) / (1 - corr + 1e-10)) * math.sqrt(max(1, n - 3))
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
            return p > alpha, p
        except (ValueError, statistics.StatisticsError):
            return True, 1.0
    else:
        # 有条件：按 Z 分层，每层内测 X-Y 独立性
        # FIX: 原代码 group_lo/group_hi 命名混乱，导致两个分支用同一条件
        # 正确做法：按 Z 中位数分 low (<= median) 和 high (> median) 两组
        stratum_corrs = []
        for z_var in z_set:
            z_vals = [o.get(z_var) for o in observations if o.get(z_var) is not None]
            if len(z_vals) < 5:
                continue
            try:
                z_median = statistics.median([float(v) for v in z_vals])
            except (ValueError, TypeError):
                continue
            # FIX: 正确分组 — is_low=True 取低组, is_low=False 取高组
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
        # FIX: 条件独立 = 分层后相关性显著降低
        full_corr = abs(_safe_corr([p[0] for p in paired_full], [p[1] for p in paired_full]))
        is_independent = avg_corr < 0.15 or avg_corr < full_corr * 0.4
        return is_independent, 1.0 - avg_corr


# ═══════════════════════════════════════════════════════════════
# 主引擎
# ═══════════════════════════════════════════════════════════════

class UnifiedCausalEngine:
    """
    统一因果推理引擎

    实现 Pearl 三层级因果推理:
    - Level 1 关联: P(Y|X) — 看到 X 时 Y 的概率
    - Level 2 干预: P(Y|do(X)) — 做了 X 时 Y 的概率
    - Level 3 反事实: P(Y_x|X', Y') — 如果当初做 x 而非 X'

    用法:
        engine = UnifiedCausalEngine(quantum_dim=64, name="小茜Causal")
        engine.observe({"sleep": 5, "mood": 0.3})
        engine.discover()
        causes = engine.query_cause("mood")
        cf = engine.counterfactual(5, 0.3, 8)  # 如果睡 8 小时，心情会怎样
    """

    def __init__(self, quantum_dim: int = 64, name: str = "CausalEngine",
                 persist_path: Optional[str] = None):
        self.quantum_dim = quantum_dim
        self.name = name
        self.persist_path = Path(persist_path) if persist_path else None

        # 观测存储
        self.observations: List[Observation] = []
        self._max_observations = 2000          # 滑动窗口

        # 因果图
        self.graph = CausalGraph()

        # 已知混杂变量
        self._confounders: Set[str] = set()

        # 量子核投影矩阵（随机投影模拟，固定种子保证可复现）
        rng = random.Random(42)
        self._projection_seed = 42
        self._projection_cache: Dict[str, List[float]] = {}

        # 统计
        self._inferences = 0
        self._counterfactuals = 0
        self._discoveries = 0

        # 恢复持久化数据
        if self.persist_path and self.persist_path.exists():
            self._load()

        logger.info(f"[{self.name}] 因果引擎初始化 (quantum_dim={quantum_dim})")

    # ── 观测 ──────────────────────────────────────────────

    def observe(self, event: Union[Dict, Observation]):
        """记录一次观测"""
        if isinstance(event, dict):
            # 兼容两种格式：
            # 1. {"variables": {...}, "context": {...}} 显式格式
            # 2. {"var1": val1, "var2": val2} 扁平格式（自动当 variables）
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
        # 滑动窗口
        if len(self.observations) > self._max_observations:
            self.observations = self.observations[-self._max_observations:]
        # 自动注册节点
        for var in event.variables:
            if var not in self.graph.nodes:
                self.graph.add_node(var)

    def observe_many(self, events: List[Dict]):
        """批量记录观测"""
        for e in events:
            self.observe(e)

    # ── 因果发现（PC 算法简化版）──────────────────────────

    def discover(self, alpha: float = 0.05,
                 temporal_order: Optional[List[str]] = None) -> Dict:
        """
        从观测数据自动发现因果图

        基于 PC 算法:
        1. 初始化：所有变量对之间有潜在边（完全图）
        2. 骨架学习：用条件独立性测试删除非因果边
        3. 方向定向：v-structure 检测 + ANM 评分 / 时序先验
        4. 效应量估计：计算每条边的效应方向和大小
        5. 混杂识别：检测共同父节点（真正的混杂变量定义）

        Args:
            alpha: 条件独立性检验显著性水平
            temporal_order: 变量的时序顺序（早→晚），如 ["work","sleep","mood"]。
                若提供，优先用时序定向；否则用 ANM 综合评分。

        Returns:
            {"edges_discovered": int, "nodes": int, "confounders": [...], "details": [...]}
        """
        if len(self.observations) < 10:
            return {"edges_discovered": 0, "reason": "观测不足（< 10）", "confounders": []}

        nodes = list(self.graph.nodes)
        if len(nodes) < 2:
            return {"edges_discovered": 0, "reason": "变量不足", "confounders": []}

        # FIX: 用邻接表记录无向骨架，便于 v-structure 检测
        skeleton = defaultdict(set)  # node -> set of neighbors
        removed_by = {}  # (x,y) -> z that made them independent (confounder)

        # Step 1+2: 骨架学习 — 对每对变量测条件独立性
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                x, y = nodes[i], nodes[j]
                is_ind, p = _conditional_independence_test(
                    self.observations, x, y, [], alpha
                )
                if is_ind:
                    continue
                # 测试给定其他变量后是否独立（排除混淆）
                other_nodes = [n for n in nodes if n not in (x, y)]
                confounded = False
                for z in other_nodes[:8]:
                    is_cond_ind, _ = _conditional_independence_test(
                        self.observations, x, y, [z], alpha
                    )
                    if is_cond_ind:
                        # FIX: z 是 x 和 y 的共同原因（混杂变量），标记并记录
                        self._confounders.add(z)
                        removed_by[(x, y)] = z
                        confounded = True
                        break
                if not confounded:
                    skeleton[x].add(y)
                    skeleton[y].add(x)

        # Step 3: v-structure 方向定向
        # 找所有 unshielded triples: X-Z-Y where X,Y not adjacent
        self.graph.edges.clear()
        if self.graph._nx is not None:
            self.graph._nx.clear_edges()

        directed = set()  # (parent, child) pairs
        v_structures = 0

        # FIX: v-structure 检测 — 如果 X⊥Y|Z 但 X-Z 和 Z-Y 都存在，定向为 X→Z<-Y
        # removed_by 记录的就是 "X,Y 在给定 Z 后独立" 的情况
        for (x, y), z in removed_by.items():
            # x 和 y 都连 z，但 x-y 不连 → collider
            if y in skeleton[z] and x in skeleton[z]:
                directed.add((x, z))
                directed.add((y, z))
                v_structures += 1

        # FIX: 对剩余无向边，优先用时序先验，否则用 ANM 综合评分
        temporal_rank = None
        if temporal_order:
            temporal_rank = {v: i for i, v in enumerate(temporal_order)}

        for x in list(skeleton.keys()):
            for y in list(skeleton[x]):
                if (x, y) in directed or (y, x) in directed:
                    continue
                # 选择方向
                if temporal_rank is not None and x in temporal_rank and y in temporal_rank:
                    # 时序先验：早发生的 → 晚发生的
                    if temporal_rank[x] <= temporal_rank[y]:
                        parent, child = x, y
                    else:
                        parent, child = y, x
                else:
                    # ANM 综合评分：正值→X→Y，负值→Y→X
                    score = self._anm_direction_score(x, y)
                    if score >= 0:
                        parent, child = x, y
                    else:
                        parent, child = y, x
                # 避免反向边（防止环）
                if (child, parent) not in directed:
                    directed.add((parent, child))

        # 应用定向边到因果图
        for (parent, child) in directed:
            effect_size = self._estimate_effect(parent, child)
            confidence = 0.9
            self.graph.add_edge(
                parent, child, confidence=confidence,
                effect_size=effect_size, edge_type="direct",
                n_observations=len(self.observations),
            )

        # FIX Step 5: 混杂变量识别 — 检测共同父节点
        # 真正的混杂变量 = 两个变量 X,Y 的共同原因 Z（Z→X 且 Z→Y）
        # 之前 removed_by 记录的只是"条件独立"情况，这里补充"共同父节点"检测
        node_list = list(self.graph.nodes)
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                x, y = node_list[i], node_list[j]
                px = set(self.graph.get_parents(x))
                py = set(self.graph.get_parents(y))
                common = px & py  # 共同父节点 = 混杂变量
                for z in common:
                    if z not in self._confounders:
                        self._confounders.add(z)
                        # 标记相关边为 confounded
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
            "direction_method": "temporal" if temporal_rank else "anm",
            "details": [
                {"edge": k, **asdict(v)}
                for k, v in self.graph.edges.items()
            ],
        }
        logger.info(f"[{self.name}] 发现 {len(self.graph.edges)} 条因果关系 "
                    f"(混杂变量: {self._confounders}, v-structures: {v_structures})")
        if self.persist_path:
            self._save()
        return result

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

    def _anm_direction_score(self, x: str, y: str) -> float:
        """
        ANM 方向评分：正值倾向于 X→Y，负值倾向于 Y→X。

        综合三个信号：
        1. ANM 残差独立性差异（Additive Noise Model, Hoyer 2009）
           - 拟合 Y=f(X)+N，若 N 独立于 X 则 X→Y 合理
           - |Cor(X, res_xy)| 越小，X→Y 越合理
        2. 非高斯性差异（LiNGAM, Shimizu 2006）
           - 原因通常更非高斯（偏度更大）
        3. 方差比启发式
           - 原因方差通常 >= 结果方差（系数 |a|<1 时）
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

        # 1. ANM: X→Y 方向，残差独立性
        _, _, res_y = _linear_fit(xs, ys)
        indep_xy = abs(_safe_corr(xs, res_y))

        # ANM: Y→X 方向，残差独立性
        _, _, res_x = _linear_fit(ys, xs)
        indep_yx = abs(_safe_corr(ys, res_x))

        # 正分 = X→Y 残差更独立（indep_yx 大说明反向不独立）
        anm_score = indep_yx - indep_xy

        # 2. 非高斯性：原因偏度更大
        skew_x = abs(_skewness(xs))
        skew_y = abs(_skewness(ys))
        nongauss_score = skew_x - skew_y

        # 3. 方差比：log(x_var/y_var)，正值→X 方差大→倾向 X→Y
        var_score = math.log(x_var / y_var + 1e-10)

        # 综合评分（ANM 权重最高，非高斯次之，方差比辅助）
        total = anm_score * 2.0 + nongauss_score * 0.5 + var_score * 0.2
        return total

    def _total_effect_via_paths(self, do_var: str, target_var: str,
                                max_depth: int = 6) -> float:
        """
        沿所有 do_var→target_var 的有向路径累加效应量。

        路径效应 = 路径上各边 effect_size 的乘积。
        总效应 = 所有路径效应之和（线性近似）。

        do-calculus: 干预切断了 do_var 的入边，只保留出边传播。
        """
        if do_var == target_var:
            return 1.0
        total = 0.0
        # DFS: (current_node, effect_so_far, depth, visited_set)
        stack = [(do_var, 1.0, 0, {do_var})]
        while stack:
            node, eff, depth, visited = stack.pop()
            if depth >= max_depth:
                continue
            for child in self.graph.get_children(node):
                if child in visited:
                    continue  # DAG 保证无环，但防御性检查
                edge = self.graph.edges.get(f"{node}->{child}")
                if not edge or edge.effect_size == 0:
                    continue
                new_eff = eff * edge.effect_size
                if child == target_var:
                    total += new_eff
                else:
                    stack.append((child, new_eff, depth + 1, visited | {child}))
        return total

    # ── Level 1: 关联层 ───────────────────────────────────

    def query_cause(self, effect: str, top_k: int = 5) -> List[Dict]:
        """
        查询某结果的可能原因

        返回最可能的 top_k 个原因，含置信度和效应量
        """
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
        # 加入间接原因
        direct_set = set(parents)
        for a in ancestors:
            if a not in direct_set:
                # 找路径
                causes.append({
                    "cause": a,
                    "confidence": 0.3,  # 间接原因置信度低
                    "effect_size": 0.0,
                    "relation": "indirect",
                })
        # 排序
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

    # ── Level 2: 干预层（do-calculus）──────────────────────

    def intervene(self, do_var: str, do_value: float,
                  target_var: str, n_samples: int = 100) -> Dict:
        """
        干预推理: P(target | do(do_var=do_value))

        用 do-calculus: 切断 do_var 的入边，强制设值，预测 target

        Args:
            do_var: 被干预的变量
            do_value: 强制设置的值
            target_var: 要预测的变量
            n_samples: 蒙特卡洛样本数

        Returns:
            {expected, std, samples, intervention_effect}
        """
        # 构建干预后的图（移除 do_var 的入边）
        if not self.graph.edges:
            # 无因果图，用相关性近似
            return self._intervene_correlation(do_var, do_value, target_var)

        # 找 do_var 到 target_var 的路径
        parents_of_do = self.graph.get_parents(do_var)
        children_of_do = self.graph.get_children(do_var)

        # 用线性模型近似: target ≈ a * do_var + b * (其他父变量) + noise
        # 干预时，do_var 设固定值，其他父变量用观测均值
        target_obs = [
            o.get(target_var) for o in self.observations
            if o.get(target_var) is not None
        ]
        if not target_obs:
            return {"expected": None, "reason": "无目标变量数据"}

        target_mean = statistics.mean([float(t) for t in target_obs])
        target_std = statistics.stdev([float(t) for t in target_obs]) if len(target_obs) > 1 else 1.0

        # FIX: 用路径传播估计 do_var 对 target 的总效应
        # 沿所有 do_var→target_var 的有向路径累加效应量（而非只看直接子节点）
        effect = self._total_effect_via_paths(do_var, target_var)

        # 干预效应 = do_value 相对均值的偏移 * 效应量
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
            "note": "基于相关性近似（无因果图）",
        }

    # ── Level 3: 反事实层 ──────────────────────────────────

    def counterfactual(self, observed_x: float, observed_y: float,
                       counterfactual_x: float,
                       cause_var: str = "", effect_var: str = "",
                       noise_samples: int = 50) -> Dict:
        """
        反事实推理: 如果当初 X 是 cf_x 而非 obs_x，Y 会是多少？

        三步过程 (Pearl):
        1. Abduction: 从观察推断噪声 U = Y - f(X)
        2. Action: 修改 X = cf_x
        3. Prediction: Y_cf = f(cf_x, U)

        简化为线性模型: Y = a*X + b + U

        Args:
            observed_x: 实际发生的 X
            observed_y: 实际发生的 Y
            counterfactual_x: 假设的 X
            cause_var: 原因变量名（默认自动选最强因果边）
            effect_var: 结果变量名
            noise_samples: 噪声样本数（多噪声求平均，提升稳定性）

        Returns:
            {counterfactual_y, original_y, delta, confidence}
        """
        # FIX: 参数化变量名 — 若未指定，自动从因果图选效应量最大的边
        if not cause_var or not effect_var:
            best_eff, best_cause, best_effect = 0.0, "", ""
            for e in self.graph.edges.values():
                if abs(e.effect_size) > abs(best_eff):
                    best_eff, best_cause, best_effect = e.effect_size, e.source, e.target
            if best_cause and best_effect:
                cause_var, effect_var = best_cause, best_effect
            else:
                cause_var, effect_var = "x", "y"  # fallback

        # 估计线性模型参数 Y = a*X + b
        paired = [
            (o.get(cause_var), o.get(effect_var))
            for o in self.observations
            if o.get(cause_var) is not None and o.get(effect_var) is not None
        ]
        if len(paired) < 5:
            # 数据不足，用简单差分
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
        a, b, _ = _linear_fit(xs, ys)

        # Step 1: Abduction
        predicted_y = a * observed_x + b
        u = observed_y - predicted_y

        # Step 2+3: Action + Prediction（多噪声样本）
        rng = random.Random(self._projection_seed + 1)
        cf_samples = []
        for _ in range(noise_samples):
            # 加入小扰动模拟不确定性
            u_sample = u + rng.gauss(0, 0.05 * abs(u) + 0.01)
            cf_y = a * counterfactual_x + b + u_sample
            cf_samples.append(cf_y)

        cf_y = statistics.mean(cf_samples)
        cf_std = statistics.stdev(cf_samples) if len(cf_samples) > 1 else 0.0
        delta = cf_y - observed_y

        # 置信度：基于样本量和拟合度
        confidence = min(1.0, len(paired) / 50.0)

        self._counterfactuals += 1
        return {
            "counterfactual_y": cf_y,
            "counterfactual_y_std": cf_std,
            "original_y": observed_y,
            "delta": delta,
            "cause_var": cause_var,
            "effect_var": effect_var,
            "model": {"slope": a, "intercept": b, "residual_u": u},
            "confidence": confidence,
        }

    def counterfactual_multi(self, observed: Dict[str, float],
                             counterfactual: Dict[str, float],
                             target: str,
                             noise_samples: int = 50) -> Dict:
        """
        多变量反事实推理: 如果当初多个 X 是 cf 值，target 会是多少？

        线性模型: Y = a1*X1 + a2*X2 + ... + b + U

        三步:
        1. Abduction: U = observed[target] - Σ(ai * observed[Xi]) - b
        2. Action: 修改 Xi = counterfactual[Xi]
        3. Prediction: Y_cf = Σ(ai * cf[Xi]) + b + U

        Args:
            observed: 实际观测值 {var: value, ...}，必须含 target
            counterfactual: 反事实假设值 {var: value, ...}
            target: 要预测的目标变量
            noise_samples: 噪声样本数

        Returns:
            {counterfactual_y, original_y, delta, coefficients, confidence}
        """
        if target not in observed:
            return {"error": f"target '{target}' 不在 observed 中"}

        # 确定原因变量：target 的所有父节点（从因果图）
        parents = self.graph.get_parents(target)
        if not parents:
            # fallback: 用所有与 target 相关的变量
            parents = [v for v in observed if v != target]

        # 收集有数据的变量
        cause_vars = [v for v in parents if v in observed or v in counterfactual]
        if not cause_vars:
            return {"error": "无可用原因变量"}

        # 多元线性回归: target = Σ(ai * Xi) + b
        # 用正规方程（最小二乘）求解
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

        # 加截距项
        X = [[1.0] + row for row in rows]
        y = target_vals
        coeffs = _least_squares(X, y)
        if coeffs is None:
            return {"error": "多元回归求解失败"}

        b = coeffs[0]
        weights = coeffs[1:]

        # Step 1: Abduction
        pred_obs = b + sum(weights[i] * float(observed.get(cause_vars[i], 0))
                          for i in range(n_feats))
        u = float(observed[target]) - pred_obs

        # Step 2+3: Action + Prediction
        pred_cf = b + sum(
            weights[i] * float(counterfactual.get(cause_vars[i], observed.get(cause_vars[i], 0)))
            for i in range(n_feats)
        )
        rng = random.Random(self._projection_seed + 2)
        cf_samples = []
        for _ in range(noise_samples):
            u_sample = u + rng.gauss(0, 0.05 * abs(u) + 0.01)
            cf_samples.append(pred_cf + u_sample)

        cf_y = statistics.mean(cf_samples)
        cf_std = statistics.stdev(cf_samples) if len(cf_samples) > 1 else 0.0
        delta = cf_y - float(observed[target])
        confidence = min(1.0, len(rows) / (n_feats * 10 + 10))

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
            "confidence": confidence,
        }

    # ── 量子核相似度（模拟 quantum_dim 投影）──────────────

    def _quantum_project(self, var: str) -> List[float]:
        """把变量名投影到 quantum_dim 维量子态空间（随机投影模拟）"""
        if var in self._projection_cache:
            return self._projection_cache[var]
        rng = random.Random(hash(var) & 0xFFFFFFFF)
        vec = [rng.gauss(0, 1) for _ in range(self.quantum_dim)]
        norm = math.sqrt(sum(v*v for v in vec))
        vec = [v / norm for v in vec]
        self._projection_cache[var] = vec
        return vec

    def quantum_similarity(self, var_a: str, var_b: str) -> float:
        """量子核相似度：两个变量在量子态空间的余弦相似度"""
        va = self._quantum_project(var_a)
        vb = self._quantum_project(var_b)
        return sum(a*b for a, b in zip(va, vb))

    # ── 统一入口（供 agi_subscriber.py 调用）──────────────

    def predict(self, query: Dict, mode: str = "auto", top_k: int = 3) -> Dict:
        """
        统一推理入口

        Args:
            query: {
                "type": "state_analysis" | "cause_query" | "intervention" | "counterfactual",
                "needs": {...}, "emotion": "...", "self_presence": 0.5,
                "effect": "某变量", "cause": "某变量", "do_var": "...", ...
            }
            mode: "auto" | "causal" | "statistical"
            top_k: 返回结果数

        Returns:
            推理结果字典
        """
        qtype = query.get("type", "state_analysis")

        # 自动发现（如果还没发现过）
        if mode == "auto" and not self.graph.edges and len(self.observations) >= 10:
            self.discover()

        if qtype == "cause_query":
            effect = query.get("effect", "")
            causes = self.query_cause(effect, top_k)
            return {"type": "causes", "causes": causes, "effect": effect}

        elif qtype == "effect_query":
            cause = query.get("cause", "")
            effects = self.query_effect(cause, top_k)
            return {"type": "effects", "effects": effects, "cause": cause}

        elif qtype == "intervention":
            result = self.intervene(
                query.get("do_var", ""), float(query.get("do_value", 0)),
                query.get("target_var", ""), query.get("n_samples", 100),
            )
            return {"type": "intervention", "result": result}

        elif qtype == "counterfactual":
            result = self.counterfactual(
                float(query.get("observed_x", 0)),
                float(query.get("observed_y", 0)),
                float(query.get("counterfactual_x", 0)),
                query.get("noise_samples", 50),
            )
            return {"type": "counterfactual", "result": result}

        # 默认: state_analysis
        needs = query.get("needs", {})
        emotion = query.get("emotion", "")
        snapshot = {"self_presence": query.get("self_presence", 0.5)}

        # 分析当前状态：找最强需求 + 预测情绪走向
        if needs:
            dominant_need = max(needs, key=lambda k: needs.get(k, 0))
            dominant_val = needs.get(dominant_need, 0)
        else:
            dominant_need = "unknown"
            dominant_val = 0.0

        # 查情感的原因
        emotion_causes = self.query_cause(emotion, top_k) if emotion else []

        # 量子相似度推荐相关变量
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

    # ── 持久化 ────────────────────────────────────────────

    def _save(self):
        """保存因果图和观测到文件"""
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
        }
        try:
            self.persist_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
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


    def _bic_score(self, parent_set: Set[str], target: str) -> float:
        """
        贝叶斯信息准则 (BIC) 评分: 评估给定的父节点集合对目标的解释力。

        BIC = -2 * log_likelihood + k * log(n)
        越高越好(惩罚复杂模型)。

        用于因果发现中比较不同父节点组合的优劣,
        替代纯条件独立性测试,能避免过拟合。
        """
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
            return -1e10  # 数据不足
        # 线性回归拟合
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
        # 残差平方和
        rss = sum((target_vals[i] - preds[i]) ** 2 for i in range(n))
        sigma2 = rss / n if rss > 0 else 1e-10
        log_lik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1)
        # BIC = -2 * log_lik + k * log(n), 取负使越大越好
        k = len(parent_set) + 1  # 参数个数
        bic = log_lik - 0.5 * k * math.log(n)
        return bic


    def _normalize_anm_score(self, score: float) -> float:
        """
        ANM 评分归一化到 [-1, 1] 区间。

        原始 ANM 评分范围不固定(依赖数据尺度),
        归一化后便于跨不同变量对比较方向确定性。

        使用 tanh 压缩到 [-1, 1]:
        - 接近 1: 强烈倾向 X->Y
        - 接近 -1: 强烈倾向 Y->X
        - 接近 0: 方向不确定
        """
        return math.tanh(score)

    def _anm_direction_normalized(self, x: str, y: str) -> float:
        """归一化的 ANM 方向评分, 直接用于 discover() 方向决策"""
        raw = self._anm_direction_score(x, y)
        return self._normalize_anm_score(raw)


    def _r_squared(self, cause: str, effect: str) -> float:
        """
        计算 cause 对 effect 的 R^2 (决定系数)。

        R^2 = 1 - SS_res / SS_tot
        衡量 cause 能解释 effect 多少比例的方差。

        用于因果边置信度估计:
        R^2 越高,因果关系越确定。
        """
        paired = [(float(o.get(cause)), float(o.get(effect)))
                  for o in self.observations
                  if o.get(cause) is not None and o.get(effect) is not None]
        if len(paired) < 3:
            return 0.0
        xs = [p[0] for p in paired]
        ys = [p[1] for p in paired]
        a, b, residuals = _linear_fit(xs, ys)
        ss_res = sum(r ** 2 for r in residuals)
        y_mean = statistics.mean(ys)
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        if ss_tot == 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))


    def _normalize_anm_score(self, score: float) -> float:
        """ANM 评分归一化到 [-1, 1] 区间。"""
        return math.tanh(score)

    def _anm_direction_normalized(self, x: str, y: str) -> float:
        """归一化的 ANM 方向评分"""
        raw = self._anm_direction_score(x, y)
        return self._normalize_anm_score(raw)

    def status(self) -> Dict:
        """引擎状态"""
        return {
            "name": self.name,
            "quantum_dim": self.quantum_dim,
            "observations": len(self.observations),
            "causal_edges": len(self.graph.edges),
            "confounders": len(self._confounders),
            "inferences": self._inferences,
            "counterfactuals": self._counterfactuals,
            "discoveries": self._discoveries,
            "persisted": self.persist_path.exists() if self.persist_path else False,
        }