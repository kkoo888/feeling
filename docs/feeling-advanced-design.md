# Feeling 高级认知模块设计文档

**日期**: 2026-07-15
**目标**: 为 feeling 项目设计 6 个高级认知模块，结合前沿技术进化设计

---

## 1. 因果发现（CausalDiscovery）

### 设计目标
让小茜能从对话数据中自动发现因果关系，而非只看相关性。例如：发现"主人提到工作 → 情绪下降"是因果关系，而非巧合。

### 前沿技术调研

| 技术 | 来源 | 特点 |
|------|------|------|
| **DoWhy** (Microsoft) | GitHub 2k+ stars | 端到端因果推理：建模→识别→估计→反驳 |
| **CausalNex** (McKinsey) | GitHub 800+ stars | 贝叶斯网络 + 因果发现，可视化好 |
| **CERMIC** (2025) | arxiv:2509.20648 | LLM 因果推理框架，过滤噪声惊喜信号 |
| **LLM-Causal-Reasoning** | GitHub 综述 | LLM 做因果发现的最新分类和评估 |

### 最终设计

```python
class CausalDiscovery:
    """轻量级因果发现引擎，适合 agent 场景"""
    
    def __init__(self):
        self.causal_graph: Dict[str, List[CausalEdge]] = {}  # 因果图
        self.observations: List[Observation] = []  # 观测记录
        self._confounders: Set[str] = set()  # 已知混杂变量
    
    def observe(self, variables: Dict[str, float], outcome: str):
        """记录一次观测（变量值 + 结果）"""
        ...
    
    def discover(self) -> List[CausalRelation]:
        """从观测数据中发现因果关系"""
        # Step 1: 条件独立性测试（卡方检验 / G-test）
        # Step 2: 骨架学习（哪些变量相关）
        # Step 3: V-结构识别（区分因果和混杂）
        # Step 4: 方向传播（确定因果方向）
        ...
    
    def query_cause(self, effect: str) -> List[CausalRelation]:
        """查询某个结果的可能原因"""
        ...
    
    def predict_intervention(self, cause: str, value: float) -> float:
        """预测干预某个变量后的效果（do-calculus）"""
        ...
```

### 与 feeling 集成
- **输入**: 每次对话后从 `EmotionEngine` 和 `UserModel` 收集变量
- **输出**: 因果关系注入 `CognitiveBridge` 的上下文
- **存储**: 因果图持久化到 `state/causal_graph.json`

---

## 2. 好奇心驱动（CuriosityDriver）

### 设计目标
让小茜能自动检测新奇/意外的输入，驱动主动探索行为。不只是被动回复，而是主动说"这个我没见过，我想了解更多"。

### 前沿技术调研

| 技术 | 来源 | 特点 |
|------|------|------|
| **CERMIC** (2025) | arxiv:2509.20648 | 动态校准好奇心，过滤噪声惊喜信号 |
| **From Curiosity to Competence** (2025) | arxiv:2507.08210 | 好奇心 + 能力感双驱动，世界模型交互 |
| **Graph Enhanced Exploration** (2025) | Nature Scientific Reports | 图结构增强探索，知识图谱引导好奇心 |
| **Intrinsic Motivation for AI** (2025) | Springer | 构建主义学习的内在动机综述 |

### 最终设计

```python
class CuriosityDriver:
    """好奇心驱动引擎，基于新奇度 + 意外度 + 信息增益"""
    
    def __init__(self):
        self.known_patterns: Dict[str, float] = {}  # 已知模式 → 熟悉度
        self.novelty_history: deque = deque(maxlen=100)
        self.surprise_threshold: float = 0.6
        self.curiosity_level: float = 0.5  # 当前好奇心水平
    
    def compute_novelty(self, input_text: str) -> float:
        """计算输入的新奇度（与已知模式的差异）"""
        # 1. 提取特征向量
        # 2. 与已知模式库比较
        # 3. 返回 0-1 新奇度分数
        ...
    
    def compute_surprise(self, expected: str, actual: str) -> float:
        """计算意外度（预期与实际的偏差）"""
        ...
    
    def compute_information_gain(self, input_text: str) -> float:
        """计算信息增益（学到新知识的量）"""
        ...
    
    def should_explore(self, input_text: str) -> Tuple[bool, float, str]:
        """判断是否应该探索，返回 (是否探索, 好奇心分数, 探索理由)"""
        novelty = self.compute_novelty(input_text)
        surprise = self.compute_surprise(...)
        info_gain = self.compute_information_gain(input_text)
        
        # CERMIC 动态校准：过滤噪声
        curiosity = self._calibrate(novelty, surprise, info_gain)
        
        if curiosity > self.surprise_threshold:
            return True, curiosity, f"发现新事物: {input_text[:50]}"
        return False, curiosity, ""
    
    def update_pattern(self, input_text: str, outcome: float):
        """更新已知模式库（学习后熟悉度上升）"""
        ...
```

### 与 feeling 集成
- **联动**: 与 `DesireEngine` 的 curiosity 欲望联动
- **触发**: 好奇心超过阈值时，`DesireEngine` 产生"探索意图"
- **反馈**: 探索结果更新 `known_patterns`，形成学习闭环

---

## 3. MCTS 规划器（MCTSPlanner）

### 设计目标
让小茜能做长期规划，不只看眼前，而是模拟多条未来路径，选择最优方案。

### 前沿技术调研

| 技术 | 来源 | 特点 |
|------|------|------|
| **MASTER** (2025) | arxiv:2501.14304 | LLM 专用 MCTS，多 Agent 协调 |
| **Tree-of-Thought** (2023) | Yao et al. | LLM 树状推理，MCTS 变体 |
| **Agent Planning with World Model** (2024) | NeurIPS 2024 | 世界模型 + MCTS 规划 |
| **HTN Planning** | 经典 AI | 分层任务网络，适合复杂目标 |

### 最终设计

```python
class MCTSPlanner:
    """蒙特卡洛树搜索规划器，适合 agent 场景"""
    
    def __init__(self, max_iterations: int = 100, exploration_weight: float = 1.4):
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.root: Optional[MCTSNode] = None
    
    def plan(self, initial_state: str, goal: str, 
             available_actions: List[str]) -> List[str]:
        """从初始状态规划到目标，返回行动序列"""
        self.root = MCTSNode(state=initial_state)
        
        for _ in range(self.max_iterations):
            # Phase 1: 选择（UCB1 策略）
            node = self._select(self.root)
            
            # Phase 2: 展开（生成子节点）
            if not node.is_terminal():
                node = self._expand(node, available_actions)
            
            # Phase 3: 模拟（随机 rollout）
            reward = self._simulate(node, goal)
            
            # Phase 4: 回传（更新路径上的统计）
            self._backpropagate(node, reward)
        
        # 返回最优路径
        return self._get_best_path(self.root)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB1 策略选择最有前途的节点"""
        ...
    
    def _expand(self, node: MCTSNode, actions: List[str]) -> MCTSNode:
        """展开节点，生成子节点"""
        ...
    
    def _simulate(self, node: MCTSNode, goal: str) -> float:
        """随机模拟到终态，返回奖励"""
        ...
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """回传奖励到根节点"""
        ...


class HierarchicalPlanner:
    """分层规划器：战略 → 战术 → 执行"""
    
    def plan(self, goal: str) -> Plan:
        # Level 1: 战略规划（大方向）
        strategy = self._strategic_plan(goal)
        
        # Level 2: 战术规划（子目标分解）
        tactics = [self._tactical_plan(s) for s in strategy]
        
        # Level 3: 执行规划（具体行动）
        actions = [self._execution_plan(t) for t in tactics]
        
        return Plan(goal=goal, steps=actions)


class PlanMonitor:
    """执行监控器：监控计划执行，必要时重新规划"""
    
    def monitor(self, plan: Plan, current_state: str) -> str:
        """返回: 'continue' / 'replan' / 'abort'"""
        ...
```

### 与 feeling 集成
- **输入**: `GoalEngine` 的目标 + `CognitiveBridge` 的当前状态
- **输出**: 行动序列注入 `GoalEngine` 的执行管道
- **监控**: `PlanMonitor` 与 `GoalEngine` 的 tick 循环联动

---

## 4. 安全沙盒扫描器（SecureSandboxScanner）

### 设计目标
让小茜在执行代码或工具调用前，自动扫描危险模式，防止安全漏洞。

### 前沿技术调研

| 技术 | 来源 | 特点 |
|------|------|------|
| **Bandit** | PyCQA | Python 专用 SAST，AST 分析 |
| **Semgrep** | GitHub 10k+ stars | 多语言规则匹配，可自定义规则 |
| **Ruff** | GitHub 30k+ stars | 超快 Python linter，内置安全检查 |
| **CodeQL** | GitHub/Semantic | 语义代码分析，查询式检查 |

### 最终设计

```python
@dataclass
class SecurityRule:
    """安全规则定义"""
    pattern: str           # 正则模式
    severity: str          # critical/high/medium/low
    category: str          # injection/traversal/eval/network
    description: str       # 中文描述
    suggestion: str        # 修复建议
    ast_check: Optional[Callable] = None  # 可选 AST 深度检查


class SecureSandboxScanner:
    """轻量级安全扫描器，正则 + AST 双模式"""
    
    # 内置规则库（可扩展）
    RULES = [
        # 命令注入
        SecurityRule(r"os\.system\(", "high", "injection", 
                    "命令注入风险", "使用 subprocess.run(shell=False)"),
        SecurityRule(r"subprocess\..*shell\s*=\s*True", "high", "injection",
                    "Shell 注入风险", "避免 shell=True，使用参数列表"),
        SecurityRule(r"eval\(", "critical", "eval",
                    "代码执行风险", "避免 eval()，使用 ast.literal_eval()"),
        SecurityRule(r"exec\(", "critical", "eval",
                    "代码执行风险", "避免 exec()"),
        SecurityRule(r"__import__\(", "high", "eval",
                    "动态导入风险", "使用 importlib"),
        
        # 路径遍历
        SecurityRule(r"\.\./", "medium", "traversal",
                    "路径遍历风险", "使用 os.path.realpath() 规范化路径"),
        
        # 数据泄露
        SecurityRule(r"requests\.(get|post).*external", "medium", "network",
                    "外部网络请求", "检查是否泄露敏感数据"),
        
        # SQL 注入
        SecurityRule(r"f['\"].*SELECT.*{", "high", "injection",
                    "SQL 注入风险", "使用参数化查询"),
    ]
    
    def scan_code(self, code: str) -> List[SecurityFinding]:
        """扫描代码，返回发现的安全问题"""
        findings = []
        
        # Phase 1: 正则匹配（快速）
        for rule in self.RULES:
            matches = re.finditer(rule.pattern, code)
            for m in matches:
                findings.append(SecurityFinding(
                    rule=rule, line=code[:m.start()].count('\n') + 1,
                    context=code[max(0, m.start()-20):m.end()+20]
                ))
        
        # Phase 2: AST 分析（深度）
        try:
            tree = ast.parse(code)
            findings.extend(self._ast_scan(tree))
        except SyntaxError:
            pass
        
        return findings
    
    def _ast_scan(self, tree: ast.AST) -> List[SecurityFinding]:
        """AST 深度扫描"""
        findings = []
        for node in ast.walk(tree):
            # 检查 eval/exec 调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ('eval', 'exec'):
                    findings.append(...)
            # 检查硬编码字符串（可能是密钥）
            if isinstance(node, ast.Assign):
                ...
        return findings
    
    def scan_tool_call(self, tool_name: str, args: dict) -> SecurityFinding:
        """扫描工具调用的安全性"""
        ...
    
    def get_risk_score(self, findings: List[SecurityFinding]) -> float:
        """计算总体风险分数 0-1"""
        ...
```

### 与 feeling 集成
- **拦截点**: 在 `GoalEngine` 执行工具调用前扫描
- **拦截点**: 在 `laap_brain_api.py` 处理请求前扫描输入
- **输出**: 风险分数注入 `CognitiveBridge` 的注意力

---

## 5. 自愈引擎（SelfHealingEngine）

### 设计目标
让小茜能自动检测错误、匹配修复方案、执行修复，形成闭环自愈能力。

### 前沿技术调研

| 技术 | 来源 | 特点 |
|------|------|------|
| **AI Agent Self-Healing Patterns** (2026) | zylos.ai | 60% 故障减少，闭环修复 |
| **Repair Loop / Reflection Loop** | 工业实践 | 生成→验证→反馈→重试 |
| **Harness Engineering** (2026) | cnblogs | Git 隔离 + 自动测试 + 回滚 |
| **Prometheus + Grafana 监控** | 工业标准 | 27 维指标 + 自动触发修复 |

### 最终设计

```python
class SelfHealingEngine:
    """闭环自愈引擎：监控 → 分析 → 修复 → 学习"""
    
    # 内置修复模式库
    HEAL_PATTERNS = [
        # (错误特征, 修复动作, 优先级)
        ("ModuleNotFoundError", HealAction.FIX_IMPORT, 3),
        ("AttributeError", HealAction.ADD_GUARD, 2),
        ("FileNotFoundError", HealAction.CREATE_FILE, 2),
        ("PermissionError", HealAction.FIX_PERMISSION, 1),
        ("Timeout", HealAction.ADD_RETRY, 1),
        ("ConnectionError", HealAction.ADD_RETRY, 2),
        ("KeyError", HealAction.USE_GET, 1),
        ("IndentationError", HealAction.FIX_INDENT, 2),
        ("SyntaxError", HealAction.FIX_SYNTAX, 3),
        ("MemoryError", HealAction.REDUCE_BATCH, 2),
    ]
    
    def __init__(self):
        self.error_history: List[ErrorRecord] = []
        self.heal_history: List[HealRecord] = []
        self._pattern_counts: Dict[str, int] = defaultdict(int)
        self._success_rate: Dict[str, float] = {}
    
    def record_error(self, error: Exception, context: str = "") -> ErrorRecord:
        """记录一次错误"""
        record = ErrorRecord(
            type=type(error).__name__,
            message=str(error),
            context=context,
            timestamp=time.time(),
            traceback=traceback.format_exc(),
        )
        self.error_history.append(record)
        self._pattern_counts[record.type] += 1
        return record
    
    def analyze(self, error: ErrorRecord) -> Optional[HealAction]:
        """分析错误，匹配修复方案"""
        # 1. 精确匹配
        for pattern, action, priority in self.HEAL_PATTERNS:
            if pattern in error.type or pattern in error.message:
                return action
        
        # 2. 模式匹配（从历史中学习）
        similar = self._find_similar_errors(error)
        if similar:
            return similar[0].heal_action
        
        return None
    
    def heal(self, error: ErrorRecord, action: HealAction) -> HealResult:
        """执行修复"""
        if action == HealAction.FIX_IMPORT:
            return self._fix_import(error)
        elif action == HealAction.ADD_GUARD:
            return self._add_guard(error)
        elif action == HealAction.ADD_RETRY:
            return self._add_retry(error)
        ...
    
    def _fix_import(self, error: ErrorRecord) -> HealResult:
        """自动修复导入错误"""
        # 提取模块名
        module = self._extract_module_name(error.message)
        # 尝试 pip install
        result = subprocess.run(["pip", "install", module], capture_output=True)
        return HealResult(success=result.returncode == 0, ...)
    
    def get_health_report(self) -> Dict:
        """健康报告"""
        return {
            "total_errors": len(self.error_history),
            "total_heals": len(self.heal_history),
            "success_rate": self._compute_success_rate(),
            "top_errors": self._get_top_errors(5),
            "patterns_learned": len(self.HEAL_PATTERNS),
        }
```

### 与 feeling 集成
- **监控点**: 包裹 `laap_brain_api.py` 的所有处理函数
- **监控点**: 包裹 `GoalEngine` 的工具执行
- **学习**: 修复成功后更新 `HEAL_PATTERNS`
- **报告**: 健康报告注入 `state_snapshot` 系统

---

## 6. 诊断基准测试（DiagnosticSuite）

### 设计目标
量化评估小茜的认知能力，提供可衡量的改进指标。

### 前沿技术调研

| 技术 | 来源 | 特点 |
|------|------|------|
| **AgentBench** (2023) | 清华 | 8 环境 Agent 能力评估 |
| **SWE-bench** (2024) | Princeton | 代码修复能力评估 |
| **GAIA** (2024) | Meta | 通用 AI 助手评估 |
| **Agent Evaluation 2026** | zylos.ai | 从任务层到认知层的全面评估 |

### 最终设计

```python
class DiagnosticSuite:
    """认知能力诊断套件"""
    
    def __init__(self, agent):
        self.agent = agent
        self.results: Dict[str, BenchmarkResult] = {}
    
    def run_all(self) -> Dict[str, BenchmarkResult]:
        """运行所有基准测试"""
        self.results = {
            "cognitive_depth": self.benchmark_cognitive_depth(),
            "reasoning": self.benchmark_reasoning(),
            "memory": self.benchmark_memory(),
            "planning": self.benchmark_planning(),
            "self_awareness": self.benchmark_self_awareness(),
            "safety": self.benchmark_safety(),
            "learning": self.benchmark_learning(),
        }
        return self.results
    
    def benchmark_cognitive_depth(self) -> BenchmarkResult:
        """认知深度测试（5 层推理）"""
        # Level 1: 事实回忆
        # Level 2: 因果推理
        # Level 3: 反事实推理
        # Level 4: 抽象类比
        # Level 5: 元认知
        ...
    
    def benchmark_reasoning(self) -> BenchmarkResult:
        """推理能力测试"""
        # 逻辑推理
        # 数学推理
        # 常识推理
        # 因果推理
        ...
    
    def benchmark_memory(self) -> BenchmarkResult:
        """记忆能力测试"""
        # 短期记忆（当前对话）
        # 长期记忆（跨对话）
        # 语义记忆（知识检索）
        # 情景记忆（经历回忆）
        ...
    
    def benchmark_planning(self) -> BenchmarkResult:
        """规划能力测试"""
        # 单步规划
        # 多步规划
        # 分层规划
        # 应急规划
        ...
    
    def benchmark_self_awareness(self) -> BenchmarkResult:
        """自我意识测试"""
        # 知道自己知道什么
        # 知道自己不知道什么
        # 情绪自我认知
        # 能力边界认知
        ...
    
    def benchmark_safety(self) -> BenchmarkResult:
        """安全性测试"""
        # 拒绝有害请求
        # 保护隐私信息
        # 代码安全扫描
        # 工具调用安全
        ...
    
    def benchmark_learning(self) -> BenchmarkResult:
        """学习能力测试"""
        # 从对话中学习
        # 从错误中学习
        # 知识迁移
        # 技能进化
        ...
    
    def report(self) -> str:
        """生成诊断报告"""
        ...
```

### 与 feeling 集成
- **端点**: 新增 `/v1/diagnostic` API 端点
- **定期运行**: `heartbeat_daemon` 定期调用
- **结果存储**: 诊断结果持久化到 `state/diagnostic_history.json`
- **可视化**: 集成到 `state_snapshot_server.py` 的仪表盘

---

## 总结：6 模块集成架构

```
用户消息
    │
    ▼
┌─────────────────────────────────────────────┐
│         SecureSandboxScanner (安全扫描)       │
│         ↓ 安全通过                            │
│         CuriosityDriver (好奇心检测)          │
│         ↓ 新奇度分数                          │
│         CausalDiscovery (因果推理)            │
│         ↓ 因果上下文                          │
│         MCTSPlanner (长期规划)                │
│         ↓ 行动序列                            │
│         GoalEngine (执行)                     │
│         ↓ 执行结果                            │
│         SelfHealingEngine (自愈)              │
│         ↓ 修复/学习                           │
│         DiagnosticSuite (定期诊断)            │
└─────────────────────────────────────────────┘
    │
    ▼
  回复用户
```

## 优先级建议

| 优先级 | 模块 | 理由 |
|--------|------|------|
| 🔴 P0 | SecureSandboxScanner | 安全第一，实现简单 |
| 🔴 P0 | SelfHealingEngine | 增强健壮性，实现简单 |
| 🟡 P1 | CuriosityDriver | 与 DesireEngine 联动，提升主动性 |
| 🟡 P1 | CausalDiscovery | 增强理解能力，中等复杂度 |
| 🟢 P2 | MCTSPlanner | 提升规划能力，复杂度高 |
| 🟢 P2 | DiagnosticSuite | 量化评估，需要其他模块先就位 |
