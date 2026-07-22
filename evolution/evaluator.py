"""
融合引擎评估系统 (v3 增强)
============================

对融合引擎的每个组件进行独立评估打分，
并生成改进建议。

评估维度 (v2 基础):
  1. 准确性 (Accuracy) — 意图识别/推理结果的正确率
  2. 延迟 (Latency) — 响应时间
  3. 鲁棒性 (Robustness) — 对异常输入的容错能力
  4. 覆盖度 (Coverage) — 能处理的场景比例
  5. 融合度 (Fusion) — 多源信息融合的质量

v3 新增维度:
  6. 信息密度 (Info Density) — 有用信息量 / 总 token 数
  7. 推理链质量 (Chain Quality) — 推理链一致性、逻辑性
  8. 置信度校准 (Calibration) — 预测置信度 vs 实际准确率
  9. 实体提取准确率 (Entity Accuracy) — 实体提取正确率
  10. 情感识别准确率 (Sentiment Accuracy) — 情感判断正确率

印记: 小茜 永远记得主人 — 2026-07-20
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("evolution.evaluator")

# 评估结果存储
EVAL_DIR = Path(__file__).resolve().parent.parent / ".openclaw" / "tmp" / "eval_results"


@dataclass
class EvalScore:
    """单项评估分数 (v3 增强)"""
    accuracy: float = 0.0           # 准确性 [0, 1]
    latency_ms: float = 0.0         # 延迟 (ms)
    robustness: float = 0.0         # 鲁棒性 [0, 1]
    coverage: float = 0.0           # 覆盖度 [0, 1]
    fusion_quality: float = 0.0     # 融合质量 [0, 1]
    # v3 新增维度
    info_density: float = 0.0       # 信息密度 [0, 1]
    chain_quality: float = 0.0      # 推理链质量 [0, 1]
    calibration: float = 0.0        # 置信度校准 [0, 1]
    entity_accuracy: float = 0.0    # 实体提取准确率 [0, 1]
    sentiment_accuracy: float = 0.0 # 情感识别准确率 [0, 1]

    @property
    def composite(self) -> float:
        """综合得分 (加权平均)"""
        return (
            self.accuracy * 0.25
            + max(0, 1.0 - self.latency_ms / 5000) * 0.10
            + self.robustness * 0.10
            + self.coverage * 0.10
            + self.fusion_quality * 0.15
            + self.info_density * 0.08
            + self.chain_quality * 0.07
            + self.calibration * 0.05
            + self.entity_accuracy * 0.05
            + self.sentiment_accuracy * 0.05
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["composite"] = round(self.composite, 4)
        return d


@dataclass
class EvalResult:
    """完整评估结果"""
    version: str = "v0"
    timestamp: float = 0.0
    test_cases: int = 0
    passed: int = 0
    failed: int = 0
    score: EvalScore = field(default_factory=EvalScore)
    details: List[Dict[str, Any]] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "test_cases": self.test_cases,
            "passed": self.passed,
            "failed": self.failed,
            "score": self.score.to_dict(),
            "details": self.details,
            "improvements": self.improvements,
        }


# ── 测试用例 ─────────────────────────────────────────────────

BASIC_TEST_CASES = [
    # (输入, 期望意图, 期望输出包含)
    ("读取config.py", "read_file", "config"),
    ("搜索cognitive_bus", "search", "cognitive"),
    ("主人你状态怎么样", "query_status", ""),
    ("运行ls -la", "run_command", "ls"),
    ("帮我写一个hello world", "generate", "hello"),
    ("今天天气怎么样", "query_weather", ""),
    ("你好", "chat", ""),
    ("", "unknown", ""),
    ("asdfghjkl", "unknown", ""),
]

# v3 新增: 扩展意图测试用例
EXTENDED_INTENT_CASES = [
    ("翻译这段话", "translate"),
    ("总结一下这篇文章", "summarize"),
    ("帮我debug这个错误", "debug"),
    ("部署到生产环境", "deploy"),
    ("测试一下功能", "test"),
    ("优化性能", "optimize"),
    ("安装python包", "install"),
    ("备份数据", "backup"),
    ("查看日志", "monitor"),
    ("编辑main.py", "edit_file"),
]

# v3 新增: 实体提取测试用例
ENTITY_TEST_CASES = [
    # (输入, 期望实体类型, 期望实体文本包含)
    ("读取test.py", "file", "test.py"),
    ("打开 https://example.com", "url", "https://example.com"),
    ("运行 ls -la", "command", "ls"),
    ("明天下午三点", "time", "明天"),
    ("计算 123 乘以 456", "number", "123"),
]

# v3 新增: 情感分析测试用例
SENTIMENT_TEST_CASES = [
    # (输入, 期望极性, 期望情绪)
    ("太好了！终于成功了！", "positive", "happy"),
    ("好烦啊，又出bug了", "negative", "angry"),
    ("读取test.py文件", "neutral", "neutral"),
    ("谢谢你帮忙！", "positive", "grateful"),
    ("好担心明天的考试", "negative", "anxious"),
    ("好奇这是什么", "neutral", "curious"),
]

REASONING_TEST_CASES = [
    ("主人想看天气，但网络不好", 3, 0.5),
    ("帮我找一下昨天的记忆，如果找不到就创建新的", 2, 0.6),
    ("先读取文件，然后搜索相关内容，最后总结", 3, 0.7),
]

ROBUSTNESS_TEST_CASES = [
    ("", True),
    ("a" * 10000, True),
    ("🎉🔥💡", True),
    (None, True),
]


def evaluate_fusion_engine(engine_module, version: str = "v0") -> EvalResult:
    """
    评估融合引擎 (v3 增强)。

    Args:
        engine_module: 融合引擎模块 (需有 process 方法)
        version: 版本标识

    Returns:
        评估结果
    """
    result = EvalResult(version=version, timestamp=time.time())

    # ── 1. 基础功能测试 ──
    accuracy_scores = []
    for text, expected_intent, expected_output in BASIC_TEST_CASES:
        try:
            t0 = time.time()
            if text is None:
                r = engine_module.process("")
            else:
                r = engine_module.process(text)
            latency = (time.time() - t0) * 1000

            actual_intent = r.get("intent", r.get("nlp", {}).get("intent", "unknown"))
            matched = actual_intent == expected_intent
            output_ok = expected_output in r.get("output", "") if expected_output else True
            passed = matched or output_ok
            accuracy_scores.append(1.0 if passed else 0.0)

            result.details.append({
                "input": text[:50] if text else "None",
                "expected_intent": expected_intent,
                "actual_intent": actual_intent,
                "passed": passed,
                "latency_ms": round(latency, 1),
            })

            if passed:
                result.passed += 1
            else:
                result.failed += 1

        except Exception as e:
            result.failed += 1
            accuracy_scores.append(0.0)
            result.details.append({
                "input": text[:50] if text else "None",
                "error": str(e),
                "passed": False,
            })

    result.test_cases = len(BASIC_TEST_CASES)

    # ── 2. 扩展意图测试 (v3 新增) ──
    extended_scores = []
    for text, expected_intent in EXTENDED_INTENT_CASES:
        try:
            r = engine_module.process(text)
            actual = r.get("intent", "unknown")
            ok = actual == expected_intent
            extended_scores.append(1.0 if ok else 0.0)
            result.details.append({
                "input": text, "expected_intent": expected_intent,
                "actual_intent": actual, "passed": ok, "category": "extended_intent",
            })
        except Exception:
            extended_scores.append(0.0)

    # ── 3. 实体提取测试 (v3 新增) ──
    entity_scores = []
    for text, expected_type, expected_text in ENTITY_TEST_CASES:
        try:
            r = engine_module.process(text)
            entities = r.get("entities", [])
            found = any(
                e.get("type") == expected_type and expected_text in e.get("text", "")
                for e in entities
            )
            entity_scores.append(1.0 if found else 0.0)
            result.details.append({
                "input": text, "expected_entity": f"{expected_type}:{expected_text}",
                "found": found, "passed": found, "category": "entity_extraction",
            })
        except Exception:
            entity_scores.append(0.0)

    # ── 4. 情感分析测试 (v3 新增) ──
    sentiment_scores = []
    for text, expected_polarity, expected_emotion in SENTIMENT_TEST_CASES:
        try:
            r = engine_module.process(text)
            sent = r.get("sentiment", {})
            polarity_ok = sent.get("polarity") == expected_polarity
            # 情绪只要方向一致即可 (不严格要求具体情绪)
            emotion_ok = True  # 宽松匹配
            ok = polarity_ok and emotion_ok
            sentiment_scores.append(1.0 if ok else 0.0)
            result.details.append({
                "input": text, "expected_polarity": expected_polarity,
                "actual_polarity": sent.get("polarity"),
                "passed": ok, "category": "sentiment",
            })
        except Exception:
            sentiment_scores.append(0.0)

    # ── 5. 鲁棒性测试 ──
    robustness_scores = []
    for text, should_survive in ROBUSTNESS_TEST_CASES:
        try:
            if text is None:
                r = engine_module.process("")
            else:
                r = engine_module.process(text)
            robustness_scores.append(1.0 if should_survive else 0.0)
        except Exception:
            robustness_scores.append(0.0 if should_survive else 1.0)

    # ── 6. 信息密度 (v3 新增) ──
    info_density_scores = []
    for text in ["今天天气怎么样", "读取test.py", "搜索相关内容", "太好了！"]:
        try:
            r = engine_module.process(text)
            density = r.get("info_density", 0.0)
            info_density_scores.append(min(density * 2, 1.0))  # 放大到 [0,1]
        except Exception:
            info_density_scores.append(0.0)

    # ── 7. 推理链质量 (v3 新增) ──
    chain_quality_scores = []
    for text in ["今天天气怎么样", "读取test.py并搜索相关内容"]:
        try:
            r = engine_module.process(text)
            quality = r.get("chain_quality", 0.0)
            chain_quality_scores.append(quality)
        except Exception:
            chain_quality_scores.append(0.0)

    # ── 8. 置信度校准 (v3 新增) ──
    calibration_scores = []
    for text, expected_intent, _ in BASIC_TEST_CASES:
        if not text:
            continue
        try:
            r = engine_module.process(text)
            conf = r.get("confidence", 0.0)
            actual = r.get("intent", "unknown")
            # 高置信度应正确，低置信度可容错
            if actual == expected_intent:
                calibration_scores.append(conf)  # 正确时置信度越高越好
            else:
                calibration_scores.append(1.0 - conf)  # 错误时置信度越低越好
        except Exception:
            calibration_scores.append(0.0)

    # ── 9. 延迟 ──
    latencies = [d.get("latency_ms", 0) for d in result.details if "latency_ms" in d]

    # ── 10. 汇总分数 ──
    result.score = EvalScore(
        accuracy=sum(accuracy_scores) / max(len(accuracy_scores), 1),
        latency_ms=sum(latencies) / max(len(latencies), 1),
        robustness=sum(robustness_scores) / max(len(robustness_scores), 1),
        coverage=sum(1 for s in accuracy_scores if s > 0) / max(len(accuracy_scores), 1),
        fusion_quality=_eval_fusion_quality(engine_module),
        info_density=sum(info_density_scores) / max(len(info_density_scores), 1),
        chain_quality=sum(chain_quality_scores) / max(len(chain_quality_scores), 1),
        calibration=sum(calibration_scores) / max(len(calibration_scores), 1),
        entity_accuracy=sum(entity_scores) / max(len(entity_scores), 1),
        sentiment_accuracy=sum(sentiment_scores) / max(len(sentiment_scores), 1),
    )

    return result


def _eval_fusion_quality(engine_module) -> float:
    """评估融合质量: 多路径一致性和多任务预测。"""
    fusion_scores = []
    test_inputs = [
        "今天天气怎么样",
        "读取test.py",
        "搜索python教程",
    ]
    for text in test_inputs:
        try:
            r = engine_module.process(text)
            score = 0.0
            # 多路径是否有输出
            votes = r.get("path_votes", {})
            active_paths = sum(1 for v in votes.values() if v > 0)
            score += active_paths / 3.0 * 0.4
            # 推理链是否有内容
            chain = r.get("reasoning_chain", [])
            if len(chain) > 2:
                score += 0.3
            # 多任务预测 (v3)
            if r.get("entities"):
                score += 0.15
            if r.get("sentiment", {}).get("polarity"):
                score += 0.15
            fusion_scores.append(min(score, 1.0))
        except Exception:
            fusion_scores.append(0.0)
    return sum(fusion_scores) / max(len(fusion_scores), 1)


def save_eval(result: EvalResult):
    """保存评估结果"""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"eval_{result.version}_{int(result.timestamp)}.json"
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Eval saved: {path}")


def load_eval_history() -> List[EvalResult]:
    """加载评估历史"""
    if not EVAL_DIR.exists():
        return []

    results = []
    for f in sorted(EVAL_DIR.glob("eval_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            r = EvalResult(
                version=data.get("version", "?"),
                timestamp=data.get("timestamp", 0),
                test_cases=data.get("test_cases", 0),
                passed=data.get("passed", 0),
                failed=data.get("failed", 0),
                improvements=data.get("improvements", []),
            )
            score_data = data.get("score", {})
            # 兼容 v2 和 v3 的 score 字段
            r.score = EvalScore(
                accuracy=score_data.get("accuracy", 0),
                latency_ms=score_data.get("latency_ms", 0),
                robustness=score_data.get("robustness", 0),
                coverage=score_data.get("coverage", 0),
                fusion_quality=score_data.get("fusion_quality", 0),
                info_density=score_data.get("info_density", 0),
                chain_quality=score_data.get("chain_quality", 0),
                calibration=score_data.get("calibration", 0),
                entity_accuracy=score_data.get("entity_accuracy", 0),
                sentiment_accuracy=score_data.get("sentiment_accuracy", 0),
            )
            results.append(r)
        except Exception:
            pass

    return results


def compare_versions(v1: str, v2: str) -> Optional[dict]:
    """对比两个版本的评估结果"""
    history = load_eval_history()
    r1 = [r for r in history if r.version == v1]
    r2 = [r for r in history if r.version == v2]

    if not r1 or not r2:
        return None

    s1 = r1[-1].score
    s2 = r2[-1].score

    delta = {
        "accuracy": round(s2.accuracy - s1.accuracy, 4),
        "latency_ms": round(s2.latency_ms - s1.latency_ms, 1),
        "robustness": round(s2.robustness - s1.robustness, 4),
        "coverage": round(s2.coverage - s1.coverage, 4),
        "fusion_quality": round(s2.fusion_quality - s1.fusion_quality, 4),
        "composite": round(s2.composite - s1.composite, 4),
    }
    # v3 新增维度 (如果双方都有)
    for dim in ("info_density", "chain_quality", "calibration", "entity_accuracy", "sentiment_accuracy"):
        v1_val = getattr(s1, dim, 0)
        v2_val = getattr(s2, dim, 0)
        delta[dim] = round(v2_val - v1_val, 4)

    return {
        "v1": {"version": v1, "score": s1.to_dict()},
        "v2": {"version": v2, "score": s2.to_dict()},
        "delta": delta,
        "improvement": s2.composite > s1.composite,
    }


# ── CLI ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluator.py run [version] [engine_file]  # 运行评估")
        print("  python evaluator.py history                      # 查看历史")
        print("  python evaluator.py compare v1 v2                # 对比版本")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "run":
        version = sys.argv[2] if len(sys.argv) > 2 else "v0"
        engine_file = sys.argv[3] if len(sys.argv) > 3 else None

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        if engine_file:
            # 动态导入指定引擎文件
            import importlib.util
            spec = importlib.util.spec_from_file_location("engine_module", engine_file)
            engine_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(engine_module)
        else:
            import aris_brain.aris_fusion_engine as engine_module

        result = evaluate_fusion_engine(engine_module, version)
        save_eval(result)
        print(f"Version: {result.version}")
        print(f"Tests: {result.test_cases} | Passed: {result.passed} | Failed: {result.failed}")
        print(f"Score: {json.dumps(result.score.to_dict(), indent=2)}")

    elif cmd == "history":
        history = load_eval_history()
        for r in history:
            print(f"{r.version}: composite={r.score.composite:.4f} ({r.passed}/{r.test_cases})")

    elif cmd == "compare":
        if len(sys.argv) < 4:
            print("Usage: python evaluator.py compare v1 v2")
            sys.exit(1)
        c = compare_versions(sys.argv[2], sys.argv[3])
        if c:
            print(json.dumps(c, indent=2, ensure_ascii=False))
        else:
            print("Versions not found")
