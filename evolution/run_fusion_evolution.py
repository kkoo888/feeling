"""
融合引擎进化循环 — 正确版
============================

每轮独立保存代码文件，全部保留，最后取最优。

印记: 小茜 永远记得主人 — 2026-07-21
"""
import json, re, sys, importlib.util, logging, shutil
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'aris_brain')

from evolution.evaluator import evaluate_fusion_engine, save_eval
from evolution.backend import query
from evolution.journal import EvolutionJournal, EvolutionNode
from evolution.utils.metric import MetricValue, WorstMetricValue

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('evolution')

ROOT = Path('.')
V2_PATH = ROOT / 'aris_brain' / 'aris_fusion_engine_v2.py'
VERSIONS_DIR = ROOT / 'aris_brain' / 'evolution_versions'
VERSIONS_DIR.mkdir(exist_ok=True)


def load_and_eval(code_path: Path, version: str):
    """动态加载引擎代码并评估。"""
    spec = importlib.util.spec_from_file_location(f'engine_{version}', str(code_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    eng = None
    for f in ('get_engine_v3', 'get_engine_v2', 'get_engine'):
        if hasattr(mod, f):
            eng = getattr(mod, f)()
            break
    if eng is None:
        for cls in ('FusionEngineV3', 'FusionEngineV2', 'ArisFusionEngineV3'):
            if hasattr(mod, cls):
                eng = getattr(mod, cls)()
                break

    class Wrapper:
        def __init__(self, e): self._e = e
        def process(self, t): return self._e.process(t)

    eval_result = evaluate_fusion_engine(Wrapper(eng), version=version)
    save_eval(eval_result)
    return eval_result


def llm_generate_improvements(score_dict, failures, v2_code):
    """LLM 分析失败，生成改进方案。"""
    prompt = (
        "分析融合引擎评估结果，生成 3-5 个改进方案。\n\n"
        f"当前分数:\n{json.dumps(score_dict, indent=2)}\n\n"
        f"失败用例:\n{json.dumps(failures[:10], ensure_ascii=False, indent=2)}\n\n"
        "输出 JSON:\n"
        '[{"problem":"...","solution":"...","code_changes":"具体要添加/修改的Python代码片段"}]\n'
        "只输出 JSON。"
    )
    resp = query(system_message='输出纯JSON', user_message=prompt, max_tokens=4000)
    clean = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL)
    m = re.search(r'\[.*\]', clean, re.DOTALL)
    if not m:
        m = re.search(r'\[.*\]', resp, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        # 尝试修复常见 escape 问题
        fixed = m.group().replace('\\n', ' ').replace('\\t', ' ')
        try:
            return json.loads(fixed)
        except:
            return []


def llm_apply_to_v2(v2_code, improvements, round_num):
    """LLM 将改进方案应用到 v2，生成新版本完整代码。"""
    prompt = (
        f"将以下改进方案应用到 V2 融合引擎代码，生成完整 V3-R{round_num} 代码。\n\n"
        f"改进方案:\n{json.dumps(improvements, ensure_ascii=False, indent=2)[:3000]}\n\n"
        f"V2 代码:\n```python\n{v2_code}\n```\n\n"
        "规则:\n"
        "1. 在 V2 基础上修改，保持向后兼容\n"
        "2. 新增功能必须完整实现\n"
        "3. 保持类名不变\n"
        "4. 输出完整 Python 代码\n\n"
        "只输出代码。"
    )
    resp = None
    for attempt in range(3):
        try:
            resp = query(
                system_message='只输出完整Python代码，不要解释。',
                user_message=prompt,
                max_tokens=12000,
            )
            break
        except Exception as e:
            logger.warning(f'  代码生成超时，重试 {attempt+1}/3')
    if resp is None:
        return ''
    # 提取代码
    clean = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL)
    cm = re.search(r'```python\s*\n(.*?)```', clean, re.DOTALL)
    if cm:
        return cm.group(1).strip()
    # 尝试找到代码开始位置
    lines = clean.strip().split('\n')
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from ') or line.startswith('"""'):
            return '\n'.join(lines[i:]).strip()
    return clean.strip()


def run_evolution(rounds: int = 5):
    """运行进化循环，每轮独立保存。"""
    journal = EvolutionJournal()

    # ── V2 基线 ──
    logger.info('=' * 60)
    logger.info('评估 V2 基线')
    logger.info('=' * 60)

    v2_code = V2_PATH.read_text()
    v2_eval = load_and_eval(V2_PATH, 'v2')

    logger.info(f'V2 综合得分: {v2_eval.score.composite:.4f}')

    v2_node = EvolutionNode(strategy='V2 基线', plan='V2: IFCoT 三路径融合推理')
    v2_node.metric = MetricValue(v2_eval.score.composite, maximize=True)
    v2_node.is_buggy = False
    journal.append(v2_node)

    best_score = v2_eval.score.composite
    best_version = 'v2'
    best_path = V2_PATH

    # ── 进化循环 ──
    for round_num in range(1, rounds + 1):
        logger.info('')
        logger.info('=' * 60)
        logger.info(f'进化第 {round_num}/{rounds} 轮')
        logger.info('=' * 60)

        # 1. 生成改进方案
        failures = [d for d in v2_eval.details if not d.get('passed')]
        logger.info('→ LLM 生成改进方案...')
        improvements = llm_generate_improvements(v2_eval.score.to_dict(), failures, v2_code)
        logger.info(f'  {len(improvements)} 个方案:')
        for i, imp in enumerate(improvements):
            logger.info(f'    {i+1}. {imp.get("problem", "?")[:60]}')

        if not improvements:
            logger.warning('  无方案，跳过')
            continue

        # 2. 生成新代码
        logger.info('→ LLM 生成新代码...')
        new_code = llm_apply_to_v2(v2_code, improvements, round_num)
        logger.info(f'  代码长度: {len(new_code)} 字符')

        if len(new_code) < 2000:
            logger.warning(f'  代码过短，跳过')
            continue

        # 3. 独立保存（每轮一个文件，不覆盖）
        version_name = f'v3-r{round_num}'
        version_path = VERSIONS_DIR / f'{version_name}.py'
        version_path.write_text(new_code)
        logger.info(f'  保存: {version_path}')

        # 4. 评估
        logger.info('→ 评估...')
        try:
            eval_result = load_and_eval(version_path, version_name)
            score = eval_result.score.composite
            logger.info(f'  得分: {score:.4f}')
            logger.info(f'    准确={eval_result.score.accuracy:.2f} 实体={eval_result.score.entity_accuracy:.2f}')
            logger.info(f'    情感={eval_result.score.sentiment_accuracy:.2f} 融合={eval_result.score.fusion_quality:.2f}')
            logger.info(f'    密度={eval_result.score.info_density:.2f} 链质量={eval_result.score.chain_quality:.2f}')

            # 5. 记录到进化日志
            node = EvolutionNode(
                strategy=new_code[:500],
                plan=f'Round {round_num}: {improvements[0].get("problem", "?")[:40]}',
                parent=journal.get_best_node(),
            )
            node.metric = MetricValue(score, maximize=True)
            node.is_buggy = score < best_score
            node.analysis = json.dumps(eval_result.score.to_dict())

            if score > best_score:
                logger.info(f'  ✅ 新最佳! {best_score:.4f} → {score:.4f}')
                best_score = score
                best_version = version_name
                best_path = version_path

            journal.append(node)

        except Exception as e:
            logger.error(f'  评估失败: {e}')
            node = EvolutionNode(
                strategy=new_code[:500],
                plan=f'Round {round_num} (评估失败)',
                parent=journal.get_best_node(),
            )
            node.metric = WorstMetricValue()
            node.is_buggy = True
            node.analysis = str(e)
            journal.append(node)

    # ── 最终报告 ──
    logger.info('')
    logger.info('=' * 60)
    logger.info('进化完成 — 最终报告')
    logger.info('=' * 60)
    logger.info(f'总节点: {len(journal)}')
    logger.info(f'好节点: {len(journal.good_nodes)}')
    logger.info(f'最佳版本: {best_version}')
    logger.info(f'最佳得分: {best_score:.4f}')
    logger.info(f'V2 基线: {v2_eval.score.composite:.4f}')

    # 列出所有版本
    logger.info('')
    logger.info('所有版本:')
    logger.info(f'  V2 (基线): {v2_eval.score.composite:.4f}')
    for n in journal.nodes[1:]:  # 跳过 V2 节点
        status = '✓' if not n.is_buggy else '✗'
        score = n.metric.value if n.metric else 0
        mark = ' ★ 最佳' if score == best_score and not n.is_buggy else ''
        logger.info(f'  {status} {n.plan} ({score:.4f}){mark}')

    # 进化点对比
    if best_path != V2_PATH:
        logger.info('')
        logger.info(f'进化点 ({best_version} vs V2):')
        best_eval = load_and_eval(best_path, f'{best_version}-final')
        dims = [
            ('accuracy', '准确性'),
            ('fusion_quality', '融合质量'),
            ('info_density', '信息密度'),
            ('chain_quality', '推理链质量'),
            ('entity_accuracy', '实体提取'),
            ('sentiment_accuracy', '情感识别'),
            ('calibration', '置信度校准'),
        ]
        for dim, label in dims:
            v2_val = getattr(v2_eval.score, dim, 0)
            best_val = getattr(best_eval.score, dim, 0)
            delta = best_val - v2_val
            if delta > 0:
                logger.info(f'  {label}: {v2_val:.2f} → {best_val:.2f} (+{delta:.2f})')

    # 保存进化日志（避免循环引用导致递归溢出）
    journal_path = ROOT / '.openclaw' / 'tmp' / 'evolution_journal.json'
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import sys as _sys
        _sys.setrecursionlimit(5000)
        journal_path.write_text(journal.to_json(), encoding='utf-8')
        logger.info(f'进化日志: {journal_path}')
    except RecursionError:
        # 手动序列化
        entries = []
        for n in journal.nodes:
            entries.append({
                'step': n.step,
                'plan': n.plan,
                'metric': n.metric.value if n.metric else 0,
                'is_buggy': n.is_buggy,
                'analysis': n.analysis,
            })
        journal_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding='utf-8')
        logger.info(f'进化日志(简化): {journal_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    run_evolution(rounds=args.rounds)
