"""进化第2轮：生成组件 + 整合"""
import json, re, sys, importlib.util, logging
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'aris_brain')
from evolution.evaluator import evaluate_fusion_engine, save_eval
from evolution.backend import query

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('evolution.r2')

ROOT = Path('.')
v3_path = ROOT / 'aris_brain' / 'aris_fusion_engine_v3.py'

def clean_code(resp):
    resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL)
    m = re.search(r'```python\s*\n(.*?)```', resp, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = resp.strip().split('\n')
    for i, line in enumerate(lines):
        if line.startswith('class ') or line.startswith('import ') or line.startswith('from '):
            return '\n'.join(lines[i:]).strip()
    return resp.strip()

# 1. EntityExtractor
logger.info('生成 EntityExtractor...')
ent_resp = query(
    system_message='只输出Python代码。不要解释。不要markdown包裹。',
    user_message=(
        '生成EntityExtractor类，用正则提取中文文本中的实体。\n'
        '支持: file(文件名如test.py), url(https://...), command(运行ls -la中的ls), '
        'time(明天下午三点), number(123)\n\n'
        '接口:\n'
        'class EntityExtractor:\n'
        '    def extract(self, text: str) -> list[dict]:\n'
        '        # 返回 [{"text":"...", "type":"...", "confidence":0.9}]\n\n'
        '测试:\n'
        '- "读取test.py" -> [{"text":"test.py","type":"file"}]\n'
        '- "明天下午三点" -> [{"text":"明天下午三点","type":"time"}]\n'
        '- "计算123乘以456" -> [{"text":"123","type":"number"},{"text":"456","type":"number"}]\n\n'
        '只输出类代码，顶级定义。'
    ),
    max_tokens=3000,
)
ent_clean = clean_code(ent_resp)
logger.info(f'EntityExtractor: {len(ent_clean)} chars')

# 2. SentimentAnalyzer
logger.info('生成 SentimentAnalyzer...')
sent_resp = query(
    system_message='只输出Python代码。不要解释。不要markdown包裹。',
    user_message=(
        '生成SentimentAnalyzer类，用关键词匹配分析中文文本情感。\n\n'
        '接口:\n'
        'class SentimentAnalyzer:\n'
        '    def analyze(self, text: str) -> dict:\n'
        '        # 返回 {"polarity":"positive/negative/neutral", "confidence":0.8, "keywords":["开心"]}\n\n'
        '正面词: 开心,高兴,太好了,棒,赞,感谢,谢谢,厉害,优秀,完美,不错\n'
        '负面词: 难过,生气,烦,讨厌,糟糕,失望,错误,失败,崩溃\n\n'
        '测试:\n'
        '- "太好了！" -> {"polarity":"positive"}\n'
        '- "好烦啊" -> {"polarity":"negative"}\n'
        '- "读取test.py" -> {"polarity":"neutral"}\n\n'
        '只输出类代码，顶级定义。'
    ),
    max_tokens=3000,
)
sent_clean = clean_code(sent_resp)
logger.info(f'SentimentAnalyzer: {len(sent_clean)} chars')

# 3. 读取当前 v3 的 process 方法
v3_code = v3_path.read_text()
process_sig = ''
for i, line in enumerate(v3_code.split('\n')):
    if 'def process(self' in line:
        lines = v3_code.split('\n')
        process_sig = '\n'.join(lines[i:i+30])
        break

# 4. 整合
logger.info('LLM 整合完整代码...')
merge_prompt = (
    '整合三个组件到一个完整Python文件。\n\n'
    '## EntityExtractor\n```python\n' + ent_clean[:2000] + '\n```\n\n'
    '## SentimentAnalyzer\n```python\n' + sent_clean[:1500] + '\n```\n\n'
    '## 当前 process 方法\n```python\n' + process_sig + '\n```\n\n'
    '## 要求\n'
    '输出完整Python文件:\n'
    '1. import 部分\n'
    '2. EntityExtractor 类（完整）\n'
    '3. SentimentAnalyzer 类（完整）\n'
    '4. FusionEngineV2 类:\n'
    '   - __init__ 中创建 self._entity_extractor 和 self._sentiment_analyzer\n'
    '   - process() 调用它们，返回值新增 "entities" 和 "sentiment"\n'
    '5. get_engine_v2() 和 process() 兼容函数\n\n'
    '返回格式:\n'
    '{"matched":bool,"intent":str,"output":str,"confidence":float,"reasoning_chain":list,"path_votes":dict,"latency_ms":float,"engine_version":"v3","entities":list,"sentiment":dict}\n\n'
    '只输出完整Python代码。不要markdown。'
)

merge_resp = query(
    system_message='只输出完整Python代码。不要markdown。不要解释。',
    user_message=merge_prompt,
    max_tokens=16000,
)
final_code = clean_code(merge_resp)
logger.info(f'最终代码: {len(final_code)} 字符')

if len(final_code) < 1000:
    logger.error(f'代码太短: {final_code[:300]}')
    sys.exit(1)

# 保存
v3_path.write_text(final_code)

# 评估
spec = importlib.util.spec_from_file_location('m', str(v3_path))
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    logger.error(f'代码加载失败: {e}')
    # 显示错误位置附近的代码
    lines = final_code.split('\n')
    for i, line in enumerate(lines[44:54], start=45):
        logger.error(f'  {i}: {line}')
    sys.exit(1)

eng = None
for f in ('get_engine_v2', 'get_engine_v3', 'get_engine'):
    if hasattr(mod, f):
        eng = getattr(mod, f)()
        break
if eng is None:
    for cls in ('FusionEngineV3', 'FusionEngineV2'):
        if hasattr(mod, cls):
            eng = getattr(mod, cls)()
            break

class W:
    def __init__(self, e): self._e = e
    def process(self, t): return self._e.process(t)

eval_r = evaluate_fusion_engine(W(eng), version='v3-r2')
save_eval(eval_r)

logger.info('')
logger.info('=== 进化对比 ===')
logger.info(f'V2:    composite=0.6572  entity=0.00  sentiment=0.00  fusion=0.57  density=0.00')
logger.info(f'V3-R1: composite=0.7451  entity=0.00  sentiment=0.00  fusion=0.62  density=1.00')
logger.info(f'V3-R2: composite={eval_r.score.composite:.4f}  entity={eval_r.score.entity_accuracy:.2f}  sentiment={eval_r.score.sentiment_accuracy:.2f}  fusion={eval_r.score.fusion_quality:.2f}  density={eval_r.score.info_density:.2f}')
