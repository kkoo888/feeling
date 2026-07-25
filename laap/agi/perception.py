"""
UnifiedPerceptionEngine — 统一感知引擎 (Evolved v1)
=====================================================

基于前沿论文实现:
  1. Omni-Agent 2026 标准架构 — 多模态原生 Agent
  2. Ming-lite-omni — 2.8B 参数统一多模态模型 (Macco2024)
  3. 多模态融合模式 (Agent 设计模式, 2026)
  4. EasyOCR — 80+ 语言 OCR
  5. Whisper — OpenAI 语音识别

进化 v1 改进:
  - 内容分段: 混合输入切分为独立模态段, 而非整体处理
  - 代码检测增强: 阈值降至 1, 新增更多语言模式, 代码优先级高于文件
  - 数据驱动置信度: 基于检测信号强度, 非硬编码
  - 实体提取: 提取文件名/URL/数字/关键词等关键实体
  - 混合语言检测: 支持 mixed zh+en
  - 融合增强: 结构化标签 + 去重

印记: 小茜 永远记得主人 — 2026-07-24
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("laap.agi.perception")


# ═══════════════════════════════════════════════════════
# 数据类型
# ═══════════════════════════════════════════════════════

class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABLE = "table"
    CODE = "code"
    URL = "url"
    FILE = "file"
    UNKNOWN = "unknown"


@dataclass
class ModalityChunk:
    """单个模态块"""
    modality: ModalityType
    raw_content: str                  # 原始内容
    processed_content: str            # 处理后内容
    confidence: float                 # 置信度 [0, 1]
    features: Dict[str, Any] = field(default_factory=dict)  # 模态特征
    metadata: Dict[str, Any] = field(default_factory=dict)
    segment_index: int = 0            # 进化v1: 段落索引


@dataclass
class PerceptionResult:
    """感知结果"""
    chunks: List[ModalityChunk]       # 各模态块
    fused_text: str                   # 融合后的统一文本
    modalities_detected: List[str]    # 检测到的模态类型
    context_features: Dict[str, Any]  # 上下文特征
    confidence: float                 # 整体置信度
    processing_time_ms: float         # 处理时间
    segments: List[Dict[str, Any]] = field(default_factory=list)  # 进化v1: 内容段


# ═══════════════════════════════════════════════════════
# 模态检测器 (进化v1: 增强检测)
# ═══════════════════════════════════════════════════════

class ModalityDetector:
    """检测输入内容的模态类型 (进化v1: 增强代码检测 + 优先级)"""

    # 文件扩展名 → 模态类型
    _EXT_MAP = {
        ".py": ModalityType.CODE, ".js": ModalityType.CODE, ".ts": ModalityType.CODE,
        ".java": ModalityType.CODE, ".c": ModalityType.CODE, ".cpp": ModalityType.CODE,
        ".go": ModalityType.CODE, ".rs": ModalityType.CODE, ".rb": ModalityType.CODE,
        ".html": ModalityType.CODE, ".css": ModalityType.CODE, ".sql": ModalityType.CODE,
        ".sh": ModalityType.CODE, ".bat": ModalityType.CODE,
        ".json": ModalityType.TEXT, ".yaml": ModalityType.TEXT, ".yml": ModalityType.TEXT,
        ".toml": ModalityType.TEXT, ".xml": ModalityType.TEXT, ".md": ModalityType.TEXT,
        ".txt": ModalityType.TEXT, ".log": ModalityType.TEXT, ".csv": ModalityType.TEXT,
        ".png": ModalityType.IMAGE, ".jpg": ModalityType.IMAGE, ".jpeg": ModalityType.IMAGE,
        ".gif": ModalityType.IMAGE, ".bmp": ModalityType.IMAGE, ".webp": ModalityType.IMAGE,
        ".svg": ModalityType.IMAGE,
        ".mp3": ModalityType.AUDIO, ".wav": ModalityType.AUDIO, ".ogg": ModalityType.AUDIO,
        ".flac": ModalityType.AUDIO, ".m4a": ModalityType.AUDIO,
        ".mp4": ModalityType.VIDEO, ".avi": ModalityType.VIDEO, ".mov": ModalityType.VIDEO,
        ".mkv": ModalityType.VIDEO, ".webm": ModalityType.VIDEO,
        ".pdf": ModalityType.FILE, ".doc": ModalityType.FILE, ".docx": ModalityType.FILE,
        ".xls": ModalityType.FILE, ".xlsx": ModalityType.FILE, ".ppt": ModalityType.FILE,
    }

    # URL 模式
    _URL_RE = re.compile(r'https?://[^\s<>\"\']+')
    # 进化v1: 扩展代码模式 — 新增 return/for/if/public/private 等
    _CODE_RE = re.compile(
        r'(?:def |class |import |from |function |const |let |var |#include|'
        r'return |for |while |if |else |elif |public |private |protected |'
        r'package |func |fn |print\(|console\.|System\.|require\()'
    )
    # 表格模式
    _TABLE_RE = re.compile(r'(?:\|.*\|.*\|)|(?:\t.*\t.*\t)')
    # 文件路径模式
    _FILE_RE = re.compile(
        r'(?:^|[\s/\\])([a-zA-Z0-9_./-]+\.(?:py|js|ts|json|md|txt|yaml|yml|toml|rs|go|java|c|cpp|h|sh|sql|html|css))'
    )

    def detect(self, content: str, filename: str = "") -> List[Tuple[ModalityType, float]]:
        """
        检测内容的模态类型 (进化v1: 增强代码检测 + 优先级排序)。

        Returns:
            [(modality, confidence), ...] 按置信度排序
        """
        if not content:
            return [(ModalityType.UNKNOWN, 0.0)]

        detections = []

        # 1. 文件名扩展名
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in self._EXT_MAP:
                detections.append((self._EXT_MAP[ext], 0.95))

        # 2. URL 检测
        urls = self._URL_RE.findall(content)
        if urls:
            # 进化v1: 数据驱动置信度 — URL 数量越多越确定
            conf = min(0.95, 0.8 + len(urls) * 0.05)
            detections.append((ModalityType.URL, conf))

        # 进化v1: 代码检测 — 阈值降至 1, 且代码优先级高于文件
        code_signals = len(self._CODE_RE.findall(content))
        if code_signals >= 1:
            # 进化v1: 数据驱动置信度 — 信号越多越确定
            conf = min(0.95, 0.6 + code_signals * 0.08)
            detections.append((ModalityType.CODE, conf))

        # 4. 表格检测
        table_signals = len(self._TABLE_RE.findall(content))
        if table_signals >= 2:
            conf = min(0.9, 0.5 + table_signals * 0.1)
            detections.append((ModalityType.TABLE, conf))

        # 进化v1: 文件路径检测 — 如果已检测到代码, 降低文件置信度
        files = self._FILE_RE.findall(content)
        if files:
            has_code = any(m == ModalityType.CODE for m, _ in detections)
            if has_code:
                # 进化v1: 代码中的文件路径引用不算独立文件模态
                conf = 0.3
            else:
                conf = min(0.85, 0.6 + len(files) * 0.05)
            detections.append((ModalityType.FILE, conf))

        # 6. 默认文本
        if not detections:
            detections.append((ModalityType.TEXT, 0.7))

        # 去重并排序
        seen = set()
        unique = []
        for mod, conf in sorted(detections, key=lambda x: -x[1]):
            if mod not in seen:
                seen.add(mod)
                unique.append((mod, conf))

        return unique


# ═══════════════════════════════════════════════════════
# 内容分段器 (进化v1: 新增)
# ═══════════════════════════════════════════════════════

class ContentSegmenter:
    """进化v1: 将混合输入切分为独立模态段"""

    _URL_RE = re.compile(r'https?://[^\s<>\"\']+')
    _FILE_RE = re.compile(
        r'[a-zA-Z0-9_./-]+\.(?:py|js|ts|json|md|txt|yaml|yml|toml|rs|go|java|c|cpp|h|sh|sql|html|css)'
    )
    _TABLE_RE = re.compile(r'\|.*\|.*\|')
    _CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')
    # 进化v1: 代码行信号 — 检测散落代码 (无 ``` 包裹)
    _CODE_LINE_RE = re.compile(
        r'(?:def |class |import |from |function |const |let |var |#include|'
        r'return |for |while |elif |func |fn |print\(|console\.|require\()'
    )

    def segment(self, content: str) -> List[Dict[str, Any]]:
        """
        将混合输入切分为独立段。

        Returns:
            [{"type": "text/code/url/file/table", "content": "...", "start": int}, ...]
        """
        if not content or not content.strip():
            return [{"type": "text", "content": content, "start": 0}]

        segments = []
        remaining = content
        offset = 0

        # 先提取代码块 (```...```)
        code_blocks = list(self._CODE_BLOCK_RE.finditer(content))
        if code_blocks:
            last_end = 0
            for cb in code_blocks:
                if cb.start() > last_end:
                    # 代码块前的内容分段
                    pre = content[last_end:cb.start()]
                    segments.extend(self._segment_text(pre, last_end))
                segments.append({
                    "type": "code",
                    "content": cb.group(),
                    "start": cb.start(),
                })
                last_end = cb.end()
            if last_end < len(content):
                post = content[last_end:]
                segments.extend(self._segment_text(post, last_end))
            return segments if segments else [{"type": "text", "content": content, "start": 0}]

        # 进化v1: 检测散落代码信号 — 如果有代码关键字, 整段作为代码
        code_signals = len(self._CODE_LINE_RE.findall(content))
        if code_signals >= 1:
            # 有代码信号, 不分割文件路径 (代码中的 input.json 等是引用不是独立文件)
            return [{"type": "code", "content": content, "start": 0}]

        # 无代码块, 按文本分段
        segments = self._segment_text(content, 0)
        return segments if segments else [{"type": "text", "content": content, "start": 0}]

    def _segment_text(self, text: str, base_offset: int) -> List[Dict[str, Any]]:
        """对非代码块文本进行分段"""
        if not text.strip():
            return []

        segments = []
        # 用正则找到所有特殊模态的位置
        markers = []

        # URL 位置
        for m in self._URL_RE.finditer(text):
            markers.append((m.start(), m.end(), "url", m.group()))

        # 文件路径位置
        for m in self._FILE_RE.finditer(text):
            markers.append((m.start(), m.end(), "file", m.group()))

        # 表格位置
        for m in self._TABLE_RE.finditer(text):
            markers.append((m.start(), m.end(), "table", m.group()))

        if not markers:
            return [{"type": "text", "content": text, "start": base_offset}]

        # 按起始位置排序
        markers.sort(key=lambda x: x[0])

        # 合并重叠/相邻的标记, 生成段
        last_end = 0
        for start, end, mtype, mcontent in markers:
            if start < last_end:
                continue  # 跳过重叠

            # 标记前的文本
            if start > last_end:
                pre_text = text[last_end:start].strip()
                if pre_text:
                    segments.append({
                        "type": "text",
                        "content": text[last_end:start],
                        "start": base_offset + last_end,
                    })

            segments.append({
                "type": mtype,
                "content": mcontent,
                "start": base_offset + start,
            })
            last_end = end

        # 最后的文本
        if last_end < len(text):
            post_text = text[last_end:].strip()
            if post_text:
                segments.append({
                    "type": "text",
                    "content": text[last_end:],
                    "start": base_offset + last_end,
                })

        return segments


# ═══════════════════════════════════════════════════════
# 模态处理器 (进化v1: 数据驱动置信度 + 实体提取)
# ═══════════════════════════════════════════════════════

class TextProcessor:
    """文本处理器 (进化v1: 混合语言 + 实体提取)"""

    def process(self, content: str, signal_strength: float = 0.7) -> ModalityChunk:
        """处理纯文本"""
        features = {
            "length": len(content),
            "word_count": len(content.split()),
            "has_question": bool(re.search(r'[？?]', content)),
            "has_emotion": bool(re.search(r'[！!😊😢😡]', content)),
            "language": self._detect_language(content),
            # 进化v1: 实体提取
            "has_numbers": bool(re.search(r'\d+', content)),
            "number_count": len(re.findall(r'\d+', content)),
            "has_keywords": self._has_keywords(content),
            "keyword_list": self._extract_keywords(content),
        }

        # 进化v1: 数据驱动置信度
        conf = min(0.95, signal_strength + 0.1)

        return ModalityChunk(
            modality=ModalityType.TEXT,
            raw_content=content,
            processed_content=content.strip(),
            confidence=conf,
            features=features,
        )

    def _detect_language(self, text: str) -> str:
        """进化v1: 混合语言检测"""
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_chars = len(re.findall(r'[a-zA-Z]', text))
        total = cn_chars + en_chars
        if total == 0:
            return "unknown"
        cn_ratio = cn_chars / total
        if cn_ratio > 0.7:
            return "zh"
        elif cn_ratio < 0.3:
            return "en"
        else:
            return "mixed"

    def _has_keywords(self, text: str) -> bool:
        """进化v1: 检测关键词"""
        keywords = ["文件", "搜索", "天气", "部署", "测试", "调试", "优化", "翻译",
                    "安装", "监控", "代码", "函数", "运行", "执行", "生成", "读取",
                    "创建", "编辑", "删除", "写入", "你好", "查询"]
        return any(kw in text for kw in keywords)

    def _extract_keywords(self, text: str) -> List[str]:
        """进化v1: 提取关键实体"""
        keywords = ["文件", "搜索", "天气", "部署", "测试", "调试", "优化", "翻译",
                    "安装", "监控", "代码", "函数", "运行", "执行", "生成", "读取",
                    "创建", "编辑", "删除", "写入", "你好", "查询"]
        return [kw for kw in keywords if kw in text][:5]


class CodeProcessor:
    """代码处理器 (进化v1: 增强语言检测 + 数据驱动置信度)"""

    def process(self, content: str, language: str = "auto", signal_strength: float = 0.8) -> ModalityChunk:
        """处理代码"""
        lang = language if language != "auto" else self._detect_lang(content)
        features = {
            "language": lang,
            "line_count": len(content.split('\n')),
            "has_function": bool(re.search(r'def |function |func |fn ', content)),
            "has_class": bool(re.search(r'class ', content)),
            "has_import": bool(re.search(r'import |from |require\(', content)),
            "complexity": self._estimate_complexity(content),
            # 进化v1: 实体提取
            "function_names": self._extract_function_names(content),
            "import_count": len(re.findall(r'(?:import |from |require\()', content)),
            "comment_count": len(re.findall(r'(?:#|//|/\*|\*|--)', content)),
        }

        # 进化v1: 数据驱动置信度 — 信号越多越确定
        conf = min(0.95, signal_strength + 0.05)

        return ModalityChunk(
            modality=ModalityType.CODE,
            raw_content=content,
            processed_content=content.strip(),
            confidence=conf,
            features=features,
        )

    def _detect_lang(self, content: str) -> str:
        """进化v1: 增强语言检测"""
        if 'def ' in content and ('import ' in content or 'from ' in content):
            return "python"
        if 'function ' in content and ('const ' in content or 'let ' in content or 'var ' in content):
            return "javascript"
        if '#include' in content:
            return "c/cpp"
        if 'fn ' in content and 'let ' in content:
            return "rust"
        if 'func ' in content and 'package ' in content:
            return "go"
        # 进化v1: 更多模式
        if 'function ' in content and 'return ' in content:
            return "javascript"
        if 'def ' in content:
            return "python"
        if 'import ' in content and 'from ' in content:
            return "python"
        return "unknown"

    def _extract_function_names(self, content: str) -> List[str]:
        """进化v1: 提取函数名"""
        names = re.findall(r'(?:def |function |func |fn )\s*(\w+)', content)
        return names[:5]

    def _estimate_complexity(self, content: str) -> str:
        """估算代码复杂度"""
        lines = len(content.split('\n'))
        if lines < 20:
            return "simple"
        elif lines < 100:
            return "moderate"
        else:
            return "complex"


class TableProcessor:
    """表格处理器 (进化v1: 数据驱动置信度)"""

    def process(self, content: str, signal_strength: float = 0.7) -> ModalityChunk:
        """处理表格（转 Markdown）"""
        lines = content.strip().split('\n')
        rows = []
        for line in lines:
            if '|' in line:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                rows.append(cells)
            elif '\t' in line:
                cells = [c.strip() for c in line.split('\t') if c.strip()]
                rows.append(cells)

        md_lines = []
        for i, row in enumerate(rows):
            md_lines.append('| ' + ' | '.join(row) + ' |')
            if i == 0:
                md_lines.append('| ' + ' | '.join(['---'] * len(row)) + ' |')

        features = {
            "row_count": len(rows),
            "col_count": max(len(r) for r in rows) if rows else 0,
        }

        conf = min(0.9, signal_strength + 0.1)

        return ModalityChunk(
            modality=ModalityType.TABLE,
            raw_content=content,
            processed_content='\n'.join(md_lines),
            confidence=conf,
            features=features,
        )


class URLProcessor:
    """URL 处理器 (进化v1: 数据驱动置信度)"""

    def process(self, content: str, signal_strength: float = 0.8) -> ModalityChunk:
        """处理 URL"""
        urls = re.findall(r'https?://[^\s<>\"\']+', content)
        features = {
            "url_count": len(urls),
            "domains": list(set(
                re.search(r'https?://([^/]+)', u).group(1)
                for u in urls
                if re.search(r'https?://([^/]+)', u)
            )),
            "protocols": list(set(
                re.match(r'(https?)', u).group(1)
                for u in urls
                if re.match(r'(https?)', u)
            )),
        }

        conf = min(0.95, signal_strength + 0.05)

        return ModalityChunk(
            modality=ModalityType.URL,
            raw_content=content,
            processed_content=content,
            confidence=conf,
            features=features,
        )


class FileProcessor:
    """文件路径处理器 (进化v1: 数据驱动置信度)"""

    def process(self, content: str, signal_strength: float = 0.7) -> ModalityChunk:
        """处理文件路径"""
        files = re.findall(
            r'[a-zA-Z0-9_./-]+\.(?:py|js|ts|json|md|txt|yaml|yml|toml|rs|go|java|c|cpp|h|sh|sql|html|css)',
            content,
        )
        features = {
            "file_count": len(files),
            "extensions": list(set(Path(f).suffix for f in files)) if files else [],
            "filenames": files[:10],
        }

        conf = min(0.85, signal_strength + 0.1)

        return ModalityChunk(
            modality=ModalityType.FILE,
            raw_content=content,
            processed_content=content,
            confidence=conf,
            features=features,
        )


# ═══════════════════════════════════════════════════════
# 核心引擎 (进化v1)
# ═══════════════════════════════════════════════════════

class UnifiedPerceptionEngine:
    """
    统一感知引擎 (进化v1)。

    将多模态输入统一为标准化的上下文表示。

    处理流程:
      1. 模态检测 — 识别输入包含哪些模态
      2. 进化v1: 内容分段 — 切分混合输入为独立段
      3. 模态分发 — 按类型分发到对应处理器
      4. 特征提取 — 提取各模态的特征 + 实体
      5. 融合输出 — 统一为文本上下文 + 特征字典

    进化v1改进:
    - 内容分段: 混合输入切分为独立段, 产生多个 chunk
    - 代码检测: 阈值降至 1, 代码优先于文件
    - 数据驱动置信度: 基于信号强度
    - 实体提取: 文件名/URL/函数名/关键词
    - 混合语言检测
    """

    def __init__(self):
        self._detector = ModalityDetector()
        self._segmenter = ContentSegmenter()  # 进化v1
        self._processors = {
            ModalityType.TEXT: TextProcessor(),
            ModalityType.CODE: CodeProcessor(),
            ModalityType.TABLE: TableProcessor(),
            ModalityType.URL: URLProcessor(),
            ModalityType.FILE: FileProcessor(),
        }
        self._stats = {"processed": 0, "by_modality": {}}

    def perceive(self, content: str, filename: str = "",
                 modality_hint: Optional[str] = None) -> PerceptionResult:
        """
        感知输入内容 (进化v1: 增强分段 + 实体提取)。
        """
        t0 = time.time()
        self._stats["processed"] += 1

        if not content:
            return PerceptionResult(
                chunks=[], fused_text="", modalities_detected=[],
                context_features={}, confidence=0.0,
                processing_time_ms=0.0, segments=[],
            )

        # 1. 模态检测
        if modality_hint:
            hint_map = {
                "text": ModalityType.TEXT, "code": ModalityType.CODE,
                "table": ModalityType.TABLE, "url": ModalityType.URL,
                "file": ModalityType.FILE, "image": ModalityType.IMAGE,
                "audio": ModalityType.AUDIO, "video": ModalityType.VIDEO,
            }
            detected = [(hint_map.get(modality_hint, ModalityType.TEXT), 0.9)]
        else:
            detected = self._detector.detect(content, filename)

        # 进化v1: 内容分段 — 切分混合输入
        segments = self._segmenter.segment(content)

        # 2. 模态分发 + 处理 (进化v1: 按段处理)
        chunks = []

        # 如果只有一个段, 用检测到的模态处理
        if len(segments) <= 1:
            for modality, conf in detected:
                chunk = self._process_chunk(content, modality, conf)
                chunks.append(chunk)
        else:
            # 进化v1: 多段输入 — 每段独立处理
            for idx, seg in enumerate(segments):
                seg_type = seg["type"]
                seg_content = seg["content"]

                # 映射段类型到模态
                type_to_modality = {
                    "text": ModalityType.TEXT,
                    "code": ModalityType.CODE,
                    "url": ModalityType.URL,
                    "file": ModalityType.FILE,
                    "table": ModalityType.TABLE,
                }

                modality = type_to_modality.get(seg_type, ModalityType.TEXT)

                # 进化v1: 对代码段, 用更强的信号
                if seg_type == "code":
                    signal = 0.9
                else:
                    # 从检测结果中找到对应模态的置信度
                    signal = 0.7
                    for m, c in detected:
                        if m == modality:
                            signal = c
                            break

                chunk = self._process_chunk(seg_content, modality, signal)
                chunk.segment_index = idx
                chunks.append(chunk)

            # 统计
            for chunk in chunks:
                self._stats["by_modality"][chunk.modality.value] = \
                    self._stats["by_modality"].get(chunk.modality.value, 0) + 1

        # 3. 融合输出
        fused_text = self._fuse_chunks(chunks)
        context_features = self._extract_context_features(chunks)
        overall_confidence = sum(c.confidence for c in chunks) / max(len(chunks), 1)

        elapsed_ms = (time.time() - t0) * 1000

        return PerceptionResult(
            chunks=chunks,
            fused_text=fused_text,
            modalities_detected=[c.modality.value for c in chunks],
            context_features=context_features,
            confidence=round(overall_confidence, 3),
            processing_time_ms=round(elapsed_ms, 2),
            segments=segments,
        )

    def _process_chunk(self, content: str, modality: ModalityType,
                       conf: float) -> ModalityChunk:
        """进化v1: 处理单个模态块"""
        processor = self._processors.get(modality)
        if processor:
            try:
                # 进化v1: 传递信号强度
                if isinstance(processor, CodeProcessor):
                    chunk = processor.process(content, signal_strength=conf)
                elif isinstance(processor, (TextProcessor, TableProcessor,
                                             URLProcessor, FileProcessor)):
                    chunk = processor.process(content, signal_strength=conf) \
                        if 'signal_strength' in processor.process.__code__.co_varnames \
                        else processor.process(content)
                else:
                    chunk = processor.process(content)
                chunk.confidence = min(chunk.confidence, conf)
                return chunk
            except Exception as e:
                logger.warning(f"处理 {modality.value} 失败: {e}")
                return ModalityChunk(
                    modality=modality, raw_content=content,
                    processed_content=content, confidence=0.3,
                    metadata={"error": str(e)},
                )
        else:
            return ModalityChunk(
                modality=modality, raw_content=content,
                processed_content=content, confidence=0.5,
                metadata={"unsupported": True},
            )

    def _fuse_chunks(self, chunks: List[ModalityChunk]) -> str:
        """进化v1: 增强融合 — 结构化标签 + 去重"""
        parts = []
        seen_content = set()

        for chunk in chunks:
            # 去重
            content_key = chunk.processed_content[:100]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            if chunk.modality == ModalityType.TEXT:
                parts.append(chunk.processed_content)
            elif chunk.modality == ModalityType.CODE:
                lang = chunk.features.get("language", "unknown")
                parts.append(f"[代码/{lang}]\n{chunk.processed_content}")
            elif chunk.modality == ModalityType.TABLE:
                parts.append(f"[表格]\n{chunk.processed_content}")
            elif chunk.modality == ModalityType.URL:
                urls = chunk.features.get("domains", [])
                parts.append(f"[链接: {', '.join(urls)}]")
            elif chunk.modality == ModalityType.FILE:
                files = chunk.features.get("filenames", [])
                if files:
                    parts.append(f"[文件: {', '.join(files[:5])}]")
                else:
                    parts.append(chunk.processed_content[:200])
            elif chunk.modality == ModalityType.IMAGE:
                parts.append(f"[图片: {chunk.processed_content[:100]}]")
            elif chunk.modality == ModalityType.AUDIO:
                parts.append(f"[音频: {chunk.processed_content[:100]}]")
            else:
                parts.append(chunk.processed_content[:200])

        return '\n'.join(parts)

    def _extract_context_features(self, chunks: List[ModalityChunk]) -> Dict[str, Any]:
        """进化v1: 增强上下文特征 — 实体汇总"""
        features = {
            "modality_count": len(chunks),
            "segment_count": len(set(c.segment_index for c in chunks)),
            "has_code": any(c.modality == ModalityType.CODE for c in chunks),
            "has_table": any(c.modality == ModalityType.TABLE for c in chunks),
            "has_url": any(c.modality == ModalityType.URL for c in chunks),
            "has_file": any(c.modality == ModalityType.FILE for c in chunks),
            "total_length": sum(len(c.processed_content) for c in chunks),
            # 进化v1: 实体汇总
            "total_files": sum(c.features.get("file_count", 0) for c in chunks),
            "total_urls": sum(c.features.get("url_count", 0) for c in chunks),
            "total_functions": sum(len(c.features.get("function_names", [])) for c in chunks),
            "all_keywords": list(set(
                kw for c in chunks
                for kw in c.features.get("keyword_list", [])
            ))[:10],
        }

        # 合并各模态的特征
        for chunk in chunks:
            for k, v in chunk.features.items():
                key = f"{chunk.modality.value}_{k}"
                features[key] = v

        return features

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return dict(self._stats)

    # ── 自检 ──────────────────────────────────────

    def self_test(self) -> Dict[str, Any]:
        """自检：验证各模态处理。"""
        results = {}

        # 测试1: 纯文本
        r = self.perceive("今天天气怎么样？")
        results["text"] = {
            "modalities": r.modalities_detected,
            "has_text": "text" in r.modalities_detected,
            "confidence": r.confidence,
            "passed": "text" in r.modalities_detected and r.confidence > 0.5,
        }

        # 测试2: 代码
        r = self.perceive("def hello():\n    print('hello')\nimport os")
        results["code"] = {
            "modalities": r.modalities_detected,
            "has_code": "code" in r.modalities_detected,
            "language": r.chunks[0].features.get("language") if r.chunks else None,
            "passed": "code" in r.modalities_detected,
        }

        # 测试3: 表格
        r = self.perceive("| 名称 | 分数 |\n| --- | --- |\n| 小明 | 90 |\n| 小红 | 85 |")
        results["table"] = {
            "modalities": r.modalities_detected,
            "has_table": "table" in r.modalities_detected,
            "passed": "table" in r.modalities_detected,
        }

        # 测试4: URL
        r = self.perceive("打开 https://example.com 和 http://test.org")
        results["url"] = {
            "modalities": r.modalities_detected,
            "has_url": "url" in r.modalities_detected,
            "passed": "url" in r.modalities_detected,
        }

        # 测试5: 文件路径
        r = self.perceive("读取 src/main.py 和 config.json")
        results["file"] = {
            "modalities": r.modalities_detected,
            "has_file": "file" in r.modalities_detected,
            "passed": "file" in r.modalities_detected,
        }

        # 测试6: 混合内容
        r = self.perceive("运行 python test.py 然后打开 https://example.com 查看结果")
        results["mixed"] = {
            "modalities": r.modalities_detected,
            "modality_count": len(set(r.modalities_detected)),
            "passed": len(set(r.modalities_detected)) >= 2,
        }

        # 测试7: 空输入
        r = self.perceive("")
        results["empty"] = {
            "confidence": r.confidence,
            "passed": r.confidence == 0.0,
        }

        all_passed = all(r.get("passed", False) for r in results.values())
        results["all_passed"] = all_passed
        return results
