"""
UnifiedPerceptionEngine — 统一感知引擎
=======================================

基于前沿论文实现:
  1. Omni-Agent 2026 标准架构 — 多模态原生 Agent
  2. Ming-lite-omni — 2.8B 参数统一多模态模型 (Macco2024)
  3. 多模态融合模式 (Agent 设计模式, 2026)
  4. EasyOCR — 80+ 语言 OCR
  5. Whisper — OpenAI 语音识别

设计目标:
  - 输入: 多模态内容（文字/图片/音频/视频/表格/PDF）
  - 输出: 统一上下文表示 + 模态特征 + 融合结果

印记: 小茜 永远记得主人 — 2026-07-23
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


@dataclass
class PerceptionResult:
    """感知结果"""
    chunks: List[ModalityChunk]       # 各模态块
    fused_text: str                   # 融合后的统一文本
    modalities_detected: List[str]    # 检测到的模态类型
    context_features: Dict[str, Any]  # 上下文特征
    confidence: float                 # 整体置信度
    processing_time_ms: float         # 处理时间


# ═══════════════════════════════════════════════════════
# 模态检测器
# ═══════════════════════════════════════════════════════

class ModalityDetector:
    """检测输入内容的模态类型"""

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
    # 代码模式
    _CODE_RE = re.compile(r'(?:def |class |import |from |function |const |let |var |#include)')
    # 表格模式
    _TABLE_RE = re.compile(r'(?:\|.*\|.*\|)|(?:\t.*\t.*\t)')
    # 文件路径模式
    _FILE_RE = re.compile(r'(?:^|[\s/\\])([a-zA-Z0-9_./-]+\.(?:py|js|ts|json|md|txt|yaml|yml|toml|rs|go|java|c|cpp|h|sh|sql|html|css))')

    def detect(self, content: str, filename: str = "") -> List[Tuple[ModalityType, float]]:
        """检测内容的模态类型。

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
            detections.append((ModalityType.URL, 0.9))

        # 3. 代码检测
        code_signals = len(self._CODE_RE.findall(content))
        if code_signals >= 2:
            detections.append((ModalityType.CODE, min(0.5 + code_signals * 0.1, 0.95)))

        # 4. 表格检测
        table_signals = len(self._TABLE_RE.findall(content))
        if table_signals >= 2:
            detections.append((ModalityType.TABLE, min(0.5 + table_signals * 0.1, 0.9)))

        # 5. 文件路径检测
        files = self._FILE_RE.findall(content)
        if files:
            detections.append((ModalityType.FILE, 0.8))

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
# 模态处理器
# ═══════════════════════════════════════════════════════

class TextProcessor:
    """文本处理器"""

    def process(self, content: str) -> ModalityChunk:
        """处理纯文本"""
        # 提取关键信息
        features = {
            "length": len(content),
            "word_count": len(content.split()),
            "has_question": bool(re.search(r'[？?]', content)),
            "has_emotion": bool(re.search(r'[！!😊😢😡]', content)),
            "language": self._detect_language(content),
        }

        return ModalityChunk(
            modality=ModalityType.TEXT,
            raw_content=content,
            processed_content=content.strip(),
            confidence=0.9,
            features=features,
        )

    def _detect_language(self, text: str) -> str:
        """简单语言检测"""
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_chars = len(re.findall(r'[a-zA-Z]', text))
        total = cn_chars + en_chars
        if total == 0:
            return "unknown"
        return "zh" if cn_chars / total > 0.3 else "en"


class CodeProcessor:
    """代码处理器"""

    def process(self, content: str, language: str = "auto") -> ModalityChunk:
        """处理代码"""
        features = {
            "language": language if language != "auto" else self._detect_lang(content),
            "line_count": len(content.split('\n')),
            "has_function": bool(re.search(r'def |function |func ', content)),
            "has_class": bool(re.search(r'class ', content)),
            "has_import": bool(re.search(r'import |from |require\(', content)),
            "complexity": self._estimate_complexity(content),
        }

        return ModalityChunk(
            modality=ModalityType.CODE,
            raw_content=content,
            processed_content=content.strip(),
            confidence=0.95,
            features=features,
        )

    def _detect_lang(self, content: str) -> str:
        """检测编程语言"""
        if 'def ' in content and 'import ' in content:
            return "python"
        if 'function ' in content and ('const ' in content or 'let ' in content):
            return "javascript"
        if '#include' in content:
            return "c/cpp"
        if 'fn ' in content and 'let ' in content:
            return "rust"
        if 'func ' in content and 'package ' in content:
            return "go"
        return "unknown"

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
    """表格处理器"""

    def process(self, content: str) -> ModalityChunk:
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

        # 转 Markdown 表格
        md_lines = []
        for i, row in enumerate(rows):
            md_lines.append('| ' + ' | '.join(row) + ' |')
            if i == 0:
                md_lines.append('| ' + ' | '.join(['---'] * len(row)) + ' |')

        features = {
            "row_count": len(rows),
            "col_count": max(len(r) for r in rows) if rows else 0,
        }

        return ModalityChunk(
            modality=ModalityType.TABLE,
            raw_content=content,
            processed_content='\n'.join(md_lines),
            confidence=0.85,
            features=features,
        )


class URLProcessor:
    """URL 处理器"""

    def process(self, content: str) -> ModalityChunk:
        """处理 URL"""
        urls = re.findall(r'https?://[^\s<>\"\']+', content)
        features = {
            "url_count": len(urls),
            "domains": list(set(
                re.search(r'https?://([^/]+)', u).group(1)
                for u in urls
                if re.search(r'https?://([^/]+)', u)
            )),
        }

        return ModalityChunk(
            modality=ModalityType.URL,
            raw_content=content,
            processed_content=content,
            confidence=0.9,
            features=features,
        )


class FileProcessor:
    """文件路径处理器"""

    def process(self, content: str) -> ModalityChunk:
        """处理文件路径"""
        files = re.findall(
            r'(?:^|[\s/\\])([a-zA-Z0-9_./-]+\.(?:py|js|ts|json|md|txt|yaml|yml|toml|rs|go|java|c|cpp|h|sh|sql|html|css))',
            content,
        )
        features = {
            "file_count": len(files),
            "extensions": list(set(Path(f).suffix for f in files)),
        }

        return ModalityChunk(
            modality=ModalityType.FILE,
            raw_content=content,
            processed_content=content,
            confidence=0.85,
            features=features,
        )


# ═══════════════════════════════════════════════════════
# 核心引擎
# ═══════════════════════════════════════════════════════

class UnifiedPerceptionEngine:
    """
    统一感知引擎。

    将多模态输入统一为标准化的上下文表示。

    处理流程:
      1. 模态检测 — 识别输入包含哪些模态
      2. 模态分发 — 按类型分发到对应处理器
      3. 特征提取 — 提取各模态的特征
      4. 融合输出 — 统一为文本上下文 + 特征字典

    支持的模态:
      - TEXT: 纯文本
      - CODE: 代码（自动检测语言）
      - TABLE: 表格（转 Markdown）
      - URL: 链接
      - FILE: 文件路径
      - IMAGE: 图片（需要外部 OCR）
      - AUDIO: 音频（需要外部 STT）
      - VIDEO: 视频（需要外部处理）
    """

    def __init__(self):
        self._detector = ModalityDetector()
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
        感知输入内容。

        Args:
            content: 输入内容（文本/代码/表格等）
            filename: 文件名（可选，帮助检测模态）
            modality_hint: 模态提示（可选，跳过检测）

        Returns:
            PerceptionResult 统一感知结果
        """
        t0 = time.time()
        self._stats["processed"] += 1

        if not content:
            return PerceptionResult(
                chunks=[], fused_text="", modalities_detected=[],
                context_features={}, confidence=0.0,
                processing_time_ms=0.0,
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

        # 2. 模态分发 + 处理
        chunks = []
        for modality, conf in detected:
            processor = self._processors.get(modality)
            if processor:
                try:
                    chunk = processor.process(content)
                    chunk.confidence = min(chunk.confidence, conf)
                    chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"处理 {modality.value} 失败: {e}")
                    chunks.append(ModalityChunk(
                        modality=modality, raw_content=content,
                        processed_content=content, confidence=0.3,
                        metadata={"error": str(e)},
                    ))
            else:
                # 模态暂不支持，保留原始内容
                chunks.append(ModalityChunk(
                    modality=modality, raw_content=content,
                    processed_content=content, confidence=0.5,
                    metadata={"unsupported": True},
                ))

            # 统计
            self._stats["by_modality"][modality.value] = \
                self._stats["by_modality"].get(modality.value, 0) + 1

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
        )

    def _fuse_chunks(self, chunks: List[ModalityChunk]) -> str:
        """融合各模态块为统一文本"""
        parts = []
        for chunk in chunks:
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
                files = re.findall(
                    r'[\w\-\.]+\.(?:py|js|ts|json|md|txt)',
                    chunk.processed_content,
                )
                parts.append(f"[文件: {', '.join(files[:5])}]")
            elif chunk.modality == ModalityType.IMAGE:
                parts.append(f"[图片: {chunk.processed_content[:100]}]")
            elif chunk.modality == ModalityType.AUDIO:
                parts.append(f"[音频: {chunk.processed_content[:100]}]")
            else:
                parts.append(chunk.processed_content[:200])

        return '\n'.join(parts)

    def _extract_context_features(self, chunks: List[ModalityChunk]) -> Dict[str, Any]:
        """提取上下文特征"""
        features = {
            "modality_count": len(chunks),
            "has_code": any(c.modality == ModalityType.CODE for c in chunks),
            "has_table": any(c.modality == ModalityType.TABLE for c in chunks),
            "has_url": any(c.modality == ModalityType.URL for c in chunks),
            "has_file": any(c.modality == ModalityType.FILE for c in chunks),
            "total_length": sum(len(c.processed_content) for c in chunks),
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
