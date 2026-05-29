"""
RAG 系统运行入口 - 后端版（整合版）
功能：
1. Markdown智能分块
2. 混合检索（BM25 + 向量）
3. 索引持久化（保存 / 加载）
4. 测试集生成（含普通题 / 消歧题 / 综合题 / 推理题 / 多跳题 / 半导体术语题 / 负样本）
5. 测试集评估（检索 / 回答 / 分组报表）
6. 支持 Markdown YAML front matter 元数据
7. 建索引时纳入 metadata
8. 文件名命中时优先该文件，但不会完全锁死召回
"""

import os
import re
import math
import json
import random
import hashlib
import numpy as np
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, TypedDict
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict

import requests

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

try:
    import yaml
except Exception:
    yaml = None

try:
    import rjieba
except Exception:
    rjieba = None

try:
    from langgraph.graph import END, START, StateGraph
except Exception:
    END = START = StateGraph = None


# ============================================================
# 配置区域
# ============================================================
load_dotenv()

# API 配置
LLM_URL = os.environ.get("LLM_URL")
LLM_KEY = os.environ.get("LLM_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "abyss/abyss-chat")

def derive_embedding_url() -> Optional[str]:
    configured = os.environ.get("EMBEDDING_URL") or os.environ.get("EMBEDDING_API_URL")
    if configured:
        return configured
    if not LLM_URL:
        return None

    llm_url = LLM_URL.rstrip("/")
    suffix = "/chat/completions"
    if llm_url.endswith(suffix):
        return llm_url[:-len(suffix)] + "/embeddings"
    return None


EMBEDDING_URL = derive_embedding_url()
EMBEDDING_KEY = os.environ.get("EMBEDDING_KEY") or os.environ.get("EMBEDDING_API_KEY") or LLM_KEY
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "8"))
EMBEDDING_TIMEOUT = int(os.environ.get("EMBEDDING_TIMEOUT", "120"))
EMBEDDING_DIMENSIONS = os.environ.get("EMBEDDING_DIMENSIONS")

# 知识库配置
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
TOP_K = int(os.environ.get("TOP_K", "10"))

# 混合检索权重配置
VECTOR_WEIGHT = float(os.environ.get("VECTOR_WEIGHT", "0.6"))
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "0.4"))

# 元数据 / 文件名增强配置
FILENAME_MATCH_BOOST = float(os.environ.get("FILENAME_MATCH_BOOST", "0.12"))
METADATA_MATCH_BOOST = float(os.environ.get("METADATA_MATCH_BOOST", "0.08"))
PREFERRED_DOC_MAX_RATIO = float(os.environ.get("PREFERRED_DOC_MAX_RATIO", "0.6"))
MAX_INDEX_METADATA_CHARS = int(os.environ.get("MAX_INDEX_METADATA_CHARS", "300"))

# 知识源
_default_knowledge = os.environ.get("KNOWLEDGE_SOURCE", r"D:/my_work/仓库_新/Requirements Library")
KNOWLEDGE_SOURCE = [p.strip() for p in _default_knowledge.split("||") if p.strip()]
KB_SAVE_PATH = os.environ.get("KB_SAVE_PATH")

DEBUG = os.environ.get("DEBUG", "true").lower() in {"1", "true", "yes", "y"}
DEFAULT_REFUSAL_ANSWER = "根据提供的文档，没有找到相关信息。"
MIN_CHUNK_CHARS = int(os.environ.get("MIN_CHUNK_CHARS", "80"))
REQUIRED_RAG_PACKAGES = "rjieba langgraph"


# ============================================================
# 工具函数
# ============================================================
def require_dependency(module, package_name: str, feature: str):
    if module is None:
        raise ImportError(
            f"缺少依赖 {package_name}，无法{feature}。"
            f"请先安装: pip install {REQUIRED_RAG_PACKAGES}"
        )


def require_rag_runtime_dependencies():
    require_dependency(rjieba, "rjieba", "进行 BM25 分词")
    require_dependency(StateGraph, "langgraph", "初始化 RAG Pipeline")


def require_embedding_runtime_dependencies():
    if not EMBEDDING_URL:
        raise ValueError(
            "EMBEDDING_URL 未配置，无法调用 Embedding API。"
            "如果 LLM_URL 是 OpenAI 兼容的 /chat/completions 地址，"
            "也可以不显式配置 EMBEDDING_URL，程序会自动推导 /embeddings。"
        )


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？；：、,.!?;:\"'“”‘’（）()\[\]【】{}<>《》]", "", text)
    return text


def first_header_line(header_context: str, max_len: int = 120) -> str:
    if not header_context:
        return "(无章节)"
    line = header_context.splitlines()[0].strip() if header_context.splitlines() else "(无章节)"
    if len(line) > max_len:
        line = line[:max_len - 3] + "..."
    return line or "(无章节)"


def sort_metric_dict(data: Dict[str, Dict], primary_key: str = "count") -> Dict[str, Dict]:
    items = sorted(data.items(), key=lambda kv: (-kv[1].get(primary_key, 0), kv[0]))
    return {k: v for k, v in items}


def make_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    return str(value)


def flatten_dict_items(data, parent_key: str = "") -> List[Tuple[str, str]]:
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            key = f"{parent_key}.{k}" if parent_key else str(k)
            items.extend(flatten_dict_items(v, key))
    elif isinstance(data, list):
        if all(not isinstance(x, (dict, list)) for x in data):
            if parent_key:
                items.append((parent_key, ", ".join(str(x) for x in data if x is not None)))
        else:
            for i, v in enumerate(data):
                key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                items.extend(flatten_dict_items(v, key))
    else:
        if parent_key:
            items.append((parent_key, "" if data is None else str(data)))
    return items


def format_front_matter_text(front_matter: Dict, max_items: int = 20, max_len: int = 300) -> str:
    if not isinstance(front_matter, dict) or not front_matter:
        return ""
    pairs = flatten_dict_items(front_matter)
    parts = []
    for k, v in pairs:
        v = str(v).strip()
        if not k or not v:
            continue
        parts.append(f"{k}={v}")
        if len(parts) >= max_items:
            break
    text = "; ".join(parts)
    if len(pairs) > max_items:
        text += " ..."
    if len(text) > max_len:
        text = text[:max_len - 3] + "..."
    return text


def parse_simple_yaml_scalar(value: str):
    v = value.strip().strip('"').strip("'")
    if not v:
        return ""
    lower = v.lower()
    if lower in {"true", "yes"}:
        return True
    if lower in {"false", "no"}:
        return False
    if lower in {"null", "none", "~"}:
        return None
    if re.fullmatch(r"-?\d+", v):
        try:
            return int(v)
        except Exception:
            pass
    if re.fullmatch(r"-?\d+\.\d+", v):
        try:
            return float(v)
        except Exception:
            pass
    return v


def parse_simple_yaml(text: str) -> Dict:
    result = {}
    current_key = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        m = re.match(r'^([^\s:#][^:]*?)\s*:\s*(.*)$', line)
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()
            if value == "":
                result[key] = []
                current_key = key
            else:
                result[key] = parse_simple_yaml_scalar(value)
                current_key = key
            continue

        m = re.match(r'^\s*-\s+(.*)$', line)
        if m and current_key:
            if not isinstance(result.get(current_key), list):
                result[current_key] = [result.get(current_key)]
            result[current_key].append(parse_simple_yaml_scalar(m.group(1)))
        else:
            current_key = None
    return result


def extract_markdown_front_matter(text: str) -> Tuple[Dict, str, str]:
    """
    提取 Markdown 顶部 front matter:
    ---
    key: value
    ---
    正文...
    返回: (parsed_meta, body, raw_yaml)
    """
    if not text:
        return {}, "", ""

    text = text.lstrip("\ufeff")
    match = re.match(r'^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)', text, re.DOTALL)
    if not match:
        return {}, text, ""

    raw = match.group(1).strip()
    if not raw or ":" not in raw:
        return {}, text, ""

    body = text[match.end():]
    parsed = {}

    if yaml is not None:
        try:
            obj = yaml.safe_load(raw)
            if isinstance(obj, dict):
                parsed = obj
            elif obj is not None:
                parsed = {"value": obj}
        except Exception:
            parsed = parse_simple_yaml(raw)
    else:
        parsed = parse_simple_yaml(raw)

    parsed = make_json_safe(parsed or {})
    return parsed, body, raw


def format_chunk_attrs(metadata: Dict, max_items: int = 12, max_len: int = 240) -> str:
    if not metadata:
        return ""
    front = metadata.get("front_matter") or {}
    return format_front_matter_text(front, max_items=max_items, max_len=max_len)

def front_matter_json_text(metadata: Dict, max_len: int = 1200) -> str:
    """把 front matter 转成更适合喂给 LLM 的 JSON 文本"""
    if not metadata:
        return ""
    front = metadata.get("front_matter") or {}
    if not front:
        return ""
    try:
        text = json.dumps(front, ensure_ascii=False, indent=2)
    except Exception:
        text = str(front)
    if len(text) > max_len:
        text = text[:max_len - 3] + "..."
    return text


def build_result_front_matter_summary(results, max_docs: int = 3, max_len: int = 220) -> str:
    """
    从检索结果里提取去重后的 front matter 摘要，
    用于在最终回答后做“检索补充属性”展示
    """
    lines = []
    seen = set()

    for r in results:
        source_key = (
            r.chunk.metadata.get("source")
            or r.chunk.metadata.get("filename")
            or r.chunk.doc_id
        )
        if source_key in seen:
            continue
        seen.add(source_key)

        front = r.chunk.metadata.get("front_matter") or {}
        if not front:
            continue

        source = r.chunk.metadata.get("filename", r.chunk.doc_id)
        attrs = format_front_matter_text(front, max_items=12, max_len=max_len)
        if not attrs:
            continue

        lines.append(f"- {source}: {attrs}")
        if len(lines) >= max_docs:
            break

    return "\n".join(lines)

def get_case_gold_sources(tc) -> List[Dict]:
    if getattr(tc, "gold_sources", None):
        return tc.gold_sources
    if getattr(tc, "source_path", "") and getattr(tc, "source_chunk_id", -1) >= 0:
        return [{
            "source_doc": tc.source_doc,
            "source_path": tc.source_path,
            "chunk_id": tc.source_chunk_id,
            "header": tc.source_header,
            "role": "primary"
        }]
    return []


def get_case_doc_group(tc) -> str:
    golds = get_case_gold_sources(tc)
    if not golds:
        return getattr(tc, "origin_doc", "") or "(未知来源)"
    docs = sorted({g.get("source_doc", "(未知文档)") for g in golds})
    if len(docs) == 1:
        return docs[0]
    label = "MULTI::" + " | ".join(docs[:3])
    if len(docs) > 3:
        label += " ..."
    return label


def get_case_header_group(tc) -> str:
    golds = get_case_gold_sources(tc)
    if not golds:
        return f"{getattr(tc, 'origin_doc', '(未知来源)')} :: {first_header_line(getattr(tc, 'origin_header', ''), 80)}"
    if len(golds) == 1:
        g = golds[0]
        return f"{g.get('source_doc', '(未知文档)')} :: {first_header_line(g.get('header', ''), 80)}"
    parts = []
    for g in golds[:3]:
        parts.append(f"{g.get('source_doc', '(未知文档)')}::{first_header_line(g.get('header', ''), 50)}")
    label = "MULTI::" + " | ".join(parts)
    if len(golds) > 3:
        label += " ..."
    return label


# ============================================================
# 数据类
# ============================================================
@dataclass
class Document:
    content: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Chunk:
    content: str
    metadata: Dict
    doc_id: str
    chunk_id: int
    header_context: str = ""


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    metadata_score: float = 0.0
    filename_boost: float = 0.0


class RAGPipelineState(TypedDict, total=False):
    question: str
    top_k: int
    verbose: bool
    results: List[SearchResult]
    context: str
    messages: List[Dict]
    answer: str
    error: str


@dataclass
class TestCase:
    case_id: str
    question: str
    expected_behavior: str = "answer"   # answer | refuse
    expected_answer: str = ""
    evidence: str = ""
    case_type: str = "factoid"
    difficulty: str = "medium"
    source_doc: str = ""
    source_path: str = ""
    source_chunk_id: int = -1
    source_header: str = ""
    origin_doc: str = ""
    origin_path: str = ""
    origin_chunk_id: int = -1
    origin_header: str = ""
    gold_sources: List[Dict] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


# ============================================================
# Markdown 智能分块器
# ============================================================
class MarkdownChunker:
    """Markdown感知分块器：保护图片/代码块，过滤伪空白，合并短块"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_chars: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 4)
        self.min_chunk_chars = max(20, min_chunk_chars)
        self.max_merge_size = int(chunk_size * 1.35)

        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        self.list_pattern = re.compile(r'^(\s*[-*+]|\s*\d+\.)\s+', re.MULTILINE)

        self.md_image_pattern = re.compile(r'!\[[^\]]*\]\([^)]+\)')
        self.md_ref_image_use_pattern = re.compile(r'!\[[^\]]*\]\[[^\]]+\]')
        self.md_ref_def_pattern = re.compile(r'^\s*\[[^\]]+\]:\s+\S+.*$')
        self.html_img_pattern = re.compile(r'<img\b[^>]*>', re.IGNORECASE)

        self.html_comment_pattern = re.compile(r'<!--[\s\S]*?-->')
        self.anchor_pattern = re.compile(
            r'^\s*(<a\b[^>]*>\s*</a>|<a\b[^>]*/?>)\s*$',
            re.IGNORECASE
        )
        self.separator_pattern = re.compile(r'^\s*([-*_])(?:\s*\1){2,}\s*$')

    def split(self, text: str, metadata: Dict = None) -> List[Tuple[str, str]]:
        """
        智能分割Markdown文档
        返回: List[(chunk_content, header_context)]
        """
        if not text or not text.strip():
            return []

        sections = self._split_by_headers(text)
        chunks = []

        for header, content in sections:
            if not content.strip():
                continue

            section_text = f"{header}\n{content}".strip() if header else content.strip()
            body = self._strip_header_prefix(section_text, header)

            if not body or self._is_semantic_empty_text(body):
                continue

            if len(section_text) <= self.chunk_size:
                chunks.append((section_text, header))
            else:
                chunks.extend(self._split_large_section(content, header))

        return self._post_process_chunks(chunks)

    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        lines = text.split('\n')
        sections = []
        current_headers = {}
        current_content = []
        current_header = ""

        for line in lines:
            header_match = self.header_pattern.match(line)
            if header_match:
                if current_content or current_header:
                    content = '\n'.join(current_content).strip()
                    sections.append((current_header, content))

                level = len(header_match.group(1))
                header_text = line

                for l in list(current_headers.keys()):
                    if l >= level:
                        del current_headers[l]

                current_headers[level] = header_text
                header_context_parts = [current_headers[l] for l in sorted(current_headers.keys())]
                current_header = '\n'.join(header_context_parts)
                current_content = []
            else:
                current_content.append(line)

        if current_content or current_header:
            content = '\n'.join(current_content).strip()
            sections.append((current_header, content))

        return sections if sections else [("", text)]

    def _split_large_section(self, content: str, header: str) -> List[Tuple[str, str]]:
        chunks = []
        code_blocks = []

        def save_code_block(match):
            code_blocks.append(match.group(0))
            return f"\n<<CODE_BLOCK_{len(code_blocks)-1}>>\n"

        protected_content = self.code_block_pattern.sub(save_code_block, content)
        paragraphs = self._split_into_paragraphs(protected_content)

        current_chunk = ""
        header_prefix = f"{header}\n\n" if header else ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            for i, code in enumerate(code_blocks):
                para = para.replace(f"<<CODE_BLOCK_{i}>>", code)

            if self._is_semantic_empty_text(para):
                continue

            if para.startswith('```') and para.endswith('```'):
                if current_chunk.strip():
                    chunks.append(((header_prefix + current_chunk.strip()).strip(), header))
                    current_chunk = ""

                code_chunk = (header_prefix + para).strip()
                if len(code_chunk) <= self.chunk_size * 1.5:
                    chunks.append((code_chunk, header))
                else:
                    chunks.extend(self._force_split_code(para, header))
                continue

            if self._is_table(para):
                if current_chunk.strip():
                    chunks.append(((header_prefix + current_chunk.strip()).strip(), header))
                    current_chunk = ""
                chunks.append(((header_prefix + para).strip(), header))
                continue

            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            if len(header_prefix + test_chunk) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(((header_prefix + current_chunk.strip()).strip(), header))
                current_chunk = para
            else:
                current_chunk = test_chunk

        if current_chunk.strip():
            chunks.append(((header_prefix + current_chunk.strip()).strip(), header))

        if not chunks:
            fallback = (header_prefix + content).strip()
            return [(fallback, header)] if fallback else []

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        raw_paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
        paragraphs = []
        current_list = []
        pending_media = []

        def flush_list():
            nonlocal current_list
            if current_list:
                block = '\n'.join(current_list).strip()
                if block:
                    paragraphs.append(block)
                current_list = []

        for para in raw_paragraphs:
            if self._is_semantic_empty_text(para):
                continue

            if self._is_media_paragraph(para):
                flush_list()
                if paragraphs:
                    paragraphs[-1] = paragraphs[-1].rstrip() + "\n\n" + para
                else:
                    pending_media.append(para)
                continue

            if pending_media:
                para = "\n\n".join(pending_media + [para])
                pending_media = []

            if self._is_list_item(para):
                current_list.append(para)
            else:
                flush_list()
                paragraphs.append(para)

        flush_list()

        if pending_media:
            if paragraphs:
                paragraphs[-1] = paragraphs[-1].rstrip() + "\n\n" + "\n\n".join(pending_media)
            else:
                paragraphs.extend(pending_media)

        return paragraphs

    def _is_list_item(self, text: str) -> bool:
        lines = text.split('\n')
        if not lines:
            return False
        return bool(self.list_pattern.match(lines[0]))

    def _is_table(self, text: str) -> bool:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < 2:
            return False
        return all(line.startswith('|') and line.endswith('|') for line in lines)

    def _is_media_paragraph(self, text: str) -> bool:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return False

        def is_media_line(line: str) -> bool:
            return bool(
                self.md_image_pattern.fullmatch(line) or
                self.md_ref_image_use_pattern.fullmatch(line) or
                self.md_ref_def_pattern.fullmatch(line) or
                self.html_img_pattern.fullmatch(line)
            )

        return all(is_media_line(line) for line in lines)

    def _is_semantic_empty_text(self, text: str) -> bool:
        if not text or not text.strip():
            return True

        t = self.html_comment_pattern.sub("", text).strip()
        if not t:
            return True

        lines = [l.strip() for l in t.splitlines() if l.strip()]
        if not lines:
            return True

        if all(self.separator_pattern.fullmatch(line) for line in lines):
            return True

        if all(self.anchor_pattern.fullmatch(line) for line in lines):
            return True

        return False

    def _strip_header_prefix(self, content: str, header: str) -> str:
        text = (content or "").strip()
        h = (header or "").strip()
        if h and text.startswith(h):
            return text[len(h):].strip()
        return text

    def _post_process_chunks(self, chunks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        cleaned = []
        for content, header in chunks:
            text = (content or "").strip()
            if not text:
                continue

            body = self._strip_header_prefix(text, header)
            if not body or self._is_semantic_empty_text(body):
                continue

            if cleaned:
                prev_content, prev_header = cleaned[-1]
                prev_body = self._strip_header_prefix(prev_content, prev_header)
                short_prev = len(prev_body) < self.min_chunk_chars
                short_curr = len(body) < self.min_chunk_chars
                same_header = (prev_header == header)
                merged_body = (prev_body.rstrip() + "\n\n" + body.lstrip()).strip()
                merged_text = f"{header}\n\n{merged_body}".strip() if header else merged_body

                if same_header and (short_prev or short_curr) and len(merged_text) <= self.max_merge_size:
                    cleaned[-1] = (merged_text, header)
                    continue

            cleaned.append((text, header))

        return cleaned

    def _force_split_code(self, code_block: str, header: str) -> List[Tuple[str, str]]:
        chunks = []
        lines = code_block.split('\n')
        first_line = lines[0] if lines else '```'
        last_line = '```'
        content_lines = lines[1:-1] if len(lines) > 2 else []
        current_lines = [first_line]
        header_prefix = f"{header}\n\n" if header else ""

        for line in content_lines:
            test_content = '\n'.join(current_lines + [line, last_line])
            if len(header_prefix + test_content) > self.chunk_size:
                if len(current_lines) > 1:
                    current_lines.append(last_line)
                    chunks.append(((header_prefix + '\n'.join(current_lines)).strip(), header))
                    current_lines = [first_line + " (续)", line]
                else:
                    current_lines.append(line)
            else:
                current_lines.append(line)

        if len(current_lines) > 1:
            current_lines.append(last_line)
            chunks.append(((header_prefix + '\n'.join(current_lines)).strip(), header))

        return chunks


# ============================================================
# BM25 关键词检索器
# ============================================================
class BM25Retriever:
    """BM25关键词检索实现"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        require_dependency(rjieba, "rjieba", "进行 BM25 分词")
        self.k1 = k1
        self.b = b
        self.corpus_tokens: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.doc_lens: List[int] = []
        self.avgdl: float = 0
        self.N: int = 0
        self.stopwords = set([
            '的', '是', '在', '了', '和', '与', '或', '等', '这', '那', '有',
            '个', '为', '中', '上', '下', '到', '从', '被', '把', '让', '给',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'and',
            'or', 'but', 'if', 'because', 'while', 'although', 'this',
            'that', 'these', 'those', 'it', 'its'
        ])

    def tokenize(self, text: str) -> List[str]:
        text = (text or "").lower()
        tokens = []

        words = re.findall(r'[a-zA-Z][a-zA-Z0-9_\-]*', text)
        tokens.extend([w for w in words if w not in self.stopwords and len(w) > 1])

        chinese_segments = re.findall(r'[\u4e00-\u9fff]+', text)
        for seg in chinese_segments:
            if len(seg) > 1 and seg not in self.stopwords:
                tokens.append(seg)

            for word in rjieba.cut(seg):
                word = str(word).strip().lower()
                if len(word) > 1 and word not in self.stopwords:
                    tokens.append(word)

            # 保留单字召回能力，减少替换分词器对原有检索行为的影响。
            for char in seg:
                if char not in self.stopwords:
                    tokens.append(char)

        return tokens

    def fit(self, documents: List[str]):
        self.corpus_tokens = []
        self.doc_freqs = []
        self.doc_lens = []
        df_counter = Counter()

        for doc in documents:
            tokens = self.tokenize(doc)
            self.corpus_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            freq = Counter(tokens)
            self.doc_freqs.append(freq)

            for token in set(tokens):
                df_counter[token] += 1

        self.N = len(documents)
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0
        self.idf = {}

        for token, freq in df_counter.items():
            self.idf[token] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

        if DEBUG:
            print(f"  BM25索引: {self.N} 文档, {len(self.idf)} 词汇")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.corpus_tokens:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = []
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_lens[i]

            for token in query_tokens:
                if token not in self.idf:
                    continue

                tf = doc_freq.get(token, 0)
                if tf == 0:
                    continue

                idf = self.idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                score += idf * numerator / denominator

            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_all_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return scores

        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_lens[i]

            for token in query_tokens:
                if token not in self.idf:
                    continue

                tf = doc_freq.get(token, 0)
                if tf == 0:
                    continue

                idf = self.idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                score += idf * numerator / denominator

            scores[i] = score

        return scores

# ============================================================
# 文档加载器（增强版）
# ============================================================
class DocumentLoader:
    SUPPORTED = {'.txt', '.md', '.markdown', '.json', '.py', '.html', '.csv', '.xml', '.log', '.rst'}
    ENCODINGS = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin-1']

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_chars: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self.min_chunk_chars = max(20, min_chunk_chars)
        self.max_merge_size = int(chunk_size * 1.35)
        self.md_chunker = MarkdownChunker(chunk_size, chunk_overlap, min_chunk_chars=min_chunk_chars)

    def normalize_path(self, path_str: str) -> Path:
        path_str = str(path_str).replace("\\", "/")
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    def read_file(self, path: Path) -> Optional[str]:
        for enc in self.ENCODINGS:
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
            except Exception:
                return None
        return None

    def load_file(self, file_path: str) -> Optional[Document]:
        path = self.normalize_path(file_path)
        if not path.exists():
            if DEBUG:
                print(f"  [跳过] 不存在: {path}")
            return None

        if not path.is_file():
            return None

        raw_content = self.read_file(path)
        if raw_content is None:
            print(f"  [跳过] 读取失败: {path.name}")
            return None

        raw_content = raw_content.lstrip("\ufeff")
        filetype = path.suffix.lower()
        is_markdown_like = filetype in ['.md', '.markdown'] or self._looks_like_markdown(raw_content)

        yaml_meta = {}
        content = raw_content

        if is_markdown_like:
            yaml_meta, content_wo_front_matter, _yaml_raw = extract_markdown_front_matter(raw_content)
            if yaml_meta:
                content = content_wo_front_matter

        content = content.strip()
        if not content:
            print(f"  [跳过] 空文件: {path.name}")
            return None

        extra = f", YAML属性 {len(yaml_meta)} 项" if yaml_meta else ""
        print(f"  [加载] {path.name} ({len(content)} 字符{extra})")

        metadata = {
            "source": str(path),
            "filename": path.name,
            "filetype": filetype
        }
        if yaml_meta:
            metadata["front_matter"] = yaml_meta

        return Document(
            content=content,
            metadata=metadata
        )

    def load_directory(self, dir_path: str) -> List[Document]:
        path = self.normalize_path(dir_path)
        if not path.exists() or not path.is_dir():
            print(f"  [跳过] 无效目录: {path}")
            return []

        docs = []
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.suffix.lower() in self.SUPPORTED:
                doc = self.load_file(str(f))
                if doc:
                    docs.append(doc)

        return docs

    def load(self, sources: List[str]) -> List[Document]:
        docs = []
        for src in sources:
            path = self.normalize_path(src)
            if path.is_file():
                doc = self.load_file(str(path))
                if doc:
                    docs.append(doc)
            elif path.is_dir():
                docs.extend(self.load_directory(str(path)))
        return docs

    def split(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []

        for doc in documents:
            doc_id = doc.metadata.get("filename", "unknown")
            filetype = doc.metadata.get("filetype", "")
            text = doc.content

            if not text or not text.strip():
                print(f"  [跳过] 空内容: {doc_id}")
                continue

            print(f"  分块: {doc_id} ", end="", flush=True)
            is_markdown_like = filetype in ['.md', '.markdown'] or self._looks_like_markdown(text)

            if is_markdown_like:
                chunk_tuples = self.md_chunker.split(text, doc.metadata)
                chunks = []
                for i, (content, header_ctx) in enumerate(chunk_tuples):
                    chunks.append(Chunk(
                        content=content,
                        metadata=doc.metadata.copy(),
                        doc_id=doc_id,
                        chunk_id=i,
                        header_context=header_ctx
                    ))
            else:
                chunks = self._generic_split(text, doc_id, doc.metadata)

            raw_count = len(chunks)
            chunks = self._post_process_doc_chunks(chunks)
            all_chunks.extend(chunks)

            if raw_count != len(chunks):
                print(f"-> {len(chunks)} 块 (清洗前 {raw_count} 块)")
            else:
                print(f"-> {len(chunks)} 块")

        return all_chunks

    def _generic_split(self, text: str, doc_id: str, metadata: Dict) -> List[Chunk]:
        text = text.strip()
        if not text:
            return []

        chunks = []
        text_len = len(text)
        pos = 0
        chunk_id = 0

        while pos < text_len:
            end = min(pos + self.chunk_size, text_len)

            if end < text_len:
                search_start = pos + self.chunk_size // 2
                best_pos = -1
                best_sep_len = 1

                for sep in ['。', '！', '？', '\n\n', '\n', '.', '!', '?', ';', '；']:
                    idx = text.rfind(sep, search_start, end)
                    if idx > best_pos:
                        best_pos = idx
                        best_sep_len = len(sep)

                if best_pos > pos:
                    end = best_pos + best_sep_len

            chunk_text = text[pos:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata=metadata.copy(),
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    header_context=""
                ))
                chunk_id += 1

            if end >= text_len:
                break

            next_pos = end - self.chunk_overlap
            if next_pos <= pos:
                next_pos = pos + 1
            pos = next_pos

        return chunks

    def _looks_like_markdown(self, text: str) -> bool:
        if not text:
            return False

        return bool(
            re.search(r'^\s{0,3}#{1,6}\s+\S+', text, re.MULTILINE) or
            re.search(r'!\[[^\]]*\]\([^)]+\)', text) or
            re.search(r'!\[[^\]]*\]\[[^\]]+\]', text) or
            re.search(r'^\s*\[[^\]]+\]:\s+\S+', text, re.MULTILINE) or
            re.search(r'<img\b[^>]*>', text, re.IGNORECASE)
        )

    def _strip_header_prefix(self, content: str, header: str) -> str:
        text = (content or "").strip()
        h = (header or "").strip()
        if h and text.startswith(h):
            return text[len(h):].strip()
        return text

    def _chunk_body(self, chunk: Chunk) -> str:
        return self._strip_header_prefix(chunk.content, chunk.header_context)

    def _post_process_doc_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        merged = []

        for c in chunks:
            body = self._chunk_body(c)
            if not body:
                continue

            if self.md_chunker._is_semantic_empty_text(body):
                continue

            if merged:
                prev = merged[-1]
                prev_body = self._chunk_body(prev)
                same_doc = prev.doc_id == c.doc_id
                same_header = prev.header_context == c.header_context
                short_prev = len(prev_body) < self.min_chunk_chars
                short_curr = len(body) < self.min_chunk_chars
                merged_body = (prev_body.rstrip() + "\n\n" + body.lstrip()).strip()
                merged_text = f"{prev.header_context}\n\n{merged_body}".strip() if prev.header_context else merged_body

                if same_doc and same_header and (short_prev or short_curr) and len(merged_text) <= self.max_merge_size:
                    merged[-1] = Chunk(
                        content=merged_text,
                        metadata=prev.metadata.copy(),
                        doc_id=prev.doc_id,
                        chunk_id=prev.chunk_id,
                        header_context=prev.header_context
                    )
                    continue

            rebuilt = Chunk(
                content=(f"{c.header_context}\n\n{body}".strip() if c.header_context else body),
                metadata=c.metadata.copy(),
                doc_id=c.doc_id,
                chunk_id=c.chunk_id,
                header_context=c.header_context
            )
            merged.append(rebuilt)

        for i, c in enumerate(merged):
            c.chunk_id = i

        return merged


# ============================================================
# Embedding 客户端
# ============================================================
class EmbeddingClient:
    def __init__(self):
        require_embedding_runtime_dependencies()
        self.url = EMBEDDING_URL
        self.model = EMBEDDING_MODEL
        self.dimension = EMBEDDING_DIM
        self.timeout = EMBEDDING_TIMEOUT
        self.headers = {"Content-Type": "application/json"}
        if EMBEDDING_KEY:
            self.headers["Authorization"] = EMBEDDING_KEY

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        if not texts:
            return None

        try:
            texts = [t if isinstance(t, str) else str(t) for t in texts]
            payload = {
                "model": self.model,
                "input": texts
            }
            if EMBEDDING_DIMENSIONS:
                payload["dimensions"] = int(EMBEDDING_DIMENSIONS)

            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            embeddings = self._parse_embedding_response(data)
            if len(embeddings) != len(texts):
                raise RuntimeError(f"Embedding API 返回数量不匹配: {len(embeddings)} != {len(texts)}")

            arr = np.asarray(embeddings, dtype=np.float32)
            if arr.ndim != 2:
                raise RuntimeError(f"Embedding API 返回维度异常: {arr.shape}")
            self.dimension = int(arr.shape[1])
            return arr
        except Exception as e:
            print(f"Embedding 错误: {e}")
            return None

    @staticmethod
    def _parse_embedding_response(data: Dict) -> List[List[float]]:
        if isinstance(data.get("data"), list):
            rows = sorted(data["data"], key=lambda item: item.get("index", 0))
            embeddings = [item.get("embedding") for item in rows]
        elif isinstance(data.get("embeddings"), list):
            embeddings = data["embeddings"]
        elif isinstance(data.get("embedding"), list):
            embeddings = [data["embedding"]]
        else:
            raise RuntimeError(f"未知 Embedding API 响应格式: {list(data.keys())}")

        if not all(isinstance(item, list) for item in embeddings):
            raise RuntimeError("Embedding API 响应缺少 embedding 数组")
        return embeddings


# ============================================================
# LLM 客户端
# ============================================================
class LLMClient:
    def __init__(self):
        self.url = LLM_URL
        self.headers = {"Content-Type": "application/json"}
        if LLM_KEY:
            self.headers["Authorization"] = LLM_KEY

    def chat(self, messages: List[Dict], max_tokens: int = 2048, temperature: float = 0.7) -> str:
        if not self.url:
            return "LLM 错误: LLM_URL 未配置"

        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            elif "response" in data:
                return data["response"]
            else:
                return f"未知响应格式: {list(data.keys())}"
        except Exception as e:
            return f"LLM 错误: {e}"


# ============================================================
# 混合检索向量存储
# ============================================================
class HybridVectorStore:
    """混合检索向量存储：numpy 向量矩阵 + BM25 + 元数据/文件名增强"""

    def __init__(self, embedding_client: EmbeddingClient,
                 vector_weight: float = 0.6,
                 bm25_weight: float = 0.4):
        self.client = embedding_client
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.bm25 = BM25Retriever()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        self.index_texts: List[str] = []
        self.metadata_texts: List[str] = []
        self.metadata_token_sets: List[set] = []
        self.doc_to_indices = defaultdict(list)
        self.doc_aliases = {}
        self.vector_dimension: int = self.client.dimension

        self.filename_match_boost = FILENAME_MATCH_BOOST
        self.metadata_match_boost = METADATA_MATCH_BOOST
        self.preferred_doc_max_ratio = PREFERRED_DOC_MAX_RATIO

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (vectors / norms).astype(np.float32)

    def _build_chunk_metadata_text(self, chunk: Chunk, max_len: int = MAX_INDEX_METADATA_CHARS) -> str:
        parts = []

        filename = chunk.metadata.get("filename", chunk.doc_id)
        if filename:
            parts.append(f"文件名: {filename}")

        filetype = chunk.metadata.get("filetype", "")
        if filetype:
            parts.append(f"文件类型: {filetype}")

        front_text = format_chunk_attrs(chunk.metadata, max_items=20, max_len=max_len)
        if front_text:
            parts.append(f"属性: {front_text}")

        return "\n".join(parts).strip()

    def _build_index_text(self, chunk: Chunk) -> str:
        parts = []

        meta_text = self._build_chunk_metadata_text(chunk)
        if meta_text:
            parts.append(meta_text)

        if chunk.header_context:
            parts.append(f"章节:\n{chunk.header_context}")

        parts.append(f"正文:\n{chunk.content}")
        return "\n".join(parts).strip()

    def _is_meaningful_alias(self, alias: str) -> bool:
        if not alias:
            return False
        if re.search(r'[\u4e00-\u9fff]', alias):
            return len(alias) >= 2
        return len(alias) >= 3

    def _make_doc_aliases(self, filename: str) -> set:
        aliases = set()
        if not filename:
            return aliases

        stem = Path(filename).stem
        for item in [filename, stem]:
            if not item:
                continue
            item_lower = item.lower().strip()
            item_norm = normalize_text(item)
            if self._is_meaningful_alias(item_lower):
                aliases.add(item_lower)
            if self._is_meaningful_alias(item_norm):
                aliases.add(item_norm)
        return aliases

    def _query_mentions_alias(self, query_lower: str, query_norm: str, alias: str) -> bool:
        if not alias:
            return False

        if re.search(r'[\u4e00-\u9fff]', alias):
            alias_norm = normalize_text(alias)
            return alias in query_lower or (alias_norm and alias_norm in query_norm)

        if re.fullmatch(r'[a-z0-9._\-]+', alias):
            if re.search(rf'(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])', query_lower):
                return True
        elif alias in query_lower:
            return True

        alias_norm = normalize_text(alias)
        return bool(alias_norm and alias_norm in query_norm)

    def _rebuild_aux_cache(self):
        self.index_texts = []
        self.metadata_texts = []
        self.metadata_token_sets = []
        self.doc_to_indices = defaultdict(list)
        self.doc_aliases = {}

        for i, chunk in enumerate(self.chunks):
            index_text = self._build_index_text(chunk)
            metadata_text = self._build_chunk_metadata_text(chunk)

            self.index_texts.append(index_text)
            self.metadata_texts.append(metadata_text)
            self.metadata_token_sets.append(set(self.bm25.tokenize(metadata_text)) if metadata_text else set())

            filename = chunk.metadata.get("filename", chunk.doc_id)
            self.doc_to_indices[filename].append(i)

        for filename in self.doc_to_indices.keys():
            self.doc_aliases[filename] = self._make_doc_aliases(filename)

    def _detect_preferred_docs(self, query: str) -> set:
        query_lower = query.lower()
        query_norm = normalize_text(query)
        preferred_docs = set()

        for filename, aliases in self.doc_aliases.items():
            for alias in aliases:
                if self._query_mentions_alias(query_lower, query_norm, alias):
                    preferred_docs.add(filename)
                    break

        return preferred_docs

    def _compute_metadata_match_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        if not self.chunks:
            return scores

        query_tokens = set(self.bm25.tokenize(query))
        query_norm = normalize_text(query)

        for i, meta_tokens in enumerate(self.metadata_token_sets):
            score = 0.0

            if query_tokens and meta_tokens:
                overlap = len(query_tokens & meta_tokens) / max(1, len(query_tokens))
                score = max(score, overlap)

            meta_text = self.metadata_texts[i]
            if query_norm and meta_text:
                meta_norm = normalize_text(meta_text)
                if meta_norm and query_norm in meta_norm:
                    score = max(score, 1.0)

            scores[i] = score

        return scores

    def add(self, chunks: List[Chunk]):
        if not chunks:
            print("没有文档块可添加")
            return

        print(f"\n生成向量 ({len(chunks)} 个文本)...")

        batch_size = EMBEDDING_BATCH_SIZE
        pending_vectors = []
        valid_chunks = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [self._build_index_text(c) for c in batch_chunks]
            vectors = self.client.embed(batch_texts)

            if vectors is None or len(vectors) != len(batch_chunks):
                print(f"  批次 {i // batch_size + 1} 失败")
                continue

            pending_vectors.append(vectors)
            valid_chunks.extend(batch_chunks)
            print(f"  已处理 {min(i + batch_size, len(chunks))}/{len(chunks)}")

        if not pending_vectors:
            print("向量生成失败！")
            return

        vectors = self._normalize_vectors(np.vstack(pending_vectors))

        self.chunks.extend(valid_chunks)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.vector_dimension = int(vectors.shape[1])

        self._rebuild_aux_cache()

        print("构建BM25索引...")
        self.bm25.fit(self.index_texts)

        print(f"向量库大小: {len(self.chunks)} 个文档块")
        if DEBUG:
            print(f"向量矩阵: {self.vectors.shape}")

    def search(self, query: str, top_k: int = 5, verbose: bool = True) -> List[SearchResult]:
        if self.vectors is None or len(self.chunks) == 0:
            if verbose:
                print("[搜索] 向量库为空！")
            return []

        vector_scores = np.zeros(len(self.chunks), dtype=np.float32)
        q_vec = self.client.embed([query])
        if q_vec is not None and len(q_vec) > 0:
            q_vec = self._normalize_vectors(q_vec)[0]
            vector_scores = np.dot(self.vectors, q_vec)

        bm25_scores = self.bm25.get_all_scores(query)
        metadata_scores = self._compute_metadata_match_scores(query)
        preferred_docs = self._detect_preferred_docs(query)

        filename_boost_scores = np.zeros(len(self.chunks), dtype=np.float32)
        if preferred_docs:
            for doc_name in preferred_docs:
                for idx in self.doc_to_indices.get(doc_name, []):
                    filename_boost_scores[idx] = 1.0

        if vector_scores.max() > vector_scores.min():
            vector_scores_norm = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min())
        else:
            vector_scores_norm = vector_scores

        if bm25_scores.max() > bm25_scores.min():
            bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_scores_norm = bm25_scores

        if metadata_scores.max() > metadata_scores.min():
            metadata_scores_norm = (metadata_scores - metadata_scores.min()) / (metadata_scores.max() - metadata_scores.min())
        else:
            metadata_scores_norm = metadata_scores

        combined_scores = (
            self.vector_weight * vector_scores_norm +
            self.bm25_weight * bm25_scores_norm +
            self.metadata_match_boost * metadata_scores_norm +
            self.filename_match_boost * filename_boost_scores
        )

        if DEBUG and verbose:
            print(f"[搜索] 向量分数范围: {vector_scores.min():.4f} - {vector_scores.max():.4f}")
            print(f"[搜索] BM25分数范围: {bm25_scores.min():.4f} - {bm25_scores.max():.4f}")
            print(f"[搜索] 元数据分数范围: {metadata_scores.min():.4f} - {metadata_scores.max():.4f}")
            if preferred_docs:
                print(f"[搜索] 命中文件名优先: {sorted(preferred_docs)}")
            print(f"[搜索] 融合分数范围: {combined_scores.min():.4f} - {combined_scores.max():.4f}")

        ranked_indices = list(np.argsort(combined_scores)[::-1])

        # 文件优先但不锁死召回
        selected = []
        selected_set = set()

        if preferred_docs and top_k > 1:
            preferred_quota = int(math.ceil(top_k * self.preferred_doc_max_ratio))
            preferred_quota = max(1, min(preferred_quota, top_k - 1))

            for idx in ranked_indices:
                if len(selected) >= preferred_quota:
                    break
                filename = self.chunks[idx].metadata.get("filename", self.chunks[idx].doc_id)
                if filename in preferred_docs:
                    selected.append(idx)
                    selected_set.add(idx)

            for idx in ranked_indices:
                if len(selected) >= top_k:
                    break
                if idx in selected_set:
                    continue
                filename = self.chunks[idx].metadata.get("filename", self.chunks[idx].doc_id)
                if filename not in preferred_docs:
                    selected.append(idx)
                    selected_set.add(idx)

            for idx in ranked_indices:
                if len(selected) >= top_k:
                    break
                if idx in selected_set:
                    continue
                selected.append(idx)
                selected_set.add(idx)

            top_indices = selected[:top_k]
        else:
            top_indices = ranked_indices[:top_k]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(combined_scores[idx]),
                vector_score=float(vector_scores[idx]),
                bm25_score=float(bm25_scores[idx]),
                metadata_score=float(metadata_scores[idx]),
                filename_boost=float(filename_boost_scores[idx])
            ))

        if DEBUG and verbose:
            print(f"[搜索] 返回 {len(results)} 个结果:")
            for i, r in enumerate(results, 1):
                attrs = format_chunk_attrs(r.chunk.metadata, max_items=6, max_len=120)
                print(
                    f"  {i}. [综合:{r.score:.3f} V:{r.vector_score:.3f} "
                    f"B:{r.bm25_score:.3f} M:{r.metadata_score:.3f} F:{r.filename_boost:.1f}] "
                    f"{r.chunk.doc_id}"
                )
                if attrs:
                    print(f"     属性: {attrs}")

        return results

    def clear(self, reset_vector_store: bool = True):
        self.chunks = []
        self.vectors = None
        self.bm25 = BM25Retriever()
        self.index_texts = []
        self.metadata_texts = []
        self.metadata_token_sets = []
        self.doc_to_indices = defaultdict(list)
        self.doc_aliases = {}
        self.vector_dimension = self.client.dimension

    def save(self, save_dir: str):
        if self.vectors is None or not self.chunks:
            raise RuntimeError("当前索引为空，无法保存。")

        save_path = Path(save_dir).resolve()
        save_path.mkdir(parents=True, exist_ok=True)

        vectors_path = save_path / "vectors.npy"
        chunks_path = save_path / "chunks.jsonl"
        meta_path = save_path / "meta.json"

        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in self.chunks:
                row = {
                    "content": c.content,
                    "metadata": c.metadata,
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "header_context": c.header_context
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        np.save(vectors_path, self.vectors.astype(np.float32), allow_pickle=False)

        meta = {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": int(self.vectors.shape[1]),
            "chunk_count": len(self.chunks),
            "vector_store": "numpy",
            "vectors_file": "vectors.npy",
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "index_text_mode": "content+header+metadata",
            "filename_match_boost": self.filename_match_boost,
            "metadata_match_boost": self.metadata_match_boost,
            "preferred_doc_max_ratio": self.preferred_doc_max_ratio
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"\n索引已保存到: {save_path}")

    def load(self, save_dir: str):
        save_path = Path(save_dir).resolve()
        chunks_path = save_path / "chunks.jsonl"
        meta_path = save_path / "meta.json"
        vectors_path = save_path / "vectors.npy"

        if not save_path.exists():
            raise FileNotFoundError(f"索引目录不存在: {save_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"缺少文件: {chunks_path}")
        if not vectors_path.exists():
            raise FileNotFoundError(f"缺少文件: {vectors_path}")

        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunks.append(Chunk(
                    content=obj["content"],
                    metadata=obj["metadata"],
                    doc_id=obj["doc_id"],
                    chunk_id=obj["chunk_id"],
                    header_context=obj.get("header_context", "")
                ))

        self.clear(reset_vector_store=False)
        self.chunks = chunks
        self.vectors = np.load(vectors_path, allow_pickle=False).astype(np.float32)
        if len(chunks) != len(self.vectors):
            raise RuntimeError("索引损坏：chunks 数量与 vectors 数量不一致")
        self._rebuild_aux_cache()

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.vector_weight = float(meta.get("vector_weight", self.vector_weight))
            self.bm25_weight = float(meta.get("bm25_weight", self.bm25_weight))
            self.filename_match_boost = float(meta.get("filename_match_boost", self.filename_match_boost))
            self.metadata_match_boost = float(meta.get("metadata_match_boost", self.metadata_match_boost))
            self.preferred_doc_max_ratio = float(meta.get("preferred_doc_max_ratio", self.preferred_doc_max_ratio))

            if meta.get("index_text_mode") != "content+header+metadata":
                print("⚠️ 当前加载的是旧版索引：向量部分可能尚未纳入元数据，建议重新建索引后再保存。")
            self.vector_dimension = int(meta.get("embedding_dim", self.vector_dimension))

        print("重建BM25索引...")
        self.bm25.fit(self.index_texts)

        print(f"\n索引加载完成: {save_path}")
        print(f"向量库大小: {len(self.chunks)} 个文档块")
        if DEBUG:
            print(f"向量矩阵: {self.vectors.shape}")

    def __len__(self):
        return len(self.chunks)


# ============================================================
# RAG 引擎
# ============================================================
class RAG:
    def __init__(self):
        print("\n" + "=" * 60)
        print("初始化 RAG 系统（增强版）")
        print("=" * 60)
        require_rag_runtime_dependencies()

        self.loader = DocumentLoader(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            min_chunk_chars=MIN_CHUNK_CHARS
        )
        self.embedding = EmbeddingClient()
        self.llm = LLMClient()
        self.store = HybridVectorStore(
            self.embedding,
            vector_weight=VECTOR_WEIGHT,
            bm25_weight=BM25_WEIGHT
        )
        self.pipeline = self._build_pipeline()

        print("配置:")
        print(f"  Embedding API: {EMBEDDING_URL}")
        print(f"  Embedding Model: {EMBEDDING_MODEL}")
        print(f"  LLM: {LLM_URL}")
        print(f"  Chunk Size: {CHUNK_SIZE}")
        print(f"  Min Chunk Chars: {MIN_CHUNK_CHARS}")
        print(f"  Top K: {TOP_K}")
        print(f"  混合检索: 向量权重={VECTOR_WEIGHT}, BM25权重={BM25_WEIGHT}")
        print(f"  文件名命中Boost: {FILENAME_MATCH_BOOST}")
        print(f"  元数据Boost: {METADATA_MATCH_BOOST}")
        print(f"  文件优先结果占比上限: {PREFERRED_DOC_MAX_RATIO}")
        print(f"  RAG Pipeline: LangGraph")
        print(f"  Debug: {DEBUG}")
        print("初始化完成!\n")

    def _build_pipeline(self):
        require_dependency(StateGraph, "langgraph", "初始化 RAG Pipeline")
        graph = StateGraph(RAGPipelineState)
        graph.add_node("retrieve", self._pipeline_retrieve)
        graph.add_node("build_context", self._pipeline_build_context)
        graph.add_node("generate_answer", self._pipeline_generate_answer)
        graph.add_node("append_metadata_summary", self._pipeline_append_metadata_summary)
        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self._pipeline_after_retrieve,
            {
                "continue": "build_context",
                "end": END
            }
        )
        graph.add_edge("build_context", "generate_answer")
        graph.add_edge("generate_answer", "append_metadata_summary")
        graph.add_edge("append_metadata_summary", END)
        return graph.compile()

    def _pipeline_after_retrieve(self, state: RAGPipelineState) -> str:
        return "end" if state.get("error") else "continue"

    def _pipeline_retrieve(self, state: RAGPipelineState) -> RAGPipelineState:
        question = state["question"]
        top_k = state.get("top_k") or TOP_K
        verbose = state.get("verbose", True)

        if verbose:
            print("\n[步骤1] 混合检索...")

        results = self.store.search(question, top_k=top_k, verbose=verbose)
        if not results:
            return {
                **state,
                "results": [],
                "answer": "检索失败，没有找到相关内容。",
                "error": "empty_retrieval"
            }

        if verbose:
            self._print_retrieved_results(results)

        return {**state, "results": results}

    def _print_retrieved_results(self, results: List[SearchResult]):
        print(f"\n检索到 {len(results)} 个相关片段:")
        for i, r in enumerate(results, 1):
            print(
                f"\n  [{i}] 综合分数: {r.score:.4f} "
                f"(向量: {r.vector_score:.4f}, BM25: {r.bm25_score:.4f}, "
                f"元数据: {r.metadata_score:.4f}, 文件Boost: {r.filename_boost:.1f})"
            )
            print(f"      来源: {r.chunk.metadata.get('filename', '未知')}")
            if r.chunk.header_context:
                print(f"      章节: {r.chunk.header_context.split(chr(10))[0][:50]}...")
            attrs = format_chunk_attrs(r.chunk.metadata, max_items=8, max_len=220)
            if attrs:
                print(f"      属性: {attrs}")
            print(f"      内容: {r.chunk.content[:100]}...")

    def _build_context_from_results(self, results: List[SearchResult]) -> str:
        context_parts = []
        for i, r in enumerate(results, 1):
            source = r.chunk.metadata.get("filename", "未知")
            header = r.chunk.header_context
            attrs = format_chunk_attrs(r.chunk.metadata, max_items=20, max_len=400)
            front_json = front_matter_json_text(r.chunk.metadata, max_len=1200)

            context_entry = f"[文档{i}] (来源: {source}, 相关度: {r.score:.2f})"
            if header:
                context_entry += f"\n章节: {header}"
            if attrs:
                context_entry += f"\n属性摘要: {attrs}"
            if front_json:
                context_entry += f"\n前置元数据(JSON):\n{front_json}"
            context_entry += f"\n内容:\n{r.chunk.content}"
            context_parts.append(context_entry)

        separator = "\n\n" + "=" * 50 + "\n\n"
        return separator.join(context_parts)

    def _build_answer_messages(self, question: str, context: str) -> List[Dict]:
        return [
            {
                "role": "system",
                "content": """你是一个专业的AI助手。请根据提供的上下文信息回答用户问题。
    规则：
    1. 只能基于上下文回答，不要编造
    2. 优先参考相关度高的文档
    3. 如果多个文档都相关，请综合总结
    4. 如果上下文中有章节信息，可以引用章节
    5. 如果上下文中包含 front matter / 属性 / 元数据，并且与问题相关，必须明确写出字段和值
    6. 如果问题询问负责人、模块、优先级、标签、状态、版本、适用范围等属性，优先依据元数据回答
    7. 如果正文和元数据都相关，两者都要体现，不要只回答正文
    8. 不要忽略上下文里的“前置元数据(JSON)”部分
    9. 如果确实没有相关信息，请回答：根据提供的文档，没有找到相关信息

    请尽量使用如下结构：
    答案：
    依据：
    相关属性："""
            },
            {
                "role": "user",
                "content": f"""请根据以下上下文回答问题。

    上下文信息：
    {context}

    问题：{question}

    请回答："""
            }
        ]

    def _pipeline_build_context(self, state: RAGPipelineState) -> RAGPipelineState:
        results = state.get("results", [])
        context = self._build_context_from_results(results)
        verbose = state.get("verbose", True)

        if DEBUG and verbose:
            print(f"\n[调试] 上下文长度: {len(context)} 字符")

        messages = self._build_answer_messages(state["question"], context)
        return {**state, "context": context, "messages": messages}

    def _pipeline_generate_answer(self, state: RAGPipelineState) -> RAGPipelineState:
        verbose = state.get("verbose", True)
        if verbose:
            print("\n[步骤2] 调用 LLM 生成回答...")
        answer = self.llm.chat(state["messages"])
        return {**state, "answer": answer}

    def _pipeline_append_metadata_summary(self, state: RAGPipelineState) -> RAGPipelineState:
        answer = state.get("answer", "")
        results = state.get("results", [])
        fm_summary = build_result_front_matter_summary(results, max_docs=min(3, len(results)))
        if fm_summary:
            answer = answer.rstrip()
            answer += "\n\n【命中文档属性】\n" + fm_summary
        return {**state, "answer": answer}

    def load_knowledge(self, sources: List[str]) -> int:
        if not sources:
            print("没有指定知识源")
            return 0

        print("=" * 40)
        print("加载文档...")
        print("=" * 40)

        docs = self.loader.load(sources)
        if not docs:
            print("没有找到有效文档！")
            return 0

        print(f"\n成功加载 {len(docs)} 个文档")
        type_counts = Counter(d.metadata.get('filetype', 'unknown') for d in docs)
        print("文件类型分布:")
        for ft, count in type_counts.items():
            print(f"  {ft}: {count}")

        print("\n分割文档（使用Markdown感知分块）...")
        chunks = self.loader.split(docs)
        if not chunks:
            print("分块失败！")
            return 0

        print(f"生成 {len(chunks)} 个文档块")
        self.store.add(chunks)
        return len(chunks)

    def save_index(self, save_dir: str):
        self.store.save(save_dir)

    def load_index(self, save_dir: str):
        self.store.load(save_dir)

    def retrieve(self, question: str, top_k: int = None, verbose: bool = True) -> List[SearchResult]:
        top_k = top_k or TOP_K
        return self.store.search(question, top_k=top_k, verbose=verbose)

    def query(self, question: str, top_k: int = None, verbose: bool = True) -> Tuple[str, List[SearchResult]]:
        if verbose:
            print("\n" + "-" * 40)
            print(f"问题: {question}")
            print("-" * 40)

        if len(self.store) == 0:
            return "知识库为空，请先加载文档。", []

        top_k = top_k or TOP_K

        state = self.pipeline.invoke({
            "question": question,
            "top_k": top_k,
            "verbose": verbose
        })
        return state.get("answer", ""), state.get("results", [])

    def interactive(self):
        print("\n" + "=" * 60)
        print("交互问答模式（增强版）")
        print("=" * 60)
        print("命令:")
        print("  直接输入问题进行问答")
        print("  /add <路径>        - 添加文件或目录")
        print("  /list              - 列出已加载的文档")
        print("  /search <词>       - 仅搜索不回答")
        print("  /save_index <目录> - 保存当前索引")
        print("  /load_index <目录> - 加载已有索引")
        print("  /weights <v> <b>   - 调整检索权重")
        print("  /debug             - 切换调试模式")
        print("  /stats             - 显示统计信息")
        print("  /quit              - 退出")
        print("=" * 60 + "\n")

        global DEBUG

        while True:
            try:
                user_input = input("\n🧑 问题: ").strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""

                    if cmd in ["/quit", "/exit", "/q"]:
                        print("再见！")
                        break

                    elif cmd == "/add":
                        if arg:
                            self.load_knowledge([arg])
                        else:
                            print("用法: /add <文件或目录路径>")

                    elif cmd == "/list":
                        print(f"\n已加载 {len(self.store)} 个文档块:")
                        doc_names = set(c.doc_id for c in self.store.chunks)
                        for name in sorted(doc_names):
                            count = sum(1 for c in self.store.chunks if c.doc_id == name)
                            print(f"  - {name} ({count} 块)")

                    elif cmd == "/search":
                        if arg:
                            results = self.store.search(arg, TOP_K, verbose=True)
                            print(f"\n搜索 '{arg}' 的结果:")
                            for i, r in enumerate(results, 1):
                                print(f"\n{i}. [综合:{r.score:.3f} V:{r.vector_score:.3f} B:{r.bm25_score:.3f} M:{r.metadata_score:.3f} F:{r.filename_boost:.1f}]")
                                print(f"   来源: {r.chunk.doc_id}")
                                attrs = format_chunk_attrs(r.chunk.metadata, max_items=6, max_len=180)
                                if attrs:
                                    print(f"   属性: {attrs}")
                                print(f"   内容: {r.chunk.content[:150]}...")
                        else:
                            print("用法: /search <关键词>")

                    elif cmd == "/save_index":
                        if arg:
                            self.save_index(arg)
                        else:
                            print("用法: /save_index <目录>")

                    elif cmd == "/load_index":
                        if arg:
                            self.load_index(arg)
                        else:
                            print("用法: /load_index <目录>")

                    elif cmd == "/weights":
                        try:
                            sub_parts = arg.split()
                            if len(sub_parts) == 2:
                                v_weight = float(sub_parts[0])
                                b_weight = float(sub_parts[1])
                                self.store.vector_weight = v_weight
                                self.store.bm25_weight = b_weight
                                print(f"检索权重已更新: 向量={v_weight}, BM25={b_weight}")
                            else:
                                print(f"当前权重: 向量={self.store.vector_weight}, BM25={self.store.bm25_weight}")
                                print("用法: /weights <向量权重> <BM25权重>")
                        except ValueError:
                            print("请输入有效的数字")

                    elif cmd == "/debug":
                        DEBUG = not DEBUG
                        print(f"调试模式: {'开启' if DEBUG else '关闭'}")

                    elif cmd == "/stats":
                        print(f"\n统计信息:")
                        print(f"  文档块数量: {len(self.store)}")
                        print(f"  向量维度: {self.embedding.dimension}")
                        print("  向量存储: numpy")
                        if self.store.vectors is not None:
                            print(f"  向量矩阵: {self.store.vectors.shape}")
                        print(f"  BM25词汇量: {len(self.store.bm25.idf)}")
                        print(f"  检索权重: 向量={self.store.vector_weight}, BM25={self.store.bm25_weight}")
                        print(f"  文件名Boost: {self.store.filename_match_boost}")
                        print(f"  元数据Boost: {self.store.metadata_match_boost}")
                        print(f"  文件优先上限: {self.store.preferred_doc_max_ratio}")

                    else:
                        print(f"未知命令: {cmd}")

                    continue

                answer, sources = self.query(user_input, verbose=True)
                print(f"\n🤖 回答:\n{answer}")

                if sources:
                    print(f"\n📚 参考来源 (共 {len(sources)} 个):")
                    for r in sources[:5]:
                        print(f"  - [{r.score:.2f}] {r.chunk.metadata.get('filename', '未知')}")
                        if r.chunk.header_context:
                            header_line = r.chunk.header_context.split('\n')[0]
                            print(f"    章节: {header_line[:50]}...")
                        attrs = format_chunk_attrs(r.chunk.metadata, max_items=6, max_len=180)
                        if attrs:
                            print(f"    属性: {attrs}")

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()


# ============================================================
# 测试集生成器
# ============================================================
class TestsetGenerator:
    SEMICON_GLOSSARY = {
        "晶圆": ["wafer", "晶圆", "片子"],
        "晶圆厂": ["fab", "fab厂", "晶圆厂"],
        "裸片": ["die", "裸片", "die片"],
        "良率": ["yield", "良率", "yld"],
        "机台": ["tool", "机台", "设备"],
        "流片": ["tape-out", "tapeout", "流片"],
        "光罩": ["mask", "reticle", "光罩", "掩模版"],
        "刻蚀": ["etch", "刻蚀", "蚀刻"],
        "沉积": ["deposition", "沉积", "薄膜沉积"],
        "光刻": ["lithography", "光刻", "曝光"],
        "套刻": ["overlay", "套刻"],
        "关键尺寸": ["cd", "critical dimension", "关键尺寸", "线宽"],
        "在制品": ["wip", "在制品"],
        "化学机械抛光": ["cmp", "化学机械抛光", "抛光"],
        "前段工艺": ["feol", "前段工艺"],
        "后段工艺": ["beol", "后段工艺"],
        "缺陷密度": ["defect density", "缺陷密度", "d0"],
        "重布线层": ["rdl", "重布线层"],
        "凸块": ["bump", "凸块"],
        "封装": ["packaging", "封装", "pkg"]
    }

    def __init__(self, llm_client: LLMClient, store: HybridVectorStore):
        self.llm = llm_client
        self.store = store
        self.refusal_answer = DEFAULT_REFUSAL_ANSWER
        self.tokenizer = BM25Retriever()

        self.doc_chunks = defaultdict(list)
        for c in self.store.chunks:
            doc_key = c.metadata.get("source", c.doc_id)
            self.doc_chunks[doc_key].append(c)

        self.semicon_terms = set()
        for aliases in self.SEMICON_GLOSSARY.values():
            for t in aliases:
                self.semicon_terms.add(t.lower())

    @staticmethod
    def _extract_json(text: str):
        text = text.strip()
        text = re.sub(r"^```json\s*", "", text, flags=re.I)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except Exception:
            pass

        candidates = []
        s1, e1 = text.find("{"), text.rfind("}")
        s2, e2 = text.find("["), text.rfind("]")

        if s1 != -1 and e1 != -1 and e1 > s1:
            candidates.append(text[s1:e1 + 1])
        if s2 != -1 and e2 != -1 and e2 > s2:
            candidates.append(text[s2:e2 + 1])

        for c in sorted(candidates, key=len, reverse=True):
            try:
                return json.loads(c)
            except Exception:
                continue

        raise ValueError(f"LLM 输出不是有效 JSON，原始输出前200字符: {text[:200]}")

    @staticmethod
    def _make_case_id(prefix: str, question: str, keys: List[str]) -> str:
        raw = f"{prefix}::{question}::" + "||".join(sorted(keys))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _is_good_chunk(chunk: Chunk, min_chars: int = 120) -> bool:
        text = chunk.content.strip()
        if len(text) < min_chars:
            return False
        plain = re.sub(r'```[\s\S]*?```', '', text).strip()
        if len(plain) < 40:
            return False
        return True

    @staticmethod
    def _is_disamb_candidate(chunk: Chunk) -> bool:
        text = chunk.content.lower()
        keywords = [
            "区别", "不同", "差异", "对比", "相比", "分别", "选择", "优先", "而不是",
            "vs", "versus", "difference", "compare", "between", "option"
        ]
        if any(k in text for k in keywords):
            return True
        if len(re.findall(r"\n\s*[-*+]\s+", chunk.content)) >= 2:
            return True
        if "|" in chunk.content and "\n|" in chunk.content:
            return True
        return False

    @staticmethod
    def _is_reasoning_candidate(chunk: Chunk) -> bool:
        text = chunk.content.lower()
        keywords = [
            "如果", "当", "则", "条件", "阈值", "超过", "低于", "不少于", "不超过",
            "先", "后", "然后", "before", "after", "if", "when", "threshold",
            "greater than", "less than", "must", "should"
        ]
        return any(k in text for k in keywords) or bool(re.search(r"\d", text))

    def _is_semiconductor_candidate(self, chunk: Chunk) -> bool:
        text = chunk.content.lower()
        return any(term in text for term in self.semicon_terms)

    def _ctx_from_chunk(self, chunk: Chunk) -> Dict:
        return {
            "source_doc": chunk.metadata.get("filename", chunk.doc_id),
            "source_path": chunk.metadata.get("source", ""),
            "chunk_id": chunk.chunk_id,
            "header": chunk.header_context,
            "attrs": format_chunk_attrs(chunk.metadata, max_items=20, max_len=300),
            "text": chunk.content
        }

    def _format_contexts(self, contexts: List[Dict]) -> str:
        parts = []
        for i, ctx in enumerate(contexts, 1):
            part = f"[SOURCE {i}] 文档: {ctx['source_doc']}\n"
            if ctx.get("header"):
                part += f"章节:\n{ctx['header']}\n"
            if ctx.get("attrs"):
                part += f"属性: {ctx['attrs']}\n"
            part += f"内容:\n{ctx['text']}"
            parts.append(part)
        return "\n\n-----\n\n".join(parts)

    def _validate_evidence_list(self, evidence_list: List[Dict], contexts: List[Dict], answer: str) -> Optional[List[int]]:
        used_indices = []

        for item in evidence_list:
            try:
                src_idx = int(item.get("source_index", 1)) - 1
            except Exception:
                return None

            if src_idx < 0 or src_idx >= len(contexts):
                return None

            ev = str(item.get("evidence", "")).strip()
            if not ev:
                return None

            src_text_norm = normalize_text(contexts[src_idx]["text"])
            ev_norm = normalize_text(ev)
            ans_norm = normalize_text(answer)

            if ev_norm not in src_text_norm:
                if ans_norm and ans_norm in src_text_norm:
                    pass
                else:
                    return None

            used_indices.append(src_idx)

        return used_indices

    def _question_has_lexical_gap(self, question: str, source_texts: List[str]) -> bool:
        q_norm = normalize_text(question)
        source_norm = normalize_text("\n".join(source_texts))
        if q_norm and q_norm in source_norm:
            return False

        q_tokens = set(self.tokenizer.tokenize(question))
        s_tokens = set()
        for t in source_texts:
            s_tokens |= set(self.tokenizer.tokenize(t[:3000]))

        if not q_tokens:
            return True

        overlap_ratio = len(q_tokens & s_tokens) / max(1, len(q_tokens))
        return overlap_ratio < 0.85

    def _has_out_of_source_alias(self, alias_terms: List[str], source_texts: List[str]) -> bool:
        src = normalize_text("\n".join(source_texts))
        for term in alias_terms:
            term_n = normalize_text(term)
            if term_n and term_n not in src:
                return True
        return False

    def _judge_advanced_case(self,
                             question: str,
                             answer: str,
                             contexts: List[Dict],
                             require_multi_source: bool = False,
                             require_reasoning: bool = False,
                             require_paraphrase: bool = False) -> bool:
        context_text = self._format_contexts(contexts)

        prompt = f"""
你是 RAG 测试样本审查员，请判断下面的样本是否合格。
要求：
1. answer 必须能够完全根据上下文推出，不能依赖外部知识
2. 如果 require_multi_source=true，则问题必须至少需要综合两个来源，不能单靠其中一个来源完整回答
3. 如果 require_reasoning=true，则问题应需要一定推理/归纳/比较/条件判断，而不是直接摘抄一句话
4. 如果 require_paraphrase=true，则问题表述应与原文存在明显语义改写，不是简单照抄
5. 只输出 JSON
输出格式：
{{
  "valid": true/false,
  "uses_multiple_sources": true/false,
  "requires_reasoning": true/false,
  "is_paraphrased": true/false
}}
require_multi_source={str(require_multi_source).lower()}
require_reasoning={str(require_reasoning).lower()}
require_paraphrase={str(require_paraphrase).lower()}
问题：
{question}
答案：
{answer}
上下文：
{context_text}
        """.strip()

        raw = self.llm.chat(
            [
                {"role": "system", "content": "你是严格的 RAG 样本审查员，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.1
        )

        if raw.startswith("LLM 错误:"):
            return False

        try:
            obj = self._extract_json(raw)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            if not isinstance(obj, dict):
                return False
            if not bool(obj.get("valid", False)):
                return False
            if require_multi_source and not bool(obj.get("uses_multiple_sources", False)):
                return False
            if require_reasoning and not bool(obj.get("requires_reasoning", False)):
                return False
            if require_paraphrase and not bool(obj.get("is_paraphrased", False)):
                return False
            return True
        except Exception:
            return False

    def _build_answer_case(self,
                           obj: Dict,
                           contexts: List[Dict],
                           case_type: str,
                           tags: Optional[List[str]] = None,
                           require_multi_source: bool = False,
                           require_reasoning: bool = False,
                           require_paraphrase: bool = False,
                           require_alias: bool = False) -> Optional[TestCase]:
        if not isinstance(obj, dict) or obj.get("skip"):
            return None

        question = str(obj.get("question", "")).strip()
        answer = str(obj.get("answer", "")).strip()
        difficulty = str(obj.get("difficulty", "medium")).strip() or "medium"

        if not question or not answer or len(question) < 6:
            return None

        evidence_list = obj.get("evidence_list", [])
        if not evidence_list:
            ev = str(obj.get("evidence", "")).strip()
            if ev:
                evidence_list = [{"source_index": 1, "evidence": ev}]
            else:
                return None

        if not isinstance(evidence_list, list) or not evidence_list:
            return None

        used_indices = self._validate_evidence_list(evidence_list, contexts, answer)
        if used_indices is None:
            return None

        unique_indices = []
        seen = set()
        for idx in used_indices:
            key = (contexts[idx]["source_path"], contexts[idx]["chunk_id"])
            if key not in seen:
                seen.add(key)
                unique_indices.append(idx)

        if require_multi_source and len(unique_indices) < 2:
            return None

        source_texts = [ctx["text"] for ctx in contexts]

        if require_paraphrase and not self._question_has_lexical_gap(question, source_texts):
            return None

        if require_alias:
            alias_terms = obj.get("used_alias_terms", [])
            if not isinstance(alias_terms, list) or not alias_terms:
                return None
            alias_terms = [str(x).strip() for x in alias_terms if str(x).strip()]
            if not alias_terms:
                return None
            if not self._has_out_of_source_alias(alias_terms, source_texts):
                return None

        if case_type in {
            "doc_synthesis", "cross_doc_synthesis", "semantic_paraphrase",
            "reasoning", "multi_hop", "semiconductor_alias"
        }:
            if not self._judge_advanced_case(
                question=question,
                answer=answer,
                contexts=contexts,
                require_multi_source=require_multi_source,
                require_reasoning=require_reasoning,
                require_paraphrase=require_paraphrase
            ):
                return None

        gold_sources = []
        for n, idx in enumerate(unique_indices):
            ctx = contexts[idx]
            gold_sources.append({
                "source_doc": ctx["source_doc"],
                "source_path": ctx["source_path"],
                "chunk_id": int(ctx["chunk_id"]),
                "header": ctx.get("header", ""),
                "role": "primary" if n == 0 else "support"
            })

        if not gold_sources:
            return None

        primary = gold_sources[0]
        evidence_text = "\n".join(
            [f"[source{item.get('source_index', 1)}] {str(item.get('evidence', '')).strip()}" for item in evidence_list]
        )
        key_list = [f"{g['source_path']}#{g['chunk_id']}" for g in gold_sources]
        case_id = self._make_case_id(case_type, question, key_list)

        return TestCase(
            case_id=case_id,
            question=question,
            expected_behavior="answer",
            expected_answer=answer,
            evidence=evidence_text,
            case_type=case_type,
            difficulty=difficulty,
            source_doc=primary["source_doc"],
            source_path=primary["source_path"],
            source_chunk_id=primary["chunk_id"],
            source_header=primary.get("header", ""),
            origin_doc=primary["source_doc"],
            origin_path=primary["source_path"],
            origin_chunk_id=primary["chunk_id"],
            origin_header=primary.get("header", ""),
            gold_sources=gold_sources,
            tags=tags or []
        )

    def _pick_doc_chunks(self, doc_chunks: List[Chunk], k: int = 3) -> List[Chunk]:
        if len(doc_chunks) <= k:
            return doc_chunks[:]

        by_header = defaultdict(list)
        for c in doc_chunks:
            by_header[first_header_line(c.header_context, 80)].append(c)

        chosen = []
        for _, items in by_header.items():
            chosen.append(random.choice(items))
            if len(chosen) >= k:
                break

        remaining = [c for c in doc_chunks if c not in chosen]
        random.shuffle(remaining)
        for c in remaining:
            if len(chosen) >= k:
                break
            chosen.append(c)

        return chosen[:k]

    def _doc_term_set(self, doc_chunks: List[Chunk]) -> set:
        text = "\n".join(c.content[:500] for c in doc_chunks[:8])
        tokens = self.tokenizer.tokenize(text)
        freq = Counter(tokens)
        return {t for t, _ in freq.most_common(50)}

    def _rank_doc_pairs(self, doc_paths: List[str]) -> List[Tuple[str, str, int]]:
        doc_terms = {p: self._doc_term_set(self.doc_chunks[p]) for p in doc_paths}
        pairs = []

        for i in range(len(doc_paths)):
            for j in range(i + 1, len(doc_paths)):
                a = doc_paths[i]
                b = doc_paths[j]
                overlap = len(doc_terms[a] & doc_terms[b])
                pairs.append((a, b, overlap))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def _pick_best_cross_doc_pair(self, doc_a: str, doc_b: str) -> Optional[List[Dict]]:
        chunks_a = self.doc_chunks.get(doc_a, [])
        chunks_b = self.doc_chunks.get(doc_b, [])

        if not chunks_a or not chunks_b:
            return None

        sample_a = self._pick_doc_chunks(chunks_a, k=min(3, len(chunks_a)))
        sample_b = self._pick_doc_chunks(chunks_b, k=min(3, len(chunks_b)))

        best = None
        best_score = -1

        for ca in sample_a:
            ta = set(self.tokenizer.tokenize(ca.content))
            for cb in sample_b:
                tb = set(self.tokenizer.tokenize(cb.content))
                score = len(ta & tb)
                if score > best_score:
                    best_score = score
                    best = (ca, cb)

        if not best:
            return None

        return [self._ctx_from_chunk(best[0]), self._ctx_from_chunk(best[1])]

    def _build_corpus_outline(self, chunks: List[Chunk], max_files: int = 30, max_headers_per_file: int = 6) -> str:
        file_headers = defaultdict(list)

        for c in chunks:
            fn = c.metadata.get("filename", c.doc_id)
            header = first_header_line(c.header_context, 120) if c.header_context else ""
            if header and header not in file_headers[fn]:
                file_headers[fn].append(header)

        lines = []
        for i, (fn, headers) in enumerate(sorted(file_headers.items())[:max_files], 1):
            line = f"{i}. {fn}"
            if headers:
                line += " | 章节: " + " ; ".join(headers[:max_headers_per_file])
            lines.append(line)

        outline = "\n".join(lines)
        return outline[:4000]

    def _build_context_for_judge(self, results: List[SearchResult]) -> str:
        if not results:
            return "(无检索结果)"

        parts = []
        for i, r in enumerate(results, 1):
            source = r.chunk.metadata.get("filename", r.chunk.doc_id)
            part = f"[片段{i}] 来源: {source}\n"
            if r.chunk.header_context:
                part += f"章节:\n{r.chunk.header_context}\n"
            attrs = format_chunk_attrs(r.chunk.metadata, max_items=10, max_len=180)
            if attrs:
                part += f"属性: {attrs}\n"
            part += f"内容:\n{r.chunk.content}"
            parts.append(part)

        return "\n\n-----\n\n".join(parts)

    # --------------------------------------------------------
    # 基础三类：普通 / 消歧 / 负样本
    # --------------------------------------------------------
    def _generate_positive_case(self, chunk: Chunk) -> Optional[TestCase]:
        prompt = f"""
你要为一个 RAG 系统生成 1 条“可回答”的评测样本。
要求：
1. 只能依据给定文档片段，不能编造
2. 问题要像真实用户提问，不要写“根据上述内容”
3. 问题不要直接照抄原文句子
4. answer 必须简洁准确，尽量 1~3 句话
5. evidence 必须是片段中的原句或连续片段
6. 如果片段不适合出题，返回 {{"skip": true}}
7. 只输出 JSON 对象，不要解释
输出格式：
{{
  "question": "...",
  "answer": "...",
  "evidence": "...",
  "case_type": "definition|factoid|howto|constraint|parameter|comparison",
  "difficulty": "easy|medium|hard"
}}
文档名：{chunk.metadata.get("filename", "unknown")}
章节：
{chunk.header_context or "(无)"}
文档片段：
{chunk.content}
        """.strip()

        raw = self.llm.chat(
            [
                {"role": "system", "content": "你是严格的 RAG 评测样本生成助手，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.2
        )

        if raw.startswith("LLM 错误:"):
            return None

        try:
            obj = self._extract_json(raw)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            base_type = str(obj.get("case_type", "factoid")).strip() or "factoid"
            ctx = [self._ctx_from_chunk(chunk)]
            return self._build_answer_case(
                obj=obj,
                contexts=ctx,
                case_type=base_type,
                tags=["single_source"]
            )
        except Exception:
            return None

    def _generate_disambiguation_case(self, chunk: Chunk) -> Optional[TestCase]:
        prompt = f"""
你要为一个 RAG 系统生成 1 条“消歧/区分”评测样本。
要求：
1. 问题必须要求区分两个容易混淆的对象、条件、参数、阶段、选项或概念
2. 答案必须能完全从给定片段中得到
3. evidence 必须是片段中的原句或连续片段
4. 如果片段不适合生成消歧题，返回 {{"skip": true}}
5. 只输出 JSON 对象，不要解释
输出格式：
{{
  "question": "...",
  "answer": "...",
  "evidence": "...",
  "difficulty": "medium|hard"
}}
文档名：{chunk.metadata.get("filename", "unknown")}
章节：
{chunk.header_context or "(无)"}
文档片段：
{chunk.content}
        """.strip()

        raw = self.llm.chat(
            [
                {"role": "system", "content": "你是严格的 RAG 消歧样本生成助手，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=900,
            temperature=0.2
        )

        if raw.startswith("LLM 错误:"):
            return None

        try:
            obj = self._extract_json(raw)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            ctx = [self._ctx_from_chunk(chunk)]
            return self._build_answer_case(
                obj=obj,
                contexts=ctx,
                case_type="disambiguation",
                tags=["single_source", "disambiguation"]
            )
        except Exception:
            return None

    def _validate_negative_question(self, question: str, top_k: int = 5) -> bool:
        results = self.store.search(question, top_k=top_k, verbose=False)
        if not results:
            return True

        context = self._build_context_for_judge(results)
        prompt = f"""
你是 RAG 评测审查员。
请判断“问题”是否可以仅根据“上下文”得到明确答案。
判定规则：
1. 只有当上下文中存在明确、直接、充分的信息时，answerable 才为 true
2. 如果只能部分推测、需要外部知识、上下文只相关但不够回答、或问题更具体，则 answerable 必须为 false
3. 只输出 JSON，不要解释
输出格式：
{{"answerable": true/false, "reason": "..."}}
问题：
{question}
上下文：
{context}
        """.strip()

        raw = self.llm.chat(
            [
                {"role": "system", "content": "你是严格的 RAG 答案可得性审查员，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.1
        )

        if raw.startswith("LLM 错误:"):
            return False

        try:
            obj = self._extract_json(raw)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            if not isinstance(obj, dict):
                return False
            return not bool(obj.get("answerable", False))
        except Exception:
            return False

    def _generate_negative_case(self, chunk: Chunk, corpus_outline: str, max_attempts: int = 3) -> Optional[TestCase]:
        for _ in range(max_attempts):
            prompt = f"""
你要为一个 RAG 系统生成 1 条“负样本拒答”评测问题。
要求：
1. 问题主题必须与知识库相关，看起来像真实用户会问的
2. 但根据下面提供的知识库纲要和片段，无法得到明确答案
3. 不要生成和主题完全无关的问题
4. 不要写成“文档里有没有提到xxx”
5. 不要给出答案，只给问题
6. 如果做不到，返回 {{"skip": true}}
7. 只输出 JSON 对象，不要解释
输出格式：
{{
  "question": "...",
  "difficulty": "easy|medium|hard"
}}
知识库纲要：
{corpus_outline}
参考片段（用于保持主题相关，但问题不能被其回答）：
文档名：{chunk.metadata.get("filename", "unknown")}
章节：
{chunk.header_context or "(无)"}
片段：
{chunk.content}
            """.strip()

            raw = self.llm.chat(
                [
                    {"role": "system", "content": "你是严格的 RAG 负样本问题生成助手，只输出 JSON。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )

            if raw.startswith("LLM 错误:"):
                return None

            try:
                obj = self._extract_json(raw)
                if isinstance(obj, list):
                    obj = obj[0] if obj else {}
                if not isinstance(obj, dict) or obj.get("skip"):
                    continue

                question = str(obj.get("question", "")).strip()
                difficulty = str(obj.get("difficulty", "medium")).strip() or "medium"

                if not question or len(question) < 6:
                    continue

                if not self._validate_negative_question(question, top_k=5):
                    continue

                origin_path = chunk.metadata.get("source", "")
                case_id = self._make_case_id("negative", question, [f"{origin_path}#{chunk.chunk_id}"])

                return TestCase(
                    case_id=case_id,
                    question=question,
                    expected_behavior="refuse",
                    expected_answer=self.refusal_answer,
                    evidence="",
                    case_type="negative_refusal",
                    difficulty=difficulty,
                    source_doc="",
                    source_path="",
                    source_chunk_id=-1,
                    source_header="",
                    origin_doc=chunk.metadata.get("filename", chunk.doc_id),
                    origin_path=origin_path,
                    origin_chunk_id=chunk.chunk_id,
                    origin_header=chunk.header_context,
                    gold_sources=[],
                    tags=["negative"]
                )
            except Exception:
                continue

        return None

    # --------------------------------------------------------
    # 高级题型
    # --------------------------------------------------------
    def _generate_doc_synthesis_case(self, doc_path: str) -> Optional[TestCase]:
        doc_chunks = [c for c in self.doc_chunks.get(doc_path, []) if self._is_good_chunk(c, 80)]
        if len(doc_chunks) < 2:
            return None

        picked = self._pick_doc_chunks(doc_chunks, k=min(3, len(doc_chunks)))
        contexts = [self._ctx_from_chunk(c) for c in picked]

        prompt = f"""
你要为一个 RAG 系统生成 1 条“文档综合”评测样本。
要求：
1. 问题必须需要综合同一文档中至少两个不同片段/章节才能回答
2. 不能只靠任意一个来源单独回答完整
3. answer 必须能完全从给定来源得到
4. evidence_list 必须分别给出来自不同 source_index 的证据
5. 只输出 JSON
6. 如果不适合，返回 {{"skip": true}}
输出格式：
{{
  "question": "...",
  "answer": "...",
  "evidence_list": [
    {{"source_index": 1, "evidence": "..."}},
    {{"source_index": 2, "evidence": "..."}}
  ],
  "difficulty": "medium|hard"
}}
来源：
{self._format_contexts(contexts)}
        """.strip()

        raw = self.llm.chat(
            [
                {"role": "system", "content": "你是严格的 RAG 文档综合题生成助手，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )

        if raw.startswith("LLM 错误:"):
            return None

        try:
            obj = self._extract_json(raw)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            return self._build_answer_case(
                obj=obj,
                contexts=contexts,
                case_type="doc_synthesis",
                tags=["multi_source", "same_doc"],
                require_multi_source=True
            )
        except Exception:
            return None

    def _generate_cross_doc_synthesis_case(self, doc_a: str, doc_b: str) -> Optional[TestCase]:
        contexts = self._pick_best_cross_doc_pair(doc_a, doc_b)
        if not contexts or len(contexts) < 2:
            return None

        prompt = f"""
你要为一个 RAG 系统生成 1 条“跨文档综合”评测样本。
要求：
1. 问题必须需要同时综合两个不同文档的内容才能回答
2. 不能只依赖任意一个文档单独完整回答
3. answer 必须完全可由给定来源支持
4. evidence_list 必须分别包含两个来源的证据
5. 只输出 JSON
6. 如果不适合，返回 {{"skip": true}}
输出格式：
{{
  "question": "...",
  "answer": "...",
  "evidence_list": [
    {{"source_index": 1, "evidence": "..."}},
    {{"source_index": 2, "evidence": "..."}}
  ],
  "difficulty": "medium|hard"
}}
来源：
{self._format_contexts(contexts)}
        """.strip()

        raw = self.llm.chat(
            [
                {"role": "system", "content": "你是严格的 RAG 跨文档综合题生成助手，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )

        if raw.startswith("LLM 错误:"):
            return None

        try:
            obj = self._extract_json(raw)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            return self._build_answer_case(
                obj=obj,
                contexts=contexts,
                case_type="cross_doc_synthesis",
                tags=["multi_source", "cross_doc"],
                require_multi_source=True
            )
        except Exception:
            return None

    def _generate_semantic_paraphrase_case(self, chunk: Chunk) -> Optional[TestCase]:
        contexts = [self._ctx_from_chunk(chunk)]

        prompt = f"""
你要为一个 RAG 系统生成 1 条“语义改写型”评测样本。
要求：
1. 问题必须可由给定来源回答
2. 但提问方式要与原文用词明显不同，不能直接照抄文档关键词或句式
3. 尽量从用户目标、现象、场景、结果角度提问
4. evidence 必须是原文证据
5. 如果做不到，返回 {{"skip": true}}
6. 只输出 JSON
输出格式：
{{
  "question": "...",
  "answer": "...",
  "evidence": "...",
  "difficulty": "medium|hard"
}}
来源：
{self._format_contexts(contexts)}
        """.strip()

        raw = self.llm.chat(
            [
                {"role": "system", "content": "你是严格的 RAG 语义改写题生成助手，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=900,
            temperature=0.3
        )

        if raw.startswith("LLM 错误:"):
            return None

        try:
            obj = self._extract_json(raw)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            return self._build_answer_case(
                obj=obj,
                contexts=contexts,
                case_type="semantic_paraphrase",
                tags=["paraphrase", "single_source"],
                require_paraphrase=True
            )
        except Exception:
            return None

    def _generate_reasoning_case(self, chunk: Chunk) -> Optional[TestCase]:
        contexts = [self._ctx_from_chunk(chunk)]

        prompt = f"""
你要为一个 RAG 系统生成 1 条“推理型”评测样本。
要求：
1. 问题必须可由给定来源回答
2. 但答案不能只是直接摘抄一句话，应至少需要条件判断、比较、归纳、顺序推断或简单计算
3. evidence 必须是支撑推理的原文证据
4. 如果不适合，返回 {{"skip": true}}
5. 只输出 JSON
输出格式：
{{
  "question": "...",
  "answer": "...",
  "evidence": "...",
  "difficulty": "medium|hard"
}}
来源：
{self._format_contexts(contexts)}
        """.strip()

        raw = self.llm.chat(
            [
                {"role": "system", "content": "你是严格的 RAG 推理题生成助手，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=900,
            temperature=0.2
        )

        if raw.startswith("LLM 错误:"):
            return None

        try:
            obj = self._extract_json(raw)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            return self._build_answer_case(
                obj=obj,
                contexts=contexts,
                case_type="reasoning",
                tags=["reasoning", "single_source"],
                require_reasoning=True
            )
        except Exception:
            return None

    def _generate_multi_hop_case(self, doc_a: str, doc_b: str) -> Optional[TestCase]:
        contexts = self._pick_best_cross_doc_pair(doc_a, doc_b)
        if not contexts or len(contexts) < 2:
            return None

        prompt = f"""
你要为一个 RAG 系统生成 1 条“多跳问题”评测样本。
要求：
1. 问题必须需要先从一个来源得到中间事实，再结合另一个来源得到最终答案
2. 问题应体现两步以上的链式推理，而不是简单拼接
3. 不能只靠任意单个来源完整回答
4. answer 必须完全由给定来源支持
5. evidence_list 必须分别给出不同来源的证据
6. 如果不适合，返回 {{"skip": true}}
7. 只输出 JSON
输出格式：
{{
  "question": "...",
  "answer": "...",
  "evidence_list": [
    {{"source_index": 1, "evidence": "..."}},
    {{"source_index": 2, "evidence": "..."}}
  ],
  "difficulty": "hard"
}}
来源：
{self._format_contexts(contexts)}
        """.strip()

        raw = self.llm.chat(
            [
                {"role": "system", "content": "你是严格的 RAG 多跳题生成助手，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )

        if raw.startswith("LLM 错误:"):
            return None

        try:
            obj = self._extract_json(raw)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            return self._build_answer_case(
                obj=obj,
                contexts=contexts,
                case_type="multi_hop",
                tags=["multi_source", "cross_doc", "reasoning", "multi_hop"],
                require_multi_source=True,
                require_reasoning=True
            )
        except Exception:
            return None

    def _generate_semiconductor_alias_case(self, chunk: Chunk) -> Optional[TestCase]:
        if not self._is_semiconductor_candidate(chunk):
            return None

        contexts = [self._ctx_from_chunk(chunk)]
        glossary_text = "\n".join([f"- {k}: {', '.join(v)}" for k, v in self.SEMICON_GLOSSARY.items()])

        prompt = f"""
你要为一个 RAG 系统生成 1 条“半导体术语别名/黑话”评测样本。
要求：
1. 问题必须仍然可由给定来源回答
2. 提问时尽量使用半导体领域常见别名、英文缩写、黑话、口语说法，而不是直接沿用文档中的主表达
3. used_alias_terms 必须列出问题中实际使用的别名/黑话术语
4. 至少使用 1 个与原文表述不同的术语；如果做不到，返回 {{"skip": true}}
5. evidence 必须是原文证据
6. 只输出 JSON
参考术语表：
{glossary_text}
输出格式：
{{
  "question": "...",
  "answer": "...",
  "evidence": "...",
  "used_alias_terms": ["...", "..."],
  "difficulty": "medium|hard"
}}
来源：
{self._format_contexts(contexts)}
        """.strip()

        raw = self.llm.chat(
            [
                {"role": "system", "content": "你是严格的半导体领域 RAG 测试样本生成助手，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        if raw.startswith("LLM 错误:"):
            return None

        try:
            obj = self._extract_json(raw)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            return self._build_answer_case(
                obj=obj,
                contexts=contexts,
                case_type="semiconductor_alias",
                tags=["semiconductor", "alias", "paraphrase"],
                require_paraphrase=True,
                require_alias=True
            )
        except Exception:
            return None

    # --------------------------------------------------------
    # 生成入口
    # --------------------------------------------------------
    def generate_from_chunks(self,
                             chunks: List[Chunk],
                             max_positive: int = 40,
                             max_disamb: int = 15,
                             max_negative: int = 20,
                             max_doc_synth: int = 10,
                             max_cross_doc: int = 10,
                             max_semantic: int = 10,
                             max_reasoning: int = 10,
                             max_multihop: int = 10,
                             max_semiconductor: int = 10,
                             seed: int = 42,
                             min_chunk_chars: int = 120) -> List[TestCase]:
        random.seed(seed)
        candidates = [c for c in chunks if self._is_good_chunk(c, min_chunk_chars)]
        random.shuffle(candidates)

        print(f"\n开始生成测试集，候选 chunk 数量: {len(candidates)}")

        seen_questions = set()
        all_cases = []

        def try_add(tc: Optional[TestCase], bucket_name: str, current: int, target: int):
            if not tc:
                return False
            qn = normalize_text(tc.question)
            if qn in seen_questions:
                return False
            seen_questions.add(qn)
            all_cases.append(tc)
            print(f"  [{bucket_name}] {current + 1}/{target}")
            return True

        # 1) 普通题
        count = 0
        for chunk in candidates:
            if count >= max_positive:
                break
            tc = self._generate_positive_case(chunk)
            if try_add(tc, "普通样本", count, max_positive):
                count += 1

        # 2) 消歧题
        disamb_candidates = [c for c in candidates if self._is_disamb_candidate(c)]
        random.shuffle(disamb_candidates)
        count = 0
        for chunk in disamb_candidates:
            if count >= max_disamb:
                break
            tc = self._generate_disambiguation_case(chunk)
            if try_add(tc, "消歧样本", count, max_disamb):
                count += 1

        # 3) 文档综合题
        doc_candidates = [
            p for p, chs in self.doc_chunks.items()
            if len([c for c in chs if self._is_good_chunk(c, 80)]) >= 2
        ]
        random.shuffle(doc_candidates)
        count = 0
        for doc_path in doc_candidates:
            if count >= max_doc_synth:
                break
            tc = self._generate_doc_synthesis_case(doc_path)
            if try_add(tc, "文档综合", count, max_doc_synth):
                count += 1

        # 4) 跨文档综合题
        cross_doc_candidates = [
            p for p, chs in self.doc_chunks.items()
            if len([c for c in chs if self._is_good_chunk(c, 80)]) >= 1
        ]
        doc_pairs = self._rank_doc_pairs(cross_doc_candidates)
        count = 0
        for doc_a, doc_b, _score in doc_pairs:
            if count >= max_cross_doc:
                break
            tc = self._generate_cross_doc_synthesis_case(doc_a, doc_b)
            if try_add(tc, "跨文档综合", count, max_cross_doc):
                count += 1

        # 5) 语义改写题
        count = 0
        for chunk in candidates:
            if count >= max_semantic:
                break
            tc = self._generate_semantic_paraphrase_case(chunk)
            if try_add(tc, "语义改写", count, max_semantic):
                count += 1

        # 6) 推理题
        reasoning_candidates = [c for c in candidates if self._is_reasoning_candidate(c)]
        random.shuffle(reasoning_candidates)
        count = 0
        for chunk in reasoning_candidates:
            if count >= max_reasoning:
                break
            tc = self._generate_reasoning_case(chunk)
            if try_add(tc, "推理题", count, max_reasoning):
                count += 1

        # 7) 多跳题
        count = 0
        for doc_a, doc_b, _score in doc_pairs:
            if count >= max_multihop:
                break
            tc = self._generate_multi_hop_case(doc_a, doc_b)
            if try_add(tc, "多跳题", count, max_multihop):
                count += 1

        # 8) 半导体术语别名题
        semicon_candidates = [c for c in candidates if self._is_semiconductor_candidate(c)]
        random.shuffle(semicon_candidates)
        count = 0
        for chunk in semicon_candidates:
            if count >= max_semiconductor:
                break
            tc = self._generate_semiconductor_alias_case(chunk)
            if try_add(tc, "半导体别名", count, max_semiconductor):
                count += 1

        # 9) 负样本
        corpus_outline = self._build_corpus_outline(chunks)
        count = 0
        for chunk in candidates:
            if count >= max_negative:
                break
            tc = self._generate_negative_case(chunk, corpus_outline, max_attempts=3)
            if try_add(tc, "负样本", count, max_negative):
                count += 1

        random.shuffle(all_cases)
        type_counter = Counter(c.case_type for c in all_cases)

        print("\n测试集生成完成：")
        for k, v in sorted(type_counter.items()):
            print(f"  {k}: {v}")
        print(f"  总计: {len(all_cases)}")

        return all_cases

    @staticmethod
    def save_jsonl(cases: List[TestCase], path: str):
        with open(path, "w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
        print(f"\n测试集已保存到: {path}")

    @staticmethod
    def load_jsonl(path: str) -> List[TestCase]:
        cases = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj.setdefault("gold_sources", [])
                obj.setdefault("tags", [])
                cases.append(TestCase(**obj))
        return cases


# ============================================================
# 评估器
# ============================================================
class RAGEvaluator:
    def __init__(self, rag: RAG):
        self.rag = rag

    @staticmethod
    def char_f1(pred: str, gold: str) -> float:
        pred_chars = list(normalize_text(pred))
        gold_chars = list(normalize_text(gold))

        if not pred_chars or not gold_chars:
            return 0.0

        common = Counter(pred_chars) & Counter(gold_chars)
        same = sum(common.values())
        if same == 0:
            return 0.0

        precision = same / len(pred_chars)
        recall = same / len(gold_chars)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def is_refusal(answer: str) -> bool:
        if not answer:
            return False

        text = answer.strip()
        patterns = [
            r"没有找到相关信息",
            r"无法根据提供的文档",
            r"文档中未提及",
            r"上下文.*不足",
            r"无法确定",
            r"未提供相关信息"
        ]
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def summarize_testset(cases: List[TestCase]) -> Dict:
        by_case_type = Counter(c.case_type for c in cases)
        by_behavior = Counter(c.expected_behavior for c in cases)
        by_doc = Counter(get_case_doc_group(c) for c in cases if c.expected_behavior == "answer")
        by_tags = Counter(t for c in cases for t in c.tags)
        multi_source_case_count = sum(
            1 for c in cases
            if c.expected_behavior == "answer" and len(get_case_gold_sources(c)) > 1
        )

        return {
            "count": len(cases),
            "multi_source_case_count": multi_source_case_count,
            "by_case_type": dict(sorted(by_case_type.items())),
            "by_expected_behavior": dict(sorted(by_behavior.items())),
            "top_source_groups": dict(by_doc.most_common(20)),
            "top_tags": dict(by_tags.most_common(30))
        }

    def evaluate_retrieval(self, cases: List[TestCase], top_k: int = 5) -> Tuple[Dict, List[Dict]]:
        answerable = [c for c in cases if c.expected_behavior == "answer" and get_case_gold_sources(c)]
        if not answerable:
            return {"count": 0}, []

        details = []
        all_hit_count = 0
        any_hit_count = 0
        coverage_sum = 0.0
        rr_sum = 0.0
        multi_source_case_count = 0

        for idx, tc in enumerate(answerable, 1):
            golds = get_case_gold_sources(tc)
            if len(golds) > 1:
                multi_source_case_count += 1

            gold_keys = {(g["source_path"], int(g["chunk_id"])) for g in golds}
            results = self.rag.retrieve(tc.question, top_k=top_k, verbose=False)

            rank_map = {}
            top_results = []

            for rank, r in enumerate(results, 1):
                res_key = (r.chunk.metadata.get("source", ""), int(r.chunk.chunk_id))
                top_results.append({
                    "rank": rank,
                    "source_doc": r.chunk.metadata.get("filename", r.chunk.doc_id),
                    "source_path": r.chunk.metadata.get("source", ""),
                    "chunk_id": int(r.chunk.chunk_id),
                    "score": round(r.score, 6)
                })
                if res_key in gold_keys and res_key not in rank_map:
                    rank_map[res_key] = rank

            matched_keys = set(rank_map.keys())
            coverage = len(matched_keys) / len(gold_keys) if gold_keys else 0.0
            all_hit = int(len(matched_keys) == len(gold_keys) and len(gold_keys) > 0)
            any_hit = int(len(matched_keys) > 0)
            first_rank = min(rank_map.values()) if rank_map else 0
            rr = 1.0 / first_rank if first_rank else 0.0

            all_hit_count += all_hit
            any_hit_count += any_hit
            coverage_sum += coverage
            rr_sum += rr

            details.append({
                "case_id": tc.case_id,
                "question": tc.question,
                "case_type": tc.case_type,
                "source_group": get_case_doc_group(tc),
                "header_group": get_case_header_group(tc),
                "gold_source_count": len(gold_keys),
                "matched_source_count": len(matched_keys),
                "support_coverage": round(coverage, 4),
                "all_support_hit": bool(all_hit),
                "any_support_hit": bool(any_hit),
                "first_support_rank": first_rank,
                "top_results": top_results
            })

            if idx % 10 == 0 or idx == len(answerable):
                print(f"[检索评估] {idx}/{len(answerable)}")

        summary = {
            "count": len(answerable),
            "multi_source_case_count": multi_source_case_count,
            f"recall@{top_k}": round(all_hit_count / len(answerable), 4),
            f"all_support_hit_rate@{top_k}": round(all_hit_count / len(answerable), 4),
            f"any_support_hit_rate@{top_k}": round(any_hit_count / len(answerable), 4),
            f"avg_support_coverage@{top_k}": round(coverage_sum / len(answerable), 4),
            f"mrr_first_support@{top_k}": round(rr_sum / len(answerable), 4)
        }

        return summary, details

    def evaluate_generation(self, cases: List[TestCase], top_k: int = 5) -> Tuple[Dict, List[Dict]]:
        details = []

        ans_total = 0
        ans_contain_hit = 0
        ans_f1_sum = 0.0

        neg_total = 0
        neg_refusal_hit = 0

        for idx, tc in enumerate(cases, 1):
            answer, _ = self.rag.query(tc.question, top_k=top_k, verbose=False)

            if tc.expected_behavior == "refuse":
                neg_total += 1
                refusal_ok = int(self.is_refusal(answer))
                neg_refusal_hit += refusal_ok

                details.append({
                    "case_id": tc.case_id,
                    "question": tc.question,
                    "case_type": tc.case_type,
                    "expected_behavior": "refuse",
                    "source_group": get_case_doc_group(tc),
                    "header_group": get_case_header_group(tc),
                    "pred_answer": answer,
                    "refusal_ok": refusal_ok,
                    "tags": tc.tags
                })
            else:
                ans_total += 1
                pred_norm = normalize_text(answer)
                gold_norm = normalize_text(tc.expected_answer)
                contain = int(gold_norm in pred_norm) if gold_norm else 0
                f1 = self.char_f1(answer, tc.expected_answer)

                ans_contain_hit += contain
                ans_f1_sum += f1

                details.append({
                    "case_id": tc.case_id,
                    "question": tc.question,
                    "case_type": tc.case_type,
                    "expected_behavior": "answer",
                    "source_group": get_case_doc_group(tc),
                    "header_group": get_case_header_group(tc),
                    "expected_answer": tc.expected_answer,
                    "pred_answer": answer,
                    "answer_contain_hit": contain,
                    "char_f1": round(f1, 4),
                    "tags": tc.tags
                })

            if idx % 5 == 0 or idx == len(cases):
                print(f"[回答评估] {idx}/{len(cases)}")

        summary = {
            "answerable_count": ans_total,
            "answerable_contain_rate": round(ans_contain_hit / ans_total, 4) if ans_total else 0.0,
            "answerable_avg_char_f1": round(ans_f1_sum / ans_total, 4) if ans_total else 0.0,
            "negative_count": neg_total,
            "negative_refusal_rate": round(neg_refusal_hit / neg_total, 4) if neg_total else 0.0
        }

        return summary, details

    def build_retrieval_grouped_report(self, cases: List[TestCase], details: List[Dict], top_k: int = 5) -> Dict:
        by_case_type = defaultdict(lambda: {"count": 0, "all_hit": 0, "any_hit": 0, "coverage_sum": 0.0, "rr_sum": 0.0})
        by_source_doc = defaultdict(lambda: {"count": 0, "all_hit": 0, "any_hit": 0, "coverage_sum": 0.0, "rr_sum": 0.0})
        by_source_header = defaultdict(lambda: {"count": 0, "all_hit": 0, "any_hit": 0, "coverage_sum": 0.0, "rr_sum": 0.0})

        for d in details:
            case_type = d["case_type"]
            source_doc = d["source_group"]
            source_header = d["header_group"]
            all_hit = int(d["all_support_hit"])
            any_hit = int(d["any_support_hit"])
            coverage = float(d["support_coverage"])
            rr = 1.0 / d["first_support_rank"] if d["first_support_rank"] else 0.0

            for agg, key in [
                (by_case_type, case_type),
                (by_source_doc, source_doc),
                (by_source_header, source_header)
            ]:
                agg[key]["count"] += 1
                agg[key]["all_hit"] += all_hit
                agg[key]["any_hit"] += any_hit
                agg[key]["coverage_sum"] += coverage
                agg[key]["rr_sum"] += rr

        def finalize(aggs: Dict[str, Dict]) -> Dict[str, Dict]:
            out = {}
            for key, v in aggs.items():
                count = v["count"]
                out[key] = {
                    "count": count,
                    f"recall@{top_k}": round(v["all_hit"] / count, 4) if count else 0.0,
                    f"all_support_hit_rate@{top_k}": round(v["all_hit"] / count, 4) if count else 0.0,
                    f"any_support_hit_rate@{top_k}": round(v["any_hit"] / count, 4) if count else 0.0,
                    f"avg_support_coverage@{top_k}": round(v["coverage_sum"] / count, 4) if count else 0.0,
                    f"mrr_first_support@{top_k}": round(v["rr_sum"] / count, 4) if count else 0.0
                }
            return sort_metric_dict(out, primary_key="count")

        return {
            "by_case_type": finalize(by_case_type),
            "by_source_doc": finalize(by_source_doc),
            "by_source_header": finalize(by_source_header)
        }

    def build_generation_grouped_report(self, cases: List[TestCase], details: List[Dict]) -> Dict:
        ans_by_case_type = defaultdict(lambda: {"count": 0, "contain_hit": 0, "f1_sum": 0.0})
        ans_by_source_doc = defaultdict(lambda: {"count": 0, "contain_hit": 0, "f1_sum": 0.0})
        ans_by_source_header = defaultdict(lambda: {"count": 0, "contain_hit": 0, "f1_sum": 0.0})
        neg_by_case_type = defaultdict(lambda: {"count": 0, "refusal_hit": 0})
        neg_by_origin_doc = defaultdict(lambda: {"count": 0, "refusal_hit": 0})

        case_map = {c.case_id: c for c in cases}

        for d in details:
            tc = case_map[d["case_id"]]
            if tc.expected_behavior == "refuse":
                neg_by_case_type[tc.case_type]["count"] += 1
                neg_by_case_type[tc.case_type]["refusal_hit"] += int(d.get("refusal_ok", 0))
                origin_label = tc.origin_doc or "(未知来源)"
                neg_by_origin_doc[origin_label]["count"] += 1
                neg_by_origin_doc[origin_label]["refusal_hit"] += int(d.get("refusal_ok", 0))
            else:
                source_doc = get_case_doc_group(tc)
                source_header = get_case_header_group(tc)
                for agg, key in [
                    (ans_by_case_type, tc.case_type),
                    (ans_by_source_doc, source_doc),
                    (ans_by_source_header, source_header)
                ]:
                    agg[key]["count"] += 1
                    agg[key]["contain_hit"] += int(d.get("answer_contain_hit", 0))
                    agg[key]["f1_sum"] += float(d.get("char_f1", 0.0))

        def finalize_answer(aggs: Dict[str, Dict]) -> Dict[str, Dict]:
            out = {}
            for key, v in aggs.items():
                count = v["count"]
                out[key] = {
                    "count": count,
                    "answer_contain_rate": round(v["contain_hit"] / count, 4) if count else 0.0,
                    "avg_char_f1": round(v["f1_sum"] / count, 4) if count else 0.0
                }
            return sort_metric_dict(out, primary_key="count")

        def finalize_refuse(aggs: Dict[str, Dict]) -> Dict[str, Dict]:
            out = {}
            for key, v in aggs.items():
                count = v["count"]
                out[key] = {
                    "count": count,
                    "negative_refusal_rate": round(v["refusal_hit"] / count, 4) if count else 0.0
                }
            return sort_metric_dict(out, primary_key="count")

        return {
            "answerable_by_case_type": finalize_answer(ans_by_case_type),
            "answerable_by_source_doc": finalize_answer(ans_by_source_doc),
            "answerable_by_source_header": finalize_answer(ans_by_source_header),
            "negative_by_case_type": finalize_refuse(neg_by_case_type),
            "negative_by_origin_doc": finalize_refuse(neg_by_origin_doc)
        }

    @staticmethod
    def save_report(path: str, report: Dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n评估报告已保存到: {path}")


# ============================================================
# 主程序
# ============================================================
def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║           RAG 知识问答系统（增强版）                      ║
║  - Markdown智能分块                                       ║
║  - 混合检索 (BM25 + 向量)                                 ║
║  - YAML front matter 元数据支持                           ║
║  - 文件名优先但不锁死召回                                 ║
╚══════════════════════════════════════════════════════════╝
""")

    try:
        rag = RAG()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    if KNOWLEDGE_SOURCE:
        rag.load_knowledge(KNOWLEDGE_SOURCE)
    else:
        print("⚠️ 未配置知识源，请使用 /add 命令添加文档")

    rag.interactive()


if __name__ == "__main__":
    main()
