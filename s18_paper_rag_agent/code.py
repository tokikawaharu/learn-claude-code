#!/usr/bin/env python3
"""
s18_paper_rag_agent.py - 科研论文多轮检索问答 Agent（领域化二开）

上游：shareAI-lab/learn-claude-code（nano claude-code 风格的 agent harness 教学实现）
本节在其 harness 思路（agent loop / context compact / memory）之上，落地一个
可离线运行、可评测的领域化检索问答 Agent。

    user turn
       |
  Guardrails -> NLU(意图/指代消解/改写) -> Retrieve(稠密+稀疏) -> Rerank
       -> Reflect(证据不足?) --yes--> 二次检索 / 拒答
       -> Generate(引用式回答) -> Memory(会话记忆 + 压缩) -> PipelineTrace

设计约束：
- 纯标准库实现，不下载模型即可运行与评测；外部依赖全部可选（sentence-transformers / anthropic）
- 每个阶段都记录到 Turn/Stage 两级 PipelineTrace，可用 /trace 回放
- 检索组件可插拔：Embedder / Reranker 均有确定性默认实现，可用真实模型替换

用法：
    python s18_paper_rag_agent/code.py --corpus s18_paper_rag_agent/corpus
    /ask 论文里 GRPO 的 Clip-Higher 解决什么问题
    /corpus   查看语料
    /trace    查看上一轮各阶段轨迹
    /memory   查看会话记忆与压缩情况
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# --------------------------------------------------------------------------------------
# 语料与分块
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """一个可检索的最小单元，保留论文/章节等元信息以便引用回溯。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_title} § {self.section}"


_SECTION_RE = re.compile(r"^#{1,3}\s+(.+)$")
_MAX_CHARS = 320
_OVERLAP_CHARS = 80


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in parts if p]


def _sliding_window(text: str, max_chars: int = _MAX_CHARS, overlap: int = _OVERLAP_CHARS) -> list[str]:
    """段落超长时按滑动窗口切分，重叠区保证跨窗口的语义连续。"""
    if len(text) <= max_chars:
        return [text]
    step = max(1, max_chars - overlap)
    windows: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + max_chars]
        if piece.strip():
            windows.append(piece.strip())
        if start + max_chars >= len(text):
            break
    return windows


def load_corpus(root: str | Path) -> list[Chunk]:
    """按 markdown 标题切章节，章节内按段落 + 滑动窗口切块。"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"语料目录不存在：{root}")
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.md")):
        title = path.stem
        section = "正文"
        buffer: list[str] = []
        doc_index = 0
        first_heading = True

        def flush(section_name: str, paragraphs: Sequence[str]) -> None:
            nonlocal doc_index
            for paragraph in paragraphs:
                for window in _sliding_window(paragraph):
                    doc_index += 1
                    chunks.append(
                        Chunk(
                            chunk_id=f"{path.stem}#{doc_index}",
                            doc_id=path.stem,
                            doc_title=title,
                            section=section_name,
                            text=window,
                        )
                    )

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            if line.startswith("#"):
                match = _SECTION_RE.match(line)
                if match:
                    flush(section, buffer)
                    buffer = []
                    heading = match.group(1).strip()
                    if first_heading:
                        # 每篇文档的首个标题作为论文标题，后续标题作为章节名
                        title = heading
                        section = "正文"
                        first_heading = False
                    else:
                        section = heading
                    continue
                continue
            buffer.append(line)
        flush(section, buffer)
    return [c for c in chunks if c.text.strip()]


# --------------------------------------------------------------------------------------
# 分词：中文按字符二元组 + 英文/数字按词，无需外部词典即可稳定复现
# --------------------------------------------------------------------------------------

_ASCII_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+\-\.]*|\d+(?:\.\d+)?")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_STOPWORDS = {"的", "了", "是", "在", "和", "与", "及", "对", "把", "被", "用", "从", "为", "以", "后", "上", "中"}
# 以助词开头的二元组几乎不是实义短语，直接丢弃以压低噪声
_PARTICLE_PREFIX = {"的", "了", "是", "着", "呢", "吧", "啊"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """中文二元组 + ASCII 词 + 单个中文词（去停用词），保证检索对中文友好。"""
    text = normalize(text)
    tokens: list[str] = [t for t in _ASCII_TOKEN_RE.findall(text)]
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            continue
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if bigram in _STOPWORDS or bigram[0] in _PARTICLE_PREFIX:
                continue
            tokens.append(bigram)
    return tokens


def _squash(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


MIN_COVERAGE = 0.30


def query_coverage(query: str, chunk: Chunk) -> float:
    """查询词元在候选块里的覆盖率，用于域外问题判定与重排打分。"""
    q_tokens = [t for t in tokenize(query) if len(t) > 1]
    if not q_tokens:
        return 0.0
    text = normalize(chunk.text)
    return sum(1 for token in q_tokens if token in text) / len(q_tokens)


# --------------------------------------------------------------------------------------
# 检索组件（全部可插拔）
# --------------------------------------------------------------------------------------


class BM25Index:
    """稀疏召回：BM25(k1=1.5, b=0.75)。"""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(c.text) for c in self.chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, tokens in enumerate(self.docs):
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, freq in counts.items():
                self.postings.setdefault(token, []).append((index, freq))

    def search(self, query: str, limit: int = 8) -> list[tuple[Chunk, float]]:
        scores: dict[int, float] = {}
        total = len(self.chunks)
        for token in tokenize(query):
            entries = self.postings.get(token)
            if not entries:
                continue
            idf = math.log(1 + (total - len(entries) + 0.5) / (len(entries) + 0.5))
            for index, freq in entries:
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[index] / max(1e-9, self.avg_len))
                scores[index] = scores.get(index, 0.0) + idf * freq * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [(self.chunks[index], score) for index, score in ranked]


class HashingEmbedder:
    """确定性稠密向量：字符/词混合 n-gram 哈希投影。

    用于离线评测与单测；安装 sentence-transformers 后可用 BGE-M3 替换，
    对外接口保持一致（embed(text) -> list[float]）。
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _squash(vector)


class DenseIndex:
    """稠密召回：向量内积（向量已归一化，等价余弦）。"""

    def __init__(self, chunks: Sequence[Chunk], embedder: Any | None = None) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder or HashingEmbedder()
        self.vectors = [self.embedder.embed(c.text) for c in self.chunks]

    def search(self, query: str, limit: int = 8) -> list[tuple[Chunk, float]]:
        qv = self.embedder.embed(query)
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            score = sum(a * b for a, b in zip(qv, vec))
            scored.append((chunk, score))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:limit]


class LexicalCrossReranker:
    """可插拔 rerank：默认实现为词法交叉打分（覆盖度 + 邻近度 + 章节命中）。

    真实部署可换成 Cross-Encoder（bge-reranker 等），接口一致：
    rerank(query, candidates) -> list[(chunk, score)]
    """

    def __init__(self, sections_weight: float = 0.25) -> None:
        self.sections_weight = sections_weight

    def score(self, query: str, chunk: Chunk) -> float:
        q_tokens = [t for t in tokenize(query) if len(t) > 1]
        if not q_tokens:
            return 0.0
        coverage = query_coverage(query, chunk)
        text = normalize(chunk.text)
        covered = [t for t in q_tokens if t in text]
        positions = [text.find(t) for t in covered]
        positions = [p for p in positions if p >= 0]
        proximity = 0.0
        if len(positions) > 1:
            span = max(positions) - min(positions)
            proximity = 1.0 / (1.0 + span / 200.0)
        section_hit = 1.0 if any(t in normalize(chunk.section) for t in q_tokens) else 0.0
        # 覆盖度做非线性放大：只有覆盖足够多的查询词元才应该排到最前
        return coverage ** 1.5 + 0.4 * proximity + self.sections_weight * section_hit

    def rerank(self, query: str, candidates: Sequence[tuple[Chunk, float]], limit: int = 5) -> list[tuple[Chunk, float]]:
        scored = [(chunk, self.score(query, chunk)) for chunk, _ in candidates]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:limit]


def rrf_fuse(rankings: Sequence[Sequence[tuple[Chunk, float]]], k: int = 60) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion：把多路召回的排名融合成一个稳定序列。"""
    fused: dict[str, float] = {}
    seen: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, (chunk, _score) in enumerate(ranking, start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            seen[chunk.chunk_id] = chunk
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(seen[cid], score) for cid, score in ordered]


class HybridRetriever:
    """稠密 + 稀疏双路召回，RRF 融合，可选 rerank。"""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Any | None = None,
        reranker: Any | None = None,
        use_dense: bool = True,
        use_rerank: bool = True,
    ) -> None:
        self.chunks = list(chunks)
        self.sparse = BM25Index(self.chunks)
        self.dense = DenseIndex(self.chunks, embedder) if use_dense else None
        self.reranker = reranker
        self.use_rerank = use_rerank

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 8) -> tuple[list[tuple[Chunk, float]], dict[str, Any]]:
        debug: dict[str, Any] = {"mode": []}
        sparse_hits = self.sparse.search(query, limit=candidate_k)
        rankings = [sparse_hits]
        debug["mode"].append("sparse")
        if self.dense is not None:
            rankings.append(self.dense.search(query, limit=candidate_k))
            debug["mode"].append("dense")
        fused = rrf_fuse(rankings)
        debug["fused"] = len(fused)
        if self.use_rerank and self.reranker is not None and fused:
            reranked = self.reranker.rerank(query, fused[: max(top_k * 3, top_k)], limit=top_k)
            debug["mode"].append("rerank")
            debug["top_score"] = round(reranked[0][1], 4) if reranked else 0.0
            return reranked, debug
        debug["top_score"] = round(fused[0][1], 6) if fused else 0.0
        return fused[:top_k], debug


# --------------------------------------------------------------------------------------
# 轨迹与状态
# --------------------------------------------------------------------------------------


@dataclass
class StageTrace:
    stage: str
    duration_ms: float
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnTrace:
    index: int
    query: str
    rewritten_query: str = ""
    intent: str = ""
    refused: bool = False
    second_pass: bool = False
    stages: list[StageTrace] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    def add(self, stage: str, started: float, summary: str, **detail: Any) -> None:
        self.stages.append(
            StageTrace(
                stage=stage,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                summary=summary,
                detail=detail,
            )
        )


@dataclass
class MemoryEntry:
    turn: int
    kind: str  # "verbatim" | "gist"
    text: str


@dataclass
class AgentState:
    """跨轮共享状态：会话记忆 + 轨迹 + 语料句柄 + 主题锚点。"""

    retriever: HybridRetriever
    chunks: list[Chunk]
    memory: list[MemoryEntry] = field(default_factory=list)
    turns: list[TurnTrace] = field(default_factory=list)
    last_entities: dict[str, str] = field(default_factory=dict)
    topic_anchors: dict[str, str] = field(default_factory=dict)  # 词元 -> doc_id
    doc_titles: dict[str, str] = field(default_factory=dict)  # doc_id -> 标题
    doc_keywords: dict[str, str] = field(default_factory=dict)  # doc_id -> 代表关键词
    max_verbatim_turns: int = 3

    @property
    def turn_index(self) -> int:
        return len(self.turns) + 1


# --------------------------------------------------------------------------------------
# 七个阶段
# --------------------------------------------------------------------------------------

INTENTS = ("lookup", "compare", "summarize", "followup_ref", "out_of_scope")

_REFERENTIAL = ("它", "这个", "这项", "该方法", "这个方法", "上述", "上面提到", "那", "该论文", "这篇",
                "这篇文章", "该模型", "这个模型", "该技巧", "那个", "上述方法")

_INTENT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("compare", ("对比", "比较", "区别", "差异", "相比", "vs")),
    ("summarize", ("总结", "概括", "综述", "讲了什么", "主要内容", "梳理")),
    ("out_of_scope", ("天气", "股票", "吃什么", "写一首诗", "生日", "彩票", "翻译成")),
    ("followup_ref", _REFERENTIAL),
]


def build_topic_index(chunks: Sequence[Chunk]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """从语料构建 主题锚点表 / 标题表 / 每篇的代表关键词，用于指代消解。"""
    doc_titles: dict[str, str] = {}
    doc_keywords: dict[str, str] = {}
    anchors: dict[str, str] = {}
    per_doc_tokens: dict[str, dict[str, int]] = {}

    for chunk in chunks:
        doc_titles[chunk.doc_id] = chunk.doc_title
        counts = per_doc_tokens.setdefault(chunk.doc_id, {})
        for token in tokenize(chunk.text):
            if len(token) > 1:
                counts[token] = counts.get(token, 0) + 1

    for doc_id, title in doc_titles.items():
        # 标题词元作为强锚点（优先级最高）
        for token in set(tokenize(title)):
            if len(token) > 1:
                anchors.setdefault(token, doc_id)
        counts = per_doc_tokens.get(doc_id, {})
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        doc_keywords[doc_id] = " ".join(token for token, _ in top)
        for token, _freq in top[:6]:
            anchors.setdefault(token, doc_id)
    return anchors, doc_titles, doc_keywords


def detect_topic(query: str, state: AgentState, min_hits: int = 2) -> str:
    """判断这句话是否在明确指向某一篇文档（命中足够多的主题锚点）。"""
    hits: dict[str, int] = {}
    for token in tokenize(query):
        doc_id = state.topic_anchors.get(token)
        if doc_id:
            hits[doc_id] = hits.get(doc_id, 0) + 1
    if not hits:
        return ""
    doc_id, count = max(hits.items(), key=lambda kv: kv[1])
    return doc_id if count >= min_hits else ""


def stage_guardrails(query: str, max_chars: int = 400) -> tuple[bool, str]:
    """越界与超长输入拦截；返回 (是否放行, 说明)。"""
    if not query.strip():
        return False, "空查询"
    if len(query) > max_chars:
        return False, f"查询超过 {max_chars} 字，请拆分后再问"
    return True, "pass"


def stage_nlu(query: str, state: AgentState) -> dict[str, Any]:
    """意图识别 + 指代消解 + Query 改写（结构化输出）。"""
    normalized = normalize(query)
    intent = "lookup"
    for label, keywords in _INTENT_PATTERNS:
        if any(k in normalized for k in keywords):
            intent = label
            break

    # 主题判定优先于指代消解：查询本身已经指向某篇文档时，不能被上一轮的实体污染
    topic = detect_topic(query, state)
    if topic:
        state.last_entities["doc_id"] = topic
        state.last_entities["entity"] = state.doc_titles.get(topic, "")

    resolved_entity = ""
    rewritten = query.strip()
    has_pronoun = any(pronoun in normalized for pronoun in _REFERENTIAL)
    entity_doc = state.last_entities.get("doc_id", "")
    keywords = state.doc_keywords.get(entity_doc, "")

    if not topic and has_pronoun:
        if keywords:
            resolved_entity = state.last_entities.get("entity", "")
            # 用该篇的代表关键词替换代词：标题里的词不一定出现在正文块中
            replaced = False
            for pronoun in _REFERENTIAL:
                if pronoun in rewritten:
                    rewritten = rewritten.replace(pronoun, keywords, 1)
                    replaced = True
                    break
            if not replaced:
                rewritten = f"{keywords} {rewritten}"
            if intent != "compare":
                intent = "followup_ref"
        else:
            rewritten = ""  # 无法消解，交给 Reflect 判为证据不足

    # Query 改写：去掉口语填充词，补全领域同义表述
    rewritten = re.sub(r"(请问|麻烦|帮我|我想知道|能不能)", "", rewritten).strip()
    synonyms = {"效果": "性能 提升", "怎么做的": "方法 实现", "训练不稳": "训练 不稳定 熵", "准确率": "accuracy"}
    for src, dst in synonyms.items():
        if src in rewritten and dst.split()[0] not in rewritten:
            rewritten = f"{rewritten} {dst}"
    return {"intent": intent, "rewritten": rewritten, "resolved_entity": resolved_entity, "topic": topic}


def stage_retrieve(query: str, state: AgentState, top_k: int = 5) -> tuple[list[tuple[Chunk, float]], dict[str, Any]]:
    if not query:
        return [], {"mode": [], "reason": "empty_query"}
    return state.retriever.retrieve(query, top_k=top_k)


def stage_reflect(
    query: str,
    hints: list[tuple[Chunk, float]],
    state: AgentState,
    threshold: float,
    intent: str = "lookup",
) -> tuple[bool, str, list[tuple[Chunk, float]]]:
    """证据充分性判定；越界意图直接拒答，证据不足先二次检索、仍不足则拒答。"""
    if intent == "out_of_scope":
        return True, "out_of_scope", []
    if not hints:
        return True, "no_hits", []
    top_chunk, top_score = hints[0]
    # 第一道闸：查询自身的词元在最佳命中里几乎不出现 → 判定为域外问题，不做二次检索
    coverage = query_coverage(query, top_chunk)
    if coverage < MIN_COVERAGE:
        return True, f"out_of_domain(coverage={coverage:.2f})", hints
    if top_score >= threshold:
        return False, "ok", hints
    # 第二道闸：证据边缘不足时，用 Top 命中里的关键词扩充查询做二次检索
    expansion = " ".join(tokenize(top_chunk.text)[:6])
    retry, debug = state.retriever.retrieve(f"{query} {expansion}", top_k=5)
    if retry and retry[0][1] >= threshold and query_coverage(query, retry[0][0]) >= MIN_COVERAGE:
        if debug.get("mode"):
            debug["mode"].append("reflect_retry")
        return False, "second_pass", retry
    return True, f"insufficient_evidence(top={top_score:.3f}, cov={coverage:.2f})", retry or hints


def stage_generate(query: str, hints: list[tuple[Chunk, float]], llm: Callable[[str], str] | None) -> str:
    """引用式生成：默认离线模板（可换成 Claude/本地模型），逐条给出 [n] 引用。"""
    if llm is not None:
        context = "\n".join(f"[{i+1}] {c.citation}: {c.text}" for i, (c, _) in enumerate(hints))
        return llm(f"仅依据下列材料回答问题，逐句标注引用编号。\n问题：{query}\n材料：\n{context}")
    lines = [f"依据检索到的 {min(3, len(hints))} 条证据作答："]
    for i, (chunk, _) in enumerate(hints[:3], start=1):
        snippet = re.sub(r"\s+", " ", chunk.text).strip()
        if len(snippet) > 110:
            snippet = snippet[:110] + "…"
        lines.append(f"- {snippet} [{i}]")
    lines.append("引用：" + "；".join(f"[{i}] {c.citation}" for i, (c, _) in enumerate(hints[:3], start=1)))
    return "\n".join(lines)


def stage_memory(state: AgentState, turn: TurnTrace, answer: str, recall: int = 8) -> dict[str, Any]:
    """会话记忆更新 + 按需压缩：保留最近 N 轮原文，更早的折叠为 gist。"""
    entity = ""
    if turn.citations:
        cited_title = turn.citations[0].split(" § ")[0]
        doc_id = next((doc for doc, title in state.doc_titles.items() if title == cited_title), "")
        if doc_id:
            state.last_entities["doc_id"] = doc_id
            state.last_entities["entity"] = cited_title
        entity = cited_title
    elif turn.rewritten_query:
        entity = turn.rewritten_query[:20]

    state.memory.append(MemoryEntry(turn=turn.index, kind="verbatim", text=answer))

    compressed = 0
    verbatim = [m for m in state.memory if m.kind == "verbatim"]
    if len(verbatim) > state.max_verbatim_turns:
        overflow = verbatim[: len(verbatim) - state.max_verbatim_turns]
        for entry in overflow:
            summary = re.sub(r"\s+", " ", entry.text).strip()[:60]
            entry.kind = "gist"
            entry.text = f"T{entry.turn}: {summary}"
            compressed += 1
    return {"entity": entity, "compressed": compressed, "memory_size": len(state.memory)}


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


class PaperQAAgent:
    """七阶段可追溯问答 Agent。"""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Any | None = None,
        reranker: Any | None = None,
        use_dense: bool = True,
        use_rerank: bool = True,
        threshold: float = 0.35,
        llm: Callable[[str], str] | None = None,
    ) -> None:
        retriever = HybridRetriever(chunks, embedder=embedder, reranker=reranker or LexicalCrossReranker(),
                                    use_dense=use_dense, use_rerank=use_rerank)
        anchors, doc_titles, doc_keywords = build_topic_index(chunks)
        self.state = AgentState(
            retriever=retriever,
            chunks=list(chunks),
            topic_anchors=anchors,
            doc_titles=doc_titles,
            doc_keywords=doc_keywords,
        )
        self.threshold = threshold
        self.llm = llm

    # -- 主流程 ------------------------------------------------------------------
    def ask(self, query: str) -> dict[str, Any]:
        turn = TurnTrace(index=self.state.turn_index, query=query)
        self.state.turns.append(turn)

        started = time.perf_counter()
        allowed, reason = stage_guardrails(query)
        turn.add("Guardrails", started, reason)
        if not allowed:
            turn.refused = True
            return self._result(turn, f"无法处理该请求：{reason}")

        started = time.perf_counter()
        nlu = stage_nlu(query, self.state)
        turn.intent = nlu["intent"]
        turn.rewritten_query = nlu["rewritten"]
        turn.add("NLU", started, f"intent={nlu['intent']}", resolved_entity=nlu["resolved_entity"],
                 rewritten=nlu["rewritten"])

        started = time.perf_counter()
        hints, debug = stage_retrieve(turn.rewritten_query, self.state)
        turn.add("Retrieve", started, f"{len(hints)} hits", mode=debug.get("mode", []))

        started = time.perf_counter()
        refused, why, final_hints = stage_reflect(turn.rewritten_query, hints, self.state, self.threshold,
                                                  intent=turn.intent)
        turn.second_pass = why == "second_pass"
        turn.add("Reflect", started, why, second_pass=turn.second_pass)
        if refused:
            turn.refused = True
            answer = ("该问题超出语料范围，已拒答（仅回答论文库内的问题）。" if why == "out_of_scope"
                      else "检索到的证据不足以可靠回答该问题，已拒答（避免无依据作答）。")
            self._remember(turn, answer)
            return self._result(turn, answer)

        started = time.perf_counter()
        answer = stage_generate(turn.rewritten_query, final_hints, self.llm)
        turn.citations = [c.citation for c, _ in final_hints[:3]]
        turn.add("Generate", started, f"{len(turn.citations)} citations", citations=turn.citations)

        started = time.perf_counter()
        memory_info = stage_memory(self.state, turn, answer)
        turn.add("Memory", started, f"entity={memory_info['entity'] or '-'} compressed={memory_info['compressed']}",
                 **memory_info)
        return self._result(turn, answer)

    def _remember(self, turn: TurnTrace, answer: str) -> None:
        started = time.perf_counter()
        info = stage_memory(self.state, turn, answer)
        turn.add("Memory", started, f"refused-turn stored", **info)

    def _result(self, turn: TurnTrace, answer: str) -> dict[str, Any]:
        return {
            "turn": turn.index,
            "query": turn.query,
            "rewritten": turn.rewritten_query,
            "intent": turn.intent,
            "refused": turn.refused,
            "second_pass": turn.second_pass,
            "answer": answer,
            "citations": turn.citations,
            "trace": turn,
        }

    # -- 可观测 ------------------------------------------------------------------
    def render_trace(self, turn_index: int | None = None) -> str:
        turns = self.state.turns
        if not turns:
            return "（暂无轨迹）"
        target = turns[-1] if turn_index is None else turns[turn_index - 1]
        lines = [f"Turn {target.index} | query={target.query!r} | intent={target.intent} | refused={target.refused}"]
        for step in target.stages:
            detail = ", ".join(f"{k}={v}" for k, v in step.detail.items() if v not in ("", None, [], {}))
            lines.append(f"  {step.stage:<10} {step.duration_ms:>7.3f}ms  {step.summary}"
                         + (f"  ({detail})" if detail else ""))
        if target.citations:
            lines.append("  citations: " + "; ".join(target.citations))
        return "\n".join(lines)

    def render_memory(self) -> str:
        if not self.state.memory:
            return "（暂无记忆）"
        lines = []
        for entry in self.state.memory:
            text = re.sub(r"\s+", " ", entry.text).strip()
            lines.append(f"  T{entry.turn} [{entry.kind}] {text[:80]}")
        return "\n".join(lines)

    def render_corpus(self) -> str:
        lines = [f"语料块数：{len(self.state.chunks)}"]
        seen: dict[str, list[Chunk]] = {}
        for chunk in self.state.chunks:
            seen.setdefault(chunk.doc_id, []).append(chunk)
        for doc_id, items in seen.items():
            sections = sorted({c.section for c in items})
            lines.append(f"  {doc_id} | {items[0].doc_title} | {len(items)} 块 | 章节: {', '.join(sections)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="科研论文多轮检索问答 Agent（s18）")
    parser.add_argument("--corpus", default=str(Path(__file__).with_name("corpus")), help="语料目录")
    parser.add_argument("--no-dense", action="store_true", help="关闭稠密召回（仅 BM25）")
    parser.add_argument("--no-rerank", action="store_true", help="关闭 rerank")
    parser.add_argument("--ask", help="非交互模式：只回答一个问题后退出")
    parser.add_argument("--threshold", type=float, default=0.35, help="证据充分性阈值")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    chunks = load_corpus(args.corpus)
    if not chunks:
        print(f"语料目录为空：{args.corpus}", file=sys.stderr)
        return 2
    agent = PaperQAAgent(chunks, use_dense=not args.no_dense, use_rerank=not args.no_rerank,
                         threshold=args.threshold)

    if args.ask:
        result = agent.ask(args.ask)
        print(result["answer"])
        print()
        print(agent.render_trace())
        return 0

    print(f"已加载 {len(chunks)} 个语料块。命令：/corpus /trace /memory /quit")
    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue
        if raw in ("/quit", "/exit"):
            return 0
        if raw == "/corpus":
            print(agent.render_corpus())
            continue
        if raw == "/trace":
            print(agent.render_trace())
            continue
        if raw == "/memory":
            print(agent.render_memory())
            continue
        result = agent.ask(raw)
        print(result["answer"])
        print(f"[intent={result['intent']} refused={result['refused']} second_pass={result['second_pass']}]")


if __name__ == "__main__":
    raise SystemExit(main())
