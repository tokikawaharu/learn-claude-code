# s18: A Multi-turn Paper Retrieval Agent (domain fork)

[中文](README.zh.md) · [English](README.md)

`s01` → … → `s17` → **`s18`**

> *This chapter applies the harness ideas from the previous chapters (agent loop / context compact / memory) to a real domain: multi-turn Q&A over a research paper library.*

- **Fork note**: built on the teaching skeleton of [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code). It adds `s18_paper_rag_agent/` and does not touch any existing chapter.
- **Runs without any model**: standard library only, with deterministic embedder/reranker defaults, so `python s18_paper_rag_agent/eval.py` reproduces every number in the table below.

## The three failure modes it targets

1. **Missing reference** — "what does *it* reduce?" on turn three: which paper is *it*?
2. **Weak evidence** — the top-1 hit cannot answer the question, yet the model still answers.
3. **Unsupported answer** — the answer sounds fine but the citation does not match the source.

They are handled by **NLU reference resolution**, a **Reflect evidence gate**, and per-stage **PipelineTrace**.

## Seven stages

```
Guardrails -> NLU -> Retrieve (BM25 + dense, RRF) -> Rerank
           -> Reflect (second pass / refuse) -> Generate (cited) -> Memory (compact)
```

## Design notes

- **Chunking**: split by Markdown headings, then by paragraph, with a sliding window (320 chars / 80 overlap) for long paragraphs; each chunk keeps `doc_id`, paper title and section name so citations resolve to a section.
- **Tokenization without a dictionary**: CJK character bigrams + ASCII words, dropping particle-led bigrams. Zero dependencies, fully deterministic.
- **Two-way recall + RRF**: BM25 for exact tokens (model names, hyperparameters, numbers), hashing embeddings for overall similarity, fused by Reciprocal Rank Fusion (k=60) — ranks only, so no score-scale calibration is needed.
- **Reflect has two gates**: (1) if query-token coverage against the top-1 chunk is below 0.30, the question is out of domain and is refused without a second pass; (2) otherwise, if the rerank score is below the threshold, a second retrieval is issued with keywords from the top-1 chunk, and a refusal follows if it still fails.
- **Memory**: a topic-anchor index (title tokens + frequent tokens per document) decides whether the new turn already names a document; only when it does not, and a pronoun appears, is the pronoun replaced by that document's keywords rather than its title (titles rarely appear verbatim inside chunks).
- **Compaction**: the last three turns stay verbatim, older ones collapse into `Tn: gist` entries — 72.5% fewer tokens in the measurement below.

## Run

```bash
python s18_paper_rag_agent/code.py            # interactive: /corpus /trace /memory
python s18_paper_rag_agent/code.py --ask "RRF 的平滑常数取多少"
python s18_paper_rag_agent/code.py --no-dense --no-rerank
python s18_paper_rag_agent/eval.py            # deterministic, offline
python -m unittest discover -s tests -p "test_s18*.py" -v
```

## Results

6 papers / 30 chunks; labelled sets: 20 intent, 20 aligned retrieval, 10 paraphrased retrieval, 10 out-of-domain, 20 answerable, 6 four-turn conversations (24 turns).

| Metric | Result |
| --- | --- |
| Intent classification (5 classes) | **85.0%** (17/20) |
| Recall@5, aligned queries | 100% for sparse / dense / hybrid / hybrid+rerank |
| Recall@5, paraphrased queries | sparse **80%** → dense **90%** → hybrid **100%** → hybrid+rerank **100%** |
| Refusal accuracy (10 out-of-domain) | **100%**; false refusal on 20 answerable queries **0%** |
| Pipeline success (24 turns) | **83.3%** |
| Memory consistency (18 follow-up turns) | **77.8%** |
| Token overhead after compaction | **−72.5%** |

Recall saturates on aligned queries — with 30 chunks retrieval is not the bottleneck, and the table says so. The gap only shows up on paraphrased queries that share no literal tokens with the corpus.

## Boundary with real models

The default embedder and reranker are deterministic stand-ins so the whole pipeline is testable offline. Swap them for real models through the same interfaces:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
embedder = type("BgeM3", (), {"embed": lambda self, t: model.encode(t, normalize_embeddings=True).tolist()})()
agent = PaperQAAgent(chunks, embedder=embedder)
```

A cross-encoder reranker only needs `rerank(query, candidates) -> [(chunk, score)]`. Intent classification is rule-based; replacing it with a small classifier touches `stage_nlu` only.
