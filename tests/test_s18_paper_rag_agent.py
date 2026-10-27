"""s18_paper_rag_agent 的单元测试（纯标准库 + unittest，无需网络与模型）。

覆盖：语料切块与标题解析、中文分词、BM25 排序、RRF 融合、rerank 排序、
Guardrails、域外拒答、指代消解、记忆压缩、端到端流水线、评测指标可复算。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parent / "s18_paper_rag_agent"


def _load(name: str, filename: str):
    if str(CHAPTER) not in sys.path:
        sys.path.insert(0, str(CHAPTER))
    spec = importlib.util.spec_from_file_location(name, CHAPTER / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


code = _load("s18_code_under_test", "code.py")
ev = _load("s18_eval_under_test", "eval.py")


class CorpusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = code.load_corpus(CHAPTER / "corpus")

    def test_corpus_loaded(self) -> None:
        self.assertGreaterEqual(len(self.chunks), 20)

    def test_first_heading_becomes_title(self) -> None:
        titles = {c.doc_id: c.doc_title for c in self.chunks}
        self.assertEqual(titles["grpo_rl"], "GRPO 与 DAPO 在数学推理强化学习中的对比研究")
        self.assertEqual(titles["kv_cache"], "长上下文推理中的 KV Cache 压缩")

    def test_chunks_carry_section_metadata(self) -> None:
        sections = {c.section for c in self.chunks}
        self.assertIn("摘要", sections)
        self.assertNotIn("正文", sections - {"正文"})

    def test_sliding_window_respects_max(self) -> None:
        long_text = "策略" * 400
        windows = code._sliding_window(long_text, max_chars=100, overlap=20)
        self.assertGreater(len(windows), 1)
        self.assertTrue(all(len(w) <= 100 for w in windows))

    def test_missing_corpus_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            code.load_corpus(CHAPTER / "not-exists")


class TokenizerTest(unittest.TestCase):
    def test_cjk_bigram_and_ascii(self) -> None:
        tokens = code.tokenize("GRPO 的策略熵坍缩")
        self.assertIn("grpo", tokens)
        self.assertIn("策略", tokens)
        self.assertIn("坍缩", tokens)

    def test_stopword_bigram_dropped(self) -> None:
        self.assertNotIn("的策", code.tokenize("的策略"))


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = code.load_corpus(CHAPTER / "corpus")
        self.sparse = code.BM25Index(self.chunks)
        self.dense = code.DenseIndex(self.chunks, code.HashingEmbedder())

    def test_bm25_hits_exact_term_chunk(self) -> None:
        hits = self.sparse.search("k1=1.5, b=0.75", limit=3)
        self.assertEqual(hits[0][0].doc_id, "hybrid_retrieval")

    def test_dense_returns_scores_desc(self) -> None:
        hits = self.dense.search("RRF 融合", limit=5)
        scores = [score for _chunk, score in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rrf_prefers_chunk_ranked_by_both_paths(self) -> None:
        a, b, c = self.chunks[0], self.chunks[1], self.chunks[2]
        fused = code.rrf_fuse([[(a, 1.0), (b, 0.9)], [(b, 0.8), (c, 0.7)]])
        self.assertEqual(fused[0][0].chunk_id, b.chunk_id)

    def test_reranker_pushes_full_coverage_first(self) -> None:
        query = "RRF 只使用排名而不使用原始分数"
        reranker = code.LexicalCrossReranker()
        candidates = [(chunk, 0.0) for chunk in self.chunks]
        top = reranker.rerank(query, candidates, limit=1)[0][0]
        self.assertIn("RRF", top.text)

    def test_hybrid_mode_switch(self) -> None:
        hybrid = code.HybridRetriever(self.chunks, use_dense=True, use_rerank=True,
                                     reranker=code.LexicalCrossReranker())
        _hits, debug = hybrid.retrieve("拒答准确率提升多少", top_k=3)
        self.assertIn("sparse", debug["mode"])
        self.assertIn("dense", debug["mode"])
        self.assertIn("rerank", debug["mode"])

        sparse_only = code.HybridRetriever(self.chunks, use_dense=False, use_rerank=False)
        _hits, debug2 = sparse_only.retrieve("拒答准确率提升多少", top_k=3)
        self.assertEqual(debug2["mode"], ["sparse"])


class StageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = code.load_corpus(CHAPTER / "corpus")
        self.agent = code.PaperQAAgent(self.chunks, threshold=ev.THRESHOLD)

    def test_guardrails_reject_empty_and_long(self) -> None:
        self.assertFalse(code.stage_guardrails("  ")[0])
        self.assertFalse(code.stage_guardrails("问" * 500)[0])
        self.assertTrue(code.stage_guardrails("正常问题")[0])

    def test_out_of_domain_refused_without_calling_generate(self) -> None:
        result = self.agent.ask("帮我订一张去上海的机票")
        self.assertTrue(result["refused"])
        self.assertEqual(result["citations"], [])

    def test_answerable_question_not_refused(self) -> None:
        result = self.agent.ask("BM25 的 k1 和 b 取值")
        self.assertFalse(result["refused"])
        self.assertTrue(result["citations"])

    def test_trace_has_seven_stage_shape(self) -> None:
        self.agent.ask("RRF 的平滑常数取多少")
        stages = [step.stage for step in self.agent.state.turns[-1].stages]
        self.assertEqual(stages, ["Guardrails", "NLU", "Retrieve", "Reflect", "Generate", "Memory"])

    def test_pronoun_resolution_uses_memory(self) -> None:
        self.agent.ask("KV Cache 压缩那篇对比了几种策略")
        result = self.agent.ask("它的混合策略降低多少显存")
        self.assertEqual(result["intent"], "followup_ref")
        self.assertNotEqual(result["rewritten"], "它的混合策略降低多少显存")

    def test_topic_switch_is_not_polluted_by_memory(self) -> None:
        self.agent.ask("KV Cache 压缩那篇对比了几种策略")
        result = self.agent.ask("混合检索这篇论文的主指标是什么")
        self.assertIn("主指标", result["rewritten"])
        self.assertNotIn("KV", result["rewritten"])

    def test_memory_compaction_keeps_recent_verbatim(self) -> None:
        for query in ["RRF 的平滑常数取多少", "BM25 的 k1 和 b 取值",
                      "意图分成哪五类", "PipelineTrace 分哪两级", "32K 上下文下的检索准确率保持多少"]:
            self.agent.ask(query)
        kinds = [entry.kind for entry in self.agent.state.memory]
        self.assertEqual(kinds.count("verbatim"), self.agent.state.max_verbatim_turns)
        self.assertIn("gist", kinds)

    def test_render_helpers_do_not_crash(self) -> None:
        self.agent.ask("意图分成哪五类")
        self.assertIn("Turn 1", self.agent.render_trace())
        self.assertIn("T1", self.agent.render_memory())
        self.assertIn("语料块数", self.agent.render_corpus())


class EvalTest(unittest.TestCase):
    """评测必须确定性：同一份语料两次运行结果完全一致。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = ev.run_eval(CHAPTER / "corpus")

    def test_report_keys(self) -> None:
        for key in ("corpus_chunks", "intent", "retrieval", "boundary", "e2e"):
            self.assertIn(key, self.report)

    def test_metrics_in_range(self) -> None:
        self.assertGreaterEqual(self.report["intent"]["accuracy"], 0.6)
        self.assertLessEqual(self.report["boundary"]["false_refusal_rate"], 0.2)
        self.assertGreaterEqual(self.report["boundary"]["refusal_accuracy"], 0.8)
        self.assertGreaterEqual(self.report["e2e"]["pipeline_success_rate"], 0.6)
        self.assertGreater(self.report["e2e"]["token_overhead_reduction"], 0.3)

    def test_hybrid_not_worse_than_single_path_on_paraphrase(self) -> None:
        para = self.report["retrieval"]["recall_at_5_paraphrase"]
        self.assertGreaterEqual(para["hybrid+RRF"], para["sparse(BM25)"])
        self.assertGreaterEqual(para["hybrid+RRF"], para["dense(hash)"])

    def test_deterministic(self) -> None:
        again = ev.run_eval(CHAPTER / "corpus")
        self.assertEqual(again, self.report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
