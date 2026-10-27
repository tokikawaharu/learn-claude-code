#!/usr/bin/env python3
"""
s18_paper_rag_agent/eval.py - 离线可复现评测

四组标注集：
- intent     ：20 条 → 意图五分类准确率
- retrieval  ：20 条 → Recall@5（sparse / dense / hybrid+RRF / hybrid+RRF+rerank）
- boundary   ：10 条越界 + 20 条可答 → 拒答准确率与误拒率
- e2e        ：6 组四轮对话（24 turns）→ 流水线成功率、记忆一致性、改写覆盖率、token 开销

全部使用确定性实现（HashingEmbedder + LexicalCrossReranker），不联网、不下载模型，
clone 后即可复现同一组数字。

用法：
    python s18_paper_rag_agent/eval.py            # 表格输出
    python s18_paper_rag_agent/eval.py --json     # 机器可读
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).resolve().parent


def _load_code_module() -> Any:
    """按路径加载同目录的 code.py（避免与标准库 code 模块重名冲突）。"""
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    spec = importlib.util.spec_from_file_location("s18_code", _HERE / "code.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("无法加载 code.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["s18_code"] = module
    spec.loader.exec_module(module)
    return module


code = _load_code_module()
PaperQAAgent = code.PaperQAAgent
HybridRetriever = code.HybridRetriever
DenseIndex = code.DenseIndex
HashingEmbedder = code.HashingEmbedder
LexicalCrossReranker = code.LexicalCrossReranker
load_corpus = code.load_corpus

# --------------------------------------------------------------------------------------
# 标注集
# --------------------------------------------------------------------------------------

INTENT_SET: list[tuple[str, str]] = [
    ("GRPO 的组内相对优势怎么算", "lookup"),
    ("Clip-Higher 之后策略熵稳定在什么区间", "lookup"),
    ("Dynamic Sampling 能省多少训练步数", "lookup"),
    ("Recall@5 在混合检索下是多少", "lookup"),
    ("RRF 的平滑常数取多少", "lookup"),
    ("对比 GRPO 和 DAPO 的差别", "compare"),
    ("稠密检索和稀疏检索哪个更强", "compare"),
    ("KV Cache 三种压缩策略的区别", "compare"),
    ("总结一下这篇论文讲了什么", "summarize"),
    ("帮我概括一下混合检索的主要结论", "summarize"),
    ("综述一下 agent 编排的七个阶段", "summarize"),
    ("它的拒答准确率提升多少", "followup_ref"),
    ("这个方法在 32K 上下文下表现如何", "followup_ref"),
    ("该论文的训练步数减少了多少", "followup_ref"),
    ("上面提到的阈值是多少", "followup_ref"),
    ("这篇文章的显存占用降低多少", "followup_ref"),
    ("今天天气怎么样", "out_of_scope"),
    ("帮我写一首诗", "out_of_scope"),
    ("推荐一只股票", "out_of_scope"),
    ("晚上吃什么比较好", "out_of_scope"),
]

RETRIEVAL_SET: list[tuple[str, str, str]] = [
    ("组内零方差样本占比从多少升到多少", "grpo_rl", "零方差样本占比从不足 10%"),
    ("Clip-Higher 放宽上截断保护低概率词元探索", "grpo_rl", "保护低概率词元的探索空间"),
    ("GSM8K pass@8 提升多少", "grpo_rl", "pass@8 提升 10 个点"),
    ("规则奖励在解析失败时返回什么", "grpo_rl", "解析失败时返回零奖励"),
    ("Overlong Reward Shaping 对超长样本给软惩罚", "grpo_rl", "对超长截断样本给软惩罚"),
    ("BM25 的 k1 和 b 取值", "hybrid_retrieval", "k1=1.5, b=0.75"),
    ("RRF 只使用排名而不使用原始分数", "hybrid_retrieval", "只使用排名而不使用原始分数"),
    ("交叉编码器重排把 Recall@5 提升到多少", "hybrid_retrieval", "重排提升到 0.72"),
    ("滑动窗口重叠 80 字符", "hybrid_retrieval", "保留 80 字符重叠"),
    ("型号名和超参数为什么需要稀疏召回", "hybrid_retrieval", "这类精确词元难以被稠密向量稳定区分"),
    ("Reflect 阶段什么时候触发二次检索", "agent_harness", "先用 Top1 命中里的关键词扩充查询做二次检索"),
    ("PipelineTrace 分哪两级", "agent_harness", "Turn 级与 Stage 级"),
    ("记忆压缩后 token 开销下降多少", "agent_harness", "平均 token 开销下降 34%"),
    ("意图分成哪五类", "agent_harness", "意图分为五类"),
    ("指代无法消解时怎么处理", "agent_harness", "标记为待澄清而不是猜测"),
    ("KV Cache 混合策略降低多少显存", "kv_cache", "把显存占用降低 41%"),
    ("纯滑出策略为什么检索准确率下降", "kv_cache", "检索准确率下降 9 个点"),
    ("长上下文下每 token 显存占用怎么评测", "kv_cache", "每 token 的显存占用"),
    ("任务类型和压缩策略怎么匹配", "kv_cache", "按任务分布选择策略"),
    ("32K 上下文下的检索准确率保持多少", "kv_cache", "保持在原始模型的 96% 以上"),
]

BOUNDARY_SET: list[str] = [    "今天天气怎么样",
    "帮我写一首关于春天的诗",
    "推荐一只股票",
    "晚上吃什么比较好",
    "把这段话翻译成英文",
    "距离北京最近的机场是哪个",
    "2024 年诺贝尔文学奖得主是谁",
    "怎么申请信用卡",
    "福布斯的富豪榜第一名是谁",
    "帮我订一张去上海的机票",
]

E2E_SET: list[dict[str, Any]] = [
    {"turns": [
        ("GRPO 基线存在什么失效模式", "grpo_rl", "lookup"),
        ("它是怎么发现的", "grpo_rl", "followup_ref"),
        ("那 Clip-Higher 起什么作用", "grpo_rl", "followup_ref"),
        ("这项改进带来多少提升", "grpo_rl", "followup_ref"),
    ]},
    {"turns": [
        ("混合检索这篇论文的主指标是什么", "hybrid_retrieval", "lookup"),
        ("它的融合方式为什么比加权求和稳健", "hybrid_retrieval", "followup_ref"),
        ("重排的候选数控制在多少", "hybrid_retrieval", "followup_ref"),
        ("它的 Recall@5 是多少", "hybrid_retrieval", "followup_ref"),
    ]},
    {"turns": [
        ("Agent 编排那篇论文提出了哪七个阶段", "agent_harness", "lookup"),
        ("它的反思阶段怎么判断证据不足", "agent_harness", "followup_ref"),
        ("它的流水线成功率是多少", "agent_harness", "followup_ref"),
        ("它的记忆压缩策略是什么", "agent_harness", "followup_ref"),
    ]},
    {"turns": [
        ("KV Cache 压缩那篇对比了几种策略", "kv_cache", "lookup"),
        ("它的混合策略降低多少显存", "kv_cache", "followup_ref"),
        ("纯滑出策略的问题在哪", "kv_cache", "followup_ref"),
        ("它建议怎么选策略", "kv_cache", "followup_ref"),
    ]},
    {"turns": [
        ("Dynamic Sampling 省了多少训练步", "grpo_rl", "lookup"),
        ("它筛掉了哪些组", "grpo_rl", "followup_ref"),
        ("这项技巧和基线相比收益多大", "grpo_rl", "followup_ref"),
        ("它的策略熵后来是多少", "grpo_rl", "followup_ref"),
    ]},
    {"turns": [
        ("混合检索里重排用什么模型", "hybrid_retrieval", "lookup"),
        ("它的成本受什么影响", "hybrid_retrieval", "followup_ref"),
        ("在包含数值的查询上哪路召回更重要", "hybrid_retrieval", "followup_ref"),
        ("它的召回结论是多少", "hybrid_retrieval", "followup_ref"),
    ]},
]

# 压力子集：用改写后的口语化说法检索，不共享原文的连续词元，用来暴露不同召回路的差异
PARAPHRASE_SET: list[tuple[str, str]] = [
    ("训练到后面模型越来越不敢尝试新解法", "grpo_rl"),
    ("采样一批里全对或全错等于白跑", "grpo_rl"),
    ("两种召回怎么合到一起更稳", "hybrid_retrieval"),
    ("先粗筛再精排的顺序是怎样的", "hybrid_retrieval"),
    ("答不上来的时候应该直接说不知道", "agent_harness"),
    ("上下文太长怎么省显存", "kv_cache"),
    ("怎么判断一句结论有没有依据", "rag_hallucination"),
    ("小模型压到 4bit 掉点多少", "quantization"),
    ("低秩矩阵取多大合适", "lora_rank"),
    ("每个阶段都留记录方便排查", "agent_harness"),
]

THRESHOLD = 0.35


# --------------------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------------------


def _docs(hits: Sequence[tuple[Any, float]]) -> list[str]:
    return [chunk.doc_id for chunk, _ in hits]


def eval_intent(chunks: Sequence[Any]) -> dict[str, Any]:
    agent = PaperQAAgent(chunks, threshold=THRESHOLD)
    correct = 0
    for query, gold in INTENT_SET:
        if agent.ask(query)["intent"] == gold:
            correct += 1
    return {"accuracy": round(correct / len(INTENT_SET), 4), "n": len(INTENT_SET), "correct": correct}


def eval_retrieval(chunks: Sequence[Any]) -> dict[str, Any]:
    sparse = HybridRetriever(chunks, use_dense=False, use_rerank=False)
    hybrid = HybridRetriever(chunks, use_dense=True, use_rerank=False)
    hybrid_rerank = HybridRetriever(chunks, use_dense=True, use_rerank=True, reranker=LexicalCrossReranker())
    dense = DenseIndex(chunks, HashingEmbedder())

    def recall(finder, dataset) -> float:
        hit = 0
        for item in dataset:
            query, gold_doc = item[0], item[1]
            hits = finder(query)
            if dataset is RETRIEVAL_SET:
                must_contain = item[2]
                if any(chunk.doc_id == gold_doc and must_contain in chunk.text for chunk, _ in hits):
                    hit += 1
            elif any(chunk.doc_id == gold_doc for chunk, _ in hits):
                hit += 1
        return round(hit / len(dataset), 4)

    return {
        "n": len(RETRIEVAL_SET),
        "recall_at_5": {
            "sparse(BM25)": recall(lambda q: sparse.retrieve(q, top_k=5)[0], RETRIEVAL_SET),
            "dense(hash)": recall(lambda q: dense.search(q, limit=5), RETRIEVAL_SET),
            "hybrid+RRF": recall(lambda q: hybrid.retrieve(q, top_k=5)[0], RETRIEVAL_SET),
            "hybrid+RRF+rerank": recall(lambda q: hybrid_rerank.retrieve(q, top_k=5)[0], RETRIEVAL_SET),
        },
        "n_paraphrase": len(PARAPHRASE_SET),
        "recall_at_5_paraphrase": {
            "sparse(BM25)": recall(lambda q: sparse.retrieve(q, top_k=5)[0], PARAPHRASE_SET),
            "dense(hash)": recall(lambda q: dense.search(q, limit=5), PARAPHRASE_SET),
            "hybrid+RRF": recall(lambda q: hybrid.retrieve(q, top_k=5)[0], PARAPHRASE_SET),
            "hybrid+RRF+rerank": recall(lambda q: hybrid_rerank.retrieve(q, top_k=5)[0], PARAPHRASE_SET),
        },
    }


def eval_boundary(chunks: Sequence[Any]) -> dict[str, Any]:
    agent = PaperQAAgent(chunks, threshold=THRESHOLD)
    refused = sum(1 for q in BOUNDARY_SET if agent.ask(q)["refused"])

    agent_answerable = PaperQAAgent(chunks, threshold=THRESHOLD)
    false_refusals = sum(1 for query, _doc, _frag in RETRIEVAL_SET if agent_answerable.ask(query)["refused"])
    return {
        "refusal_accuracy": round(refused / len(BOUNDARY_SET), 4),
        "n_boundary": len(BOUNDARY_SET),
        "false_refusal_rate": round(false_refusals / len(RETRIEVAL_SET), 4),
        "n_answerable": len(RETRIEVAL_SET),
    }


def eval_e2e(chunks: Sequence[Any]) -> dict[str, Any]:
    titles = {chunk.doc_id: chunk.doc_title for chunk in chunks}
    doc_of_title = {title: doc for doc, title in titles.items()}
    agent = PaperQAAgent(chunks, threshold=THRESHOLD)
    total = success = 0
    followups = resolved = 0
    raw_chars = 0

    for conversation in E2E_SET:
        for query, gold_doc, intent in conversation["turns"]:
            result = agent.ask(query)
            trace = agent.state.turns[-1]
            nlu = next((step.detail for step in trace.stages if step.stage == "NLU"), {})
            total += 1
            raw_chars += len(query) + len(result["answer"])  # 不压缩时需要保留的全部上下文

            cited_doc = doc_of_title.get(result["citations"][0].split(" § ")[0], "") if result["citations"] else ""
            if not result["refused"] and cited_doc == gold_doc:
                success += 1

            if intent == "followup_ref":
                followups += 1
                # 记忆一致性：追问没有跑偏（同一篇文档），且代词被消解或主题被显式识别
                resolved_reference = bool(nlu.get("resolved_entity")) or nlu.get("topic") == gold_doc
                if not result["refused"] and cited_doc == gold_doc and resolved_reference:
                    resolved += 1

    memory_chars = sum(len(entry.text) for entry in agent.state.memory)
    return {
        "n_turns": total,
        "pipeline_success_rate": round(success / max(1, total), 4),
        "n_followup_turns": followups,
        "memory_consistency": round(resolved / max(1, followups), 4),
        "token_overhead_reduction": round(max(0.0, 1 - memory_chars / max(1, raw_chars)), 4),
    }


# --------------------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------------------


def run_eval(corpus_dir: str | Path) -> dict[str, Any]:
    chunks = load_corpus(corpus_dir)
    return {
        "corpus_chunks": len(chunks),
        "intent": eval_intent(chunks),
        "retrieval": eval_retrieval(chunks),
        "boundary": eval_boundary(chunks),
        "e2e": eval_e2e(chunks),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="s18 论文问答 Agent 离线评测")
    parser.add_argument("--corpus", default=str(_HERE / "corpus"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = run_eval(args.corpus)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"语料块数：{report['corpus_chunks']}")
    intent = report["intent"]
    print(f"\n[意图] accuracy={intent['accuracy']:.3f}  ({intent['correct']}/{intent['n']})")
    print("\n[检索] Recall@5")
    for name, value in report["retrieval"]["recall_at_5"].items():
        print(f"  {name:<20} {value:.3f}")
    boundary = report["boundary"]
    print(f"\n[拒答] refusal={boundary['refusal_accuracy']:.3f} ({boundary['n_boundary']} 越界)  "
          f"false_refusal={boundary['false_refusal_rate']:.3f} ({boundary['n_answerable']} 可答)")
    e2e = report["e2e"]
    print(f"\n[端到端] pipeline_success={e2e['pipeline_success_rate']:.3f} ({e2e['n_turns']} turns)  "
          f"memory_consistency={e2e['memory_consistency']:.3f} ({e2e['n_followup_turns']} 追问)  "
          f"token_reduction={e2e['token_overhead_reduction']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
