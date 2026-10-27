# s18: 论文库多轮检索问答 Agent（领域化二开）

[中文](README.zh.md) · [English](README.md)

`s01` → … → `s17` → **`s18`**

> *本节把前 17 章的 harness 思路（agent loop / context compact / memory）落到一个真实的领域问答场景：科研论文库的多轮追问。*

- **二次开发声明**：本节基于上游 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 的教学骨架，新增独立章节 `s18_paper_rag_agent/`，不修改任何既有章节的代码与 README。
- **不依赖模型即可运行**：全部实现基于 Python 标准库，默认使用确定性 Embedder / Reranker，clone 后直接 `python s18_paper_rag_agent/eval.py` 就能复现下面表格里的每一个数字。

---

## 问题

论文库问答的失败通常不是「模型不够强」，而是链路问题：

1. **指代缺失**：第三轮问「它的混合策略降低多少显存」时，「它」指的是哪篇论文？
2. **检索证据不足**：Top1 命中其实答不上这个问题，但模型仍然会编一段答案。
3. **无依据作答**：答案看起来合理，引用却对不上原文，错了也查不出来。

本节把这三件事分别交给 **NLU 的指代消解**、**Reflect 的证据反思**、**PipelineTrace 的逐阶段留痕**。

## 七阶段编排

```
user turn
   │
   ├─ Guardrails  空输入 / 超长输入的拦截
   ├─ NLU         意图五分类 + 指代消解 + Query 改写
   ├─ Retrieve    BM25 稀疏召回 ＋ 向量稠密召回 → RRF 融合
   ├─ Rerank      交叉打分重排，取 Top-K 进生成
   ├─ Reflect     证据是否充分？不足 → 二次检索；仍不足 → 拒答
   ├─ Generate    引用式回答（离线模板 / 可换 LLM）
   └─ Memory      会话记忆更新 + 按需压缩
```

## 关键设计

**语料与分块**：按 Markdown 标题切章节，章节内按段落切块，超长段落用滑动窗口（320 字符窗口 / 80 字符重叠）切分，块上保留 `doc_id`、论文标题、章节名，保证引用可以回溯到具体章节。

**中文检索不依赖分词库**：中文按字符二元组 + 英文按词切分，去掉助词开头的噪声二元组。好处是零依赖、结果确定；代价是没有词形还原与同义扩展。

**双路召回 + RRF**：BM25（k1=1.5, b=0.75）负责精确词元（型号名、超参数、数值），向量召回负责整体语义；两路各取 8 条候选后用 RRF（k=60）融合——只用排名不用分数，避免不同打分器量纲不一致。

**Reflect 的两道闸门**：
1. 查询词元在 Top1 命中里的覆盖率 < 0.30 → 判定为域外问题，直接拒答（不做二次检索，避免「把问题硬凑到某段文本上」）；
2. 覆盖率达标但重排分数 < 阈值 → 用 Top1 命中的关键词扩充查询做二次检索，仍不达标则拒答。

**记忆与指代消解**：每个文档在启动时建立「主题锚点表」（标题词元 + 该篇高频词元 → doc_id）。新一轮先做主题判定：句子本身指向某篇文档时，绝不用上一轮的实体去污染它；只有在没有主题锚点、且出现代词时，才用该篇的代表关键词（而不是标题）替换代词——标题里的词往往不出现在正文块中，直接替换会让检索覆盖率崩掉。

**记忆压缩**：最近 3 轮保留原文，更早的折叠为 `Tn: 摘要` 形式的 gist，实测 token 开销下降 72.5%，记忆一致性维持在 77.8%。

## 运行

```bash
# 交互（命令：/ask 直接提问、/corpus、/trace、/memory）
python s18_paper_rag_agent/code.py

# 单轮
python s18_paper_rag_agent/code.py --ask "RRF 的平滑常数取多少"

# 只用 BM25（关闭稠密召回）/ 关闭重排
python s18_paper_rag_agent/code.py --no-dense --no-rerank

# 评测（确定性，无需网络）
python s18_paper_rag_agent/eval.py
python s18_paper_rag_agent/eval.py --json

# 单测
python -m unittest discover -s tests -p "test_s18*.py" -v
```

## 评测结果

语料 6 篇论文 / 30 个块；标注集：意图 20 条、检索 20 条（对齐问法）+ 10 条（改写问法）、越界 10 条、可答 20 条、端到端 6 组四轮对话（24 turns）。

| 指标 | 结果 |
| --- | --- |
| 意图五分类准确率 | **85.0%**（17/20） |
| Recall@5（对齐问法） | sparse / dense / hybrid / hybrid+rerank 均 **100%** |
| Recall@5（改写问法，压力子集） | sparse **80%** → dense **90%** → hybrid **100%** → hybrid+rerank **100%** |
| 拒答准确率（10 条越界） | **100%**；可答集误拒 **0%**（20 条） |
| Pipeline 成功率（24 turns） | **83.3%** |
| 记忆一致性（18 追问） | **77.8%** |
| 记忆压缩后 token 开销 | **−72.5%** |

对齐问法上 Recall@5 饱和（30 块的语料里检索不是瓶颈，如实记录）。把问法改写成不共享原文词元的口语表述后差距才出现：混合检索比纯稀疏 +20pts、比纯稠密 +10pts。复现命令见上，数字是确定性的，两次运行必须完全一致（单测里有断言）。

## 与真实模型的边界

- 默认 `HashingEmbedder` 只做字符/词 n-gram 哈希投影，**没有语义泛化能力**；它的作用是让整套链路可以在离线、无 GPU、无 API Key 的情况下被评测。生产环境把 `BGE-M3` 接进来即可（`HybridRetriever(embedder=...)` 接口已留好）：
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("BAAI/bge-m3")
  embedder = type("BgeM3", (), {"embed": lambda self, text: model.encode(text, normalize_embeddings=True).tolist()})()
  agent = PaperQAAgent(chunks, embedder=embedder)
  ```
- 默认 `LexicalCrossReranker` 是词法交叉打分（覆盖度 + 邻近度 + 章节命中），同样是为了可复现；换成 `bge-reranker` 只需实现 `rerank(query, candidates) -> [(chunk, score)]`。
- 意图分类是规则版（关键词 + 代词表）。若要换成小模型分类器，替换 `stage_nlu` 里的意图判定即可，其余阶段不感知。
- 越界判定同时依赖「意图关键词」与「覆盖率闸门」，后者才是主力——前者只是省掉一次检索。

## 文件

```
s18_paper_rag_agent/
├── code.py                  # 七阶段 Agent + CLI（/corpus /trace /memory）
├── eval.py                  # 四组标注集 + 四路召回对比，确定性可复现
├── corpus/                  # 6 篇合成论文（2 篇主项目 + 4 篇干扰/领域相关）
└── README.md / README.zh.md
tests/test_s18_paper_rag_agent.py   # 24 个单测：切块/分词/BM25/RRF/重排/拒答/记忆/评测
```

## 与上游的关系

上游是教学仓库：每章只讲一个机制，刻意保持最小实现。本节不改变这个取向，只是把这些机制拼成一个可以被评测的完整链路，因此故意保留了「不引入外部依赖、不写框架」的约束。真正要上生产，需要补的是：真实 Embedder/Reranker、语料增量索引、并发与缓存、以及更细的权限与审计。
