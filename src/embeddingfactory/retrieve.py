"""检索执行与评测。

这一步是**真的检索**，不是「量一个属性」：每条查询都会走一遍
「嵌入 → 索引搜索 → chunk 归并回 doc → 排序」，然后与 ground truth 比对。
召回率是**任务指标**，不是向量空间的性质 —— 这是本项目与
TokenizerFactory（量的是编码属性）最本质的区别。

chunk → doc 的归并
------------------
索引返回的是 **chunk**，ground truth 是 **doc**。归并规则：

1. 按 chunk 的排名顺序遍历，取每个 chunk 的 ``doc_id``；
2. 同一篇文档只保留**第一次出现**的位置（它代表该文档的最好名次）；
3. 截断到 top-k。

这条规则必须显式写出来，因为它是「分块策略会改变召回率」这个结论的**全部来源**：
块切得小 → 单块更精确 → 但一条答案被切散后可能没有任何一块进得了前 ``fetch_k``；
块切得大 → 一块就能覆盖整条答案 → 但块内噪声多、排序变差。

``fetch_k`` 这个口径
--------------------
索引一次返回 ``fetch_k`` 个 chunk（``fetch_k = k × fetch_factor``，默认因子 2），
再归并成 top-k 文档。
**为什么不直接返回 k 个 chunk？** 因为一篇文档有多个块，只取 k 个 chunk
很可能只覆盖 2~3 篇文档，top-k 文档根本填不满 —— 那会把所有配置的召回率
一起压低到无法区分。``fetch_k`` 是「检索器愿意吐多少个候选」，
它是**检索空间的一个真实参数**（工业界叫 ``top_k`` / ``rerank depth``），
所以它进配置、进报告，而不是藏在代码里。
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .embed import EmbeddingModel
from .index import BaseIndex
from .schema import Query


@dataclass
class QueryOutcome:
    """一条查询的结果。"""

    query_id: str
    text: str
    #: 归并后的 top-k 文档（按名次）
    ranked_docs: list[str]
    relevant: list[str]
    n_distance_computations: int
    latency_ms: float
    recall: float
    ndcg: float
    reciprocal_rank: float
    first_relevant_rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "n_distance_computations": self.n_distance_computations,
            "latency_ms": round(self.latency_ms, 4),
            "recall": round(self.recall, 6),
            "ndcg": round(self.ndcg, 6),
            "reciprocal_rank": round(self.reciprocal_rank, 6),
            "first_relevant_rank": self.first_relevant_rank,
            "ranked_docs": list(self.ranked_docs),
        }


@dataclass
class RetrievalOutcome:
    """一次「索引 + 全部查询」的完整产物。"""

    outcomes: list[QueryOutcome] = field(default_factory=list)
    #: chunk 总数（成本模型的分母口径之一）
    n_chunks: int = 0
    k: int = 0
    fetch_k: int = 0

    # -- 聚合 --------------------------------------------------------
    @property
    def mean_recall(self) -> float:
        return _mean(o.recall for o in self.outcomes)

    @property
    def mean_ndcg(self) -> float:
        return _mean(o.ndcg for o in self.outcomes)

    @property
    def mrr(self) -> float:
        return _mean(o.reciprocal_rank for o in self.outcomes)

    @property
    def mean_distance_computations(self) -> float:
        return _mean(o.n_distance_computations for o in self.outcomes)

    @property
    def mean_latency_ms(self) -> float:
        return _mean(o.latency_ms for o in self.outcomes)

    @property
    def n_zero_recall(self) -> int:
        return sum(1 for o in self.outcomes if o.recall == 0.0)

    @property
    def mean_first_relevant_rank(self) -> float:
        ranks = [o.first_relevant_rank for o in self.outcomes if o.first_relevant_rank]
        return _mean(ranks) if ranks else 0.0

    def worst(self, n: int = 3) -> list[QueryOutcome]:
        return sorted(self.outcomes, key=lambda o: (o.recall, o.query_id))[:n]

    def best(self, n: int = 3) -> list[QueryOutcome]:
        return sorted(self.outcomes, key=lambda o: (-o.recall, o.query_id))[:n]


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
def recall_at_k(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranked) & set(relevant))
    return hits / len(set(relevant))


def dcg_at_k(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    return sum(
        1.0 / math.log2(i + 2) for i, doc in enumerate(ranked) if doc in rel
    )


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """二值相关的 nDCG@k。

    IDCG 的分母口径：理想排序是「全部相关文档排在最前」，但**上限是
    ``min(k, |relevant|)``** —— 相关文档比 k 还多时，理想情况也只能放 k 个。
    不用这个上限会让 nDCG 永远小于 1，读者会以为排序还有改进空间。
    """
    if not relevant:
        return 0.0
    ideal = min(k, len(set(relevant)))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked, relevant) / idcg


def first_relevant_rank(ranked: Sequence[str], relevant: Sequence[str]) -> int:
    rel = set(relevant)
    for i, doc in enumerate(ranked):
        if doc in rel:
            return i + 1
    return 0


def merge_to_docs(
    chunk_ids: Sequence[int], chunk_doc_ids: Sequence[str], k: int
) -> list[str]:
    """把 chunk 排名归并成文档排名（同一文档只保留最好名次）。"""
    out: list[str] = []
    seen: set[str] = set()
    for idx in chunk_ids:
        doc = chunk_doc_ids[idx]
        if doc in seen:
            continue
        seen.add(doc)
        out.append(doc)
        if len(out) >= k:
            break
    return out


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def evaluate(
    index: BaseIndex,
    model: EmbeddingModel,
    queries: Sequence[Query],
    *,
    chunk_doc_ids: Sequence[str],
    k: int,
    fetch_k: int,
    latency_repeats: int,
) -> RetrievalOutcome:
    """跑全部查询并计算指标。

    ``latency_repeats`` 轮取**中位数**：单轮会被 CPU 降频与调度噪声带偏。
    但中位数只压小噪声，消不掉它 —— 所以延迟只作验证，不作主前沿的轴
    （见 index.py 模块文档）。
    """
    outcome = RetrievalOutcome(n_chunks=len(chunk_doc_ids), k=k, fetch_k=fetch_k)
    if not queries:
        return outcome

    for q in queries:
        qvec = model.transform(q.text)
        # 第一次搜索取**确定性的成本模型**（每次结果都相同）
        result = index.search(qvec, fetch_k)
        ranked = merge_to_docs(result.ids, chunk_doc_ids, k)

        # 延迟：多轮取中位数。结果不参与指标，只用于报告里的实测验证。
        samples: list[float] = []
        for _ in range(max(1, latency_repeats)):
            start = time.perf_counter()
            index.search(qvec, fetch_k)
            samples.append((time.perf_counter() - start) * 1000.0)

        rr_rank = first_relevant_rank(ranked, q.relevant)
        outcome.outcomes.append(
            QueryOutcome(
                query_id=q.query_id,
                text=q.text,
                ranked_docs=ranked,
                relevant=list(q.relevant),
                n_distance_computations=result.n_distance_computations,
                latency_ms=statistics.median(samples),
                recall=recall_at_k(ranked, q.relevant),
                ndcg=ndcg_at_k(ranked, q.relevant, k),
                reciprocal_rank=(1.0 / rr_rank) if rr_rank else 0.0,
                first_relevant_rank=rr_rank,
            )
        )
    return outcome


def sample_of(outcome: RetrievalOutcome, n: int) -> list[tuple[str, str, list[tuple[str, int, bool]]]]:
    """取前 ``n`` 条查询的样本（供 explain / query 使用）。

    形如 ``[(query_id, 查询文本, [(doc_id, 名次, 是否相关), ...]), ...]`` ——
    把「是否相关」一起存下来，报告里就能直接标出「第 3 名开始才命中」。
    """
    out: list[tuple[str, str, list[tuple[str, int, bool]]]] = []
    for o in outcome.outcomes[: max(0, n)]:
        rel = set(o.relevant)
        out.append(
            (
                o.query_id,
                o.text,
                [(doc, i + 1, doc in rel) for i, doc in enumerate(o.ranked_docs)],
            )
        )
    return out
