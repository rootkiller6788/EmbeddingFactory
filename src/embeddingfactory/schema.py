"""核心对象。

本项目的形状是**「扇出-收拢型」**（与 `TokenizerFactory` 同族，见
python-pipeline-scaffold skill 的「扇出-收拢型的额外约定」一节）：

    Documents → Chunk(×策略) → Embed(×模型) → Index(×类型×距离×搜索参数)
        → Query → Retrieve → Evaluate → Compare → Select → Export

一个语料扇出成 `embedding × chunking × index × distance × search` 条分支，
每条分支**真建一个索引、真跑一遍查询**，最后收拢成：
一份**消融矩阵** + 一条 **Latency–Recall 前沿** + 一个可导出的检索组件。

与 TokenizerFactory 的三点区别（这不是「换个词再说一遍」，是结构性的）
--------------------------------------------------------------------

1. **本项目有两个空间，而且必须分开**（README 第五节的原话）。
   ``RepresentationSpec``（embedding + chunking）决定**召回率的上限**；
   ``RetrievalSpec``（index + distance + search）决定**达到该上限要花多少成本**。
   于是本项目的核心对象也分成这两组，而不是摊成一堆平行字段 ——
   因为「换 embedding 没用但换索引有用」和「换 embedding 有用」是两种
   完全不同的结论，字段混在一起就说不清了。

2. **多了一个「查询集」**。分词器项目只需要「度量」，本项目需要**真的检索**：
   有 query、有 ground-truth 相关文档、有 top-k。于是「评测」这一步
   不再是「量一个属性」，而是「跑一个任务」—— 召回率是任务指标，不是属性。

3. **延迟是实测值，而实测值不可逐位复现**（TokenizerFactory 踩过这个坑）。
   本项目**主前沿刻意用「距离计算次数」当成本轴**：它完全由索引结构决定，
   逐位可复现。实测延迟另算一条副前沿，用来验证「成本模型确实与延迟同向」。
   这是本项目对「可复现的量化结果」这条验收标准的正面回应。

三层对象
--------

    CorpusStats       语料画像（文档数 / 分块数 / 查询数 / 主题分布）—— 「问题是什么」
    RetrievalSpec     一个检索配置 = 表示空间 × 检索空间
    RetrievalRun      一个 spec 建索引 + 跑查询的完整结果（指标 + 逐查询样本）
    RetrievalResult   整条流水线的产物：全部 run + 矩阵 + 前沿 + 消融 + 归因 + 版本
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: 参与 digest 的代码版本。口径变了就改这个值，否则新旧结果会被当成同一批。
CODE_VERSION = "embeddingfactory/0.1.0"

#: 表示空间的两个轴（决定召回率上限）
REPRESENTATION_AXES: tuple[str, ...] = ("embedding", "chunking")

#: 检索空间的三（四）个轴（决定达到上限的成本）
RETRIEVAL_AXES: tuple[str, ...] = ("index", "distance", "search")

#: 全部设计轴，顺序 = 展开顺序 = 报告里的列顺序
DESIGN_AXES: tuple[str, ...] = REPRESENTATION_AXES + RETRIEVAL_AXES

#: 搜索参数的短标签。它们会进 spec_id（进而进表格与 DAG 节点 id），
#: 所以必须短 —— 全称会让每一行多出十几个字符。
_SEARCH_ABBR: dict[str, str] = {"nprobe": "np", "ef_search": "ef"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: Any, n: int = 12) -> str:
    """稳定短 id。

    **不用内置 ``hash()``** —— 它按 ``PYTHONHASHSEED`` 加盐，跨进程不稳定。
    """
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:n]


# ----------------------------------------------------------------------
# 语料
# ----------------------------------------------------------------------
@dataclass
class Document:
    """一篇原始文档。"""

    doc_id: str
    title: str
    text: str
    topic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "topic": self.topic,
            "n_chars": len(self.text),
        }


@dataclass
class DocChunk:
    """分块后的一个检索单元。

    ``chunk_id`` 形如 ``{doc_id}#{i}``，**必须稳定**：它进 DAG 节点、
    进逐查询样本、进 explain 的输出。不稳定的 chunk_id 会让
    「同一次查询在两次运行里返回了不同的 id」这种假 bug 反复出现。
    """

    chunk_id: str
    doc_id: str
    text: str
    index_in_doc: int
    #: 这个块是不是「被硬切开」的（fixed 策略切断了句子 / sentence 策略超长被截）
    cut: bool = False

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "index_in_doc": self.index_in_doc,
            "n_chars": self.n_chars,
            "cut": self.cut,
            "text": self.text,
        }


@dataclass
class Query:
    """一条查询 + ground truth。

    ``relevant`` 是**文档级**相关集。检索返回的是 chunk，评测前要把
    chunk 归并回 doc —— 这是 doc 级评测的标准做法，也是本项目
    「chunk 大小会改变召回率」这个结论能成立的前提。
    """

    query_id: str
    text: str
    relevant: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "relevant": list(self.relevant),
        }


@dataclass
class CorpusStats:
    """语料画像。它决定「该关注哪个维度」。"""

    name: str
    n_docs: int
    n_chunks: int
    n_queries: int
    n_topics: int
    mean_chunk_chars: float
    mean_docs_per_query: float
    #: 主题 → 文档数
    topic_sizes: dict[str, int] = field(default_factory=dict)
    #: 分块策略 → (块数, 被硬切开的块数)
    chunking: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_docs": self.n_docs,
            "n_chunks": self.n_chunks,
            "n_queries": self.n_queries,
            "n_topics": self.n_topics,
            "mean_chunk_chars": round(self.mean_chunk_chars, 3),
            "mean_docs_per_query": round(self.mean_docs_per_query, 3),
            "topic_sizes": {k: v for k, v in sorted(self.topic_sizes.items())},
            "chunking": {k: dict(sorted(v.items())) for k, v in sorted(self.chunking.items())},
        }


# ----------------------------------------------------------------------
# 检索配置 —— 两个空间
# ----------------------------------------------------------------------
@dataclass
class RetrievalSpec:
    """一个检索配置。

    **字段按空间分组**（这是刻意的）：``embedding`` / ``chunking`` 属于表示空间，
    ``index`` / ``distance`` / ``search`` 属于检索空间。消融表与空间归因
    直接按这两组切分，不需要在报告层再猜哪个字段属于哪个空间。
    """

    # -- 表示空间 --
    embedding: str
    chunking: str
    # -- 检索空间 --
    index: str
    distance: str
    #: 搜索参数（ivf 的 nprobe / hnsw 的 ef_search）。flat 用 ``{}``。
    search: dict[str, Any] = field(default_factory=dict)

    @property
    def search_tag(self) -> str:
        """搜索参数的可读短标签。flat 没有参数，固定为 ``exact``。

        用 ``np`` / ``ef`` 缩写而不是 ``nprobe`` / ``ef_search``：
        这两个名字会进 spec_id，而 spec_id 又要进表格与 DAG 节点 id，
        全称会让每行多出 12 个字符，表格立刻读不动。
        """
        if not self.search:
            return "exact"
        return "-".join(
            f"{_SEARCH_ABBR.get(k, k)}{v}" for k, v in sorted(self.search.items())
        )

    @property
    def spec_id(self) -> str:
        """稳定且唯一的短标识。DAG 节点 id / 矩阵行名 / 前沿标签都用它。

        形如 ``tfidf-sent-ivf-cos-nprobe2``。四段固定（embedding / chunking /
        index / distance），第五段只在索引真的有参数时出现 ——
        这样 ``flat`` 的 spec_id 不会多出一个恒为 ``exact`` 的尾巴。
        """
        base = f"{self.embedding}-{self.chunking}-{self.index}-{self.distance}"
        return base if not self.search else f"{base}-{self.search_tag}"

    def axis(self, name: str) -> str:
        """取某个设计轴上的取值（消融与归因用）。"""
        if name == "search":
            return self.search_tag
        return str(getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "embedding": self.embedding,
            "chunking": self.chunking,
            "index": self.index,
            "distance": self.distance,
            "search": {k: self.search[k] for k in sorted(self.search)},
            "search_tag": self.search_tag,
            # 空间归属 —— 报告里直接读，不在报告层重新判断
            "representation": {
                "embedding": self.embedding,
                "chunking": self.chunking,
            },
            "retrieval": {
                "index": self.index,
                "distance": self.distance,
                "search": self.search_tag,
            },
        }

    @property
    def digest(self) -> str:
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True))


# ----------------------------------------------------------------------
# 度量
# ----------------------------------------------------------------------
@dataclass
class RetrievalMetrics:
    """检索度量。

    | 维度 | 字段 | 方向 | 为什么单独量它 |
    | --- | --- | --- | --- |
    | 质量 | ``recall_at_k`` | 越大越好 | 检索的任务指标（README 的 Recall） |
    | 排序 | ``ndcg_at_k`` | 越大越好 | 相关文档排在前面才算好（README 的 nDCG） |
    | 排序 | ``mrr`` | 越大越好 | 第一条相关结果排多前（README 的 MRR） |
    | 成本 | ``n_distance_computations`` | 越小越好 | **确定性成本模型**，替代不可复现的延迟 |
    | 成本 | ``query_latency_ms`` | 越小越好 | 实测延迟（不可逐位复现，仅作验证） |
    | 成本 | ``build_seconds`` | 越小越好 | 建索引耗时（同上） |
    | 成本 | ``index_bytes`` | 越小越好 | 索引内存占用（README 的 RAM） |

    ``n_distance_computations`` 是**每个查询平均算了多少次距离**。
    它是本项目的关键设计：实测延迟受 CPU 降频与调度影响，两次运行不一致；
    而「扫了多少个向量」完全由索引结构与搜索参数决定，逐位可复现。
    两者一起报，才能既给出可复现的前沿、又证明成本模型确实与真实延迟同向。
    """

    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    n_distance_computations: float
    query_latency_ms: float
    build_seconds: float
    index_bytes: int
    #: 原始计数，报告里要能对回现实
    n_queries: int
    n_chunks: int
    k: int
    #: 有相关文档但一个都没召回的查询数 —— 比均值更能暴露「某类查询彻底失败」
    n_queries_zero_recall: int = 0
    #: 平均第一条相关结果的排名（1 起；没召回到则为 0）
    mean_first_relevant_rank: float = 0.0

    def get(self, name: str) -> float:
        if not hasattr(self, name):
            raise KeyError(f"未知指标 {name!r}")
        return float(getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall_at_k": round(self.recall_at_k, 6),
            "ndcg_at_k": round(self.ndcg_at_k, 6),
            "mrr": round(self.mrr, 6),
            "n_distance_computations": round(self.n_distance_computations, 3),
            "query_latency_ms": round(self.query_latency_ms, 4),
            "build_seconds": round(self.build_seconds, 5),
            "index_bytes": self.index_bytes,
            "n_queries": self.n_queries,
            "n_chunks": self.n_chunks,
            "k": self.k,
            "n_queries_zero_recall": self.n_queries_zero_recall,
            "mean_first_relevant_rank": round(self.mean_first_relevant_rank, 4),
        }


# ----------------------------------------------------------------------
# 一次索引 + 评测
# ----------------------------------------------------------------------
@dataclass
class RetrievalRun:
    """一个 spec 的完整结果。

    ``sample`` 保存前若干个查询的检索结果 —— 它是 ``explain`` 与
    ``query``（CLI）的数据来源。**必须在检索时就存下来**，
    因为建好的索引对象不落盘（只落摘要），事后重建就要重建索引。
    """

    spec: RetrievalSpec
    metrics: RetrievalMetrics
    #: 逐查询样本：[(query_id, 查询文本, [(doc_id, 排名, 是否相关), ...]), ...]
    sample: list[tuple[str, str, list[tuple[str, int, bool]]]] = field(default_factory=list)
    #: 索引特有的诊断信息（ivf 的分簇大小 / hnsw 的度数分布 …）
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def spec_id(self) -> str:
        return self.spec.spec_id

    @property
    def embedding(self) -> str:
        return self.spec.embedding

    @property
    def index(self) -> str:
        return self.spec.index

    def axis(self, name: str) -> str:
        """取某个设计轴上的取值（消融与归因用）。"""
        return self.spec.axis(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "spec_id": self.spec_id,
            "metrics": self.metrics.to_dict(),
            "meta": dict(sorted(self.meta.items())),
        }


# ----------------------------------------------------------------------
# 版本
# ----------------------------------------------------------------------
@dataclass
class Version:
    """内容寻址版本：digest 由**入选的检索配置**决定。

    口径与 TokenizerFactory 一致：这里「内容」是最终选中的那个检索组件
    （embedding + chunking + index + distance + search），而不是整条流水线的
    全部产物。只有入选配置变了版本才该变。
    """

    version_id: str
    created_at: str
    digest: str
    tag: str
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "digest": self.digest,
            "tag": self.tag,
            "counts": dict(sorted(self.counts.items())),
        }


# ----------------------------------------------------------------------
# 流水线产物
# ----------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """整条流水线的产物。"""

    corpus: CorpusStats
    runs: list[RetrievalRun] = field(default_factory=list)
    #: 消融矩阵的行（每个 spec 一行），由 compare 模块生成
    matrix: list[dict[str, Any]] = field(default_factory=list)
    #: Pareto 前沿上的 spec_id（按目标对分组的字典）
    frontiers: dict[str, list[str]] = field(default_factory=dict)
    #: 单变量消融表 + 空间归因 + 达额成本，由 compare 模块生成
    ablations: dict[str, Any] = field(default_factory=dict)
    #: 最终入选的 spec_id（Select 阶段的产物）
    selected: str = ""
    dag: Any = None
    version: Version | None = None
    counters: dict[str, Any] = field(default_factory=dict)
    cfg: dict[str, Any] = field(default_factory=dict)

    # -- 便捷查询 ----------------------------------------------------
    def find(self, spec_id: str) -> RetrievalRun | None:
        for r in self.runs:
            if r.spec_id == spec_id:
                return r
        return None

    def axis_values(self, axis: str) -> list[str]:
        """某个设计轴上的全部取值（按 spec_id 排序去重）。"""
        return sorted({r.spec.axis(axis) for r in self.runs})

    def axes(self) -> list[str]:
        return [a for a in DESIGN_AXES if self.axis_values(a)]

    def frontier_of(self, key: str) -> list[RetrievalRun]:
        return [
            r for sid in self.frontiers.get(key, []) if (r := self.find(sid)) is not None
        ]

    def known_spec_ids(self) -> list[str]:
        return sorted(r.spec_id for r in self.runs)

    def best_by(self, metric: str, direction: str = "min") -> RetrievalRun | None:
        if not self.runs:
            return None
        key = lambda r: r.metrics.get(metric)  # noqa: E731
        return (min if direction == "min" else max)(self.runs, key=key)

    def reference_run(self) -> RetrievalRun | None:
        """参考配置对应的 run（消融表的基准行）。"""
        ref = (self.cfg or {}).get("ablation", {}).get("reference")
        if not ref:
            return None
        return self.find(str(ref))
