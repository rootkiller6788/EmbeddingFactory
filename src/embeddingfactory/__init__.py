"""EmbeddingFactory —— 把「表示空间」与「检索空间」解耦的检索工厂。

    Documents → Chunk(×策略) → Embed(×模型) → Index(×类型×距离×搜索参数)
        → Query → Retrieve → Evaluate → Compare → Select → Export

产物：一份**消融矩阵** + 一条 **Latency–Recall Pareto 前沿** + 一个可导出的检索组件。

核心命题（README 第五节）：**性能 = f(Embedding, Chunking, Index, Distance, Search)**，
五个因素构成一个可搜索空间，任意一环的参数变化都应能被独立消融并归因到指标变化上。

本包刻意把这两组因素做成两个**空间**：

- **表示空间**（``embedding`` × ``chunking``）决定**召回率的上限**；
- **检索空间**（``index`` × ``distance`` × ``search``）决定**达到该上限要多少成本**。

于是「换 embedding 没用但换索引有用」和「换 embedding 有用」是两种可区分的结论 ——
这是本项目与「把整条 pipeline 当黑盒调参」最本质的区别。
"""

from __future__ import annotations

from .compare import (
    assemble,
    hypervolume_of,
    non_dominated,
    reference_point,
    select,
    space_attribution,
)
from .config import DEFAULTS, expand_specs, load_config, validate
from .corpus import CHUNKERS, load_corpus, tokenize
from .embed import EMBEDDINGS, EmbeddingModel, fit_embedding
from .index import DISTANCES, INDEXES, BaseIndex, SearchResult, build_index
from .lineage import LineageDAG, LineageNode
from .metrics import METRIC_DIRECTIONS, TIMING_METRICS, point_of
from .pipeline import run_pipeline, write_outputs
from .retrieve import evaluate, merge_to_docs, ndcg_at_k, recall_at_k
from .schema import (
    CODE_VERSION,
    CorpusStats,
    DocChunk,
    Document,
    Query,
    RetrievalMetrics,
    RetrievalResult,
    RetrievalRun,
    RetrievalSpec,
    Version,
)

__version__ = "0.1.0"

__all__ = [
    "CODE_VERSION",
    "CHUNKERS",
    "CorpusStats",
    "DEFAULTS",
    "DISTANCES",
    "DocChunk",
    "Document",
    "EMBEDDINGS",
    "EmbeddingModel",
    "INDEXES",
    "LineageDAG",
    "LineageNode",
    "METRIC_DIRECTIONS",
    "Query",
    "RetrievalMetrics",
    "RetrievalResult",
    "RetrievalRun",
    "RetrievalSpec",
    "SearchResult",
    "TIMING_METRICS",
    "Version",
    "BaseIndex",
    "assemble",
    "build_index",
    "evaluate",
    "expand_specs",
    "fit_embedding",
    "hypervolume_of",
    "load_config",
    "load_corpus",
    "merge_to_docs",
    "ndcg_at_k",
    "non_dominated",
    "point_of",
    "recall_at_k",
    "reference_point",
    "run_pipeline",
    "select",
    "space_attribution",
    "tokenize",
    "validate",
    "write_outputs",
]
