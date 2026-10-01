"""度量定义 —— README 第六节「评测维度」的可执行版本。

| 维度 | 字段 | 方向 | 为什么单独量它 |
| --- | --- | --- | --- |
| 质量 | ``recall_at_k`` | 越大越好 | 检索的任务指标（README 的 Recall@k） |
| 排序 | ``ndcg_at_k`` | 越大越好 | 相关文档排在前面才算好（README 的 nDCG） |
| 排序 | ``mrr`` | 越大越好 | 第一条相关结果排多前（README 的 MRR） |
| 成本 | ``n_distance_computations`` | 越小越好 | **确定性成本模型** |
| 成本 | ``query_latency_ms`` | 越小越好 | 实测延迟（README 的 Latency） |
| 成本 | ``index_bytes`` | 越小越好 | 索引内存（README 的 RAM） |
| 成本 | ``build_seconds`` | 越小越好 | 建索引耗时（离线成本） |
| 诊断 | ``n_queries_zero_recall`` | 越小越好 | 一条都没召回的查询数 |
| 诊断 | ``mean_first_relevant_rank`` | 越小越好 | 第一条相关结果平均排多前 |

本项目最重要的一条口径：**成本轴用「距离计算次数」而不是实测延迟**
------------------------------------------------------------------------

TokenizerFactory 项目实测过：以吞吐为轴的 Pareto 前沿在两次运行之间
会换成员（``wordpiece-v128`` → ``wordpiece-v128+whitespace``），
因为吞吐是实测值，中位数只压小噪声、消不掉它。

本项目的验收标准**必须**给出 Latency–Recall 前沿。于是做法是：

1. **主前沿用 ``n_distance_computations``**：它完全由索引结构与搜索参数
   决定（扫了多少个向量），**逐位可复现**。它是一条真实的前沿，
   而且每个点都能在图上解释成「扫了这么多向量」。
2. **另算一条 ``query_latency_ms`` 的副前沿**，用来验证
   「成本模型确实与真实延迟同向」—— 这是一个**能被证伪的断言**：
   如果某个配置扫的向量少却更慢，说明成本模型选错了，报告要如实写出来。
3. 含实测轴的副前沿在报告里标注「含实测轴，不可逐位复现」，
   ``TIMING_METRICS`` 是这个约定的可执行形式。

这是本项目对「可复现的量化结果」这条验收标准的正面回应：
不是回避延迟，而是**用一个可复现的量去代理它，再用实测去验证代理成立**。
"""

from __future__ import annotations

from typing import Any, Sequence

from .schema import RetrievalMetrics, RetrievalRun

#: 指标方向。Pareto 与排序全靠它，**不要在别处再写一遍方向**
METRIC_DIRECTIONS: dict[str, str] = {
    "recall_at_k": "max",
    "ndcg_at_k": "max",
    "mrr": "max",
    "n_distance_computations": "min",
    "query_latency_ms": "min",
    "build_seconds": "min",
    "index_bytes": "min",
    "n_chunks": "min",
    "n_queries_zero_recall": "min",
    "mean_first_relevant_rank": "min",
}

#: 实测类指标（由计时得到）。它们**不是逐位可复现的** —— 见模块文档。
TIMING_METRICS: frozenset[str] = frozenset({"query_latency_ms", "build_seconds"})

#: 中文标签（报告表头与 CLI 共用，避免两处各写一遍）
METRIC_LABELS: dict[str, str] = {
    "recall_at_k": "Recall@k",
    "ndcg_at_k": "nDCG@k",
    "mrr": "MRR",
    "n_distance_computations": "距离计算次数",
    "query_latency_ms": "延迟(ms)",
    "build_seconds": "建索引(s)",
    "index_bytes": "索引字节",
    "n_chunks": "块数",
    "n_queries_zero_recall": "零召回查询数",
    "mean_first_relevant_rank": "首个相关名次",
}

#: 成本类指标（越小越好，且不是质量指标）—— 报告里用来分组
COST_METRICS: frozenset[str] = frozenset(
    {"n_distance_computations", "query_latency_ms", "build_seconds", "index_bytes"}
)

#: 质量类指标
QUALITY_METRICS: frozenset[str] = frozenset({"recall_at_k", "ndcg_at_k", "mrr"})


def has_timing_axis(objectives: Sequence[str]) -> bool:
    """这组目标轴里是否含实测指标。含的话，它的前沿不可逐位复现。"""
    return any(o in TIMING_METRICS for o in objectives)


def is_deterministic_axis(objectives: Sequence[str]) -> bool:
    """这组目标轴是否**完全可复现**（不含实测指标）。"""
    return not has_timing_axis(objectives)


# ----------------------------------------------------------------------
# 摊平
# ----------------------------------------------------------------------
def point_of(run: RetrievalRun) -> dict[str, float]:
    """把一个 run 摊平成「指标名 → 数值」，供 Pareto / 排序使用。

    ``n_chunks`` 与 ``build_seconds`` 也在这里补齐 —— 它们一部分来自
    metrics 对象、一部分来自 spec，摊平后调用方不必再分支。
    """
    m = run.metrics
    return {
        "recall_at_k": float(m.recall_at_k),
        "ndcg_at_k": float(m.ndcg_at_k),
        "mrr": float(m.mrr),
        "n_distance_computations": float(m.n_distance_computations),
        "query_latency_ms": float(m.query_latency_ms),
        "build_seconds": float(m.build_seconds),
        "index_bytes": float(m.index_bytes),
        "n_chunks": float(m.n_chunks),
        "n_queries_zero_recall": float(m.n_queries_zero_recall),
        "mean_first_relevant_rank": float(m.mean_first_relevant_rank),
    }


def metric_line(run: RetrievalRun) -> str:
    """一行摘要，用于日志与 CLI 的紧凑输出。"""
    m = run.metrics
    return (
        f"{run.spec_id:<32} "
        f"recall@{m.k}={m.recall_at_k:.4f} ndcg={m.ndcg_at_k:.4f} mrr={m.mrr:.4f} "
        f"dc={m.n_distance_computations:>7.1f} "
        f"lat={m.query_latency_ms:>7.3f}ms "
        f"idx={m.index_bytes / 1024.0:>7.1f}KB build={m.build_seconds:.4f}s"
    )


def summarize(run: RetrievalRun) -> dict[str, Any]:
    """给报告用的完整摘要（指标 + spec 的两个空间归属）。"""
    return {
        "spec_id": run.spec_id,
        "representation": run.spec.to_dict()["representation"],
        "retrieval": run.spec.to_dict()["retrieval"],
        **run.metrics.to_dict(),
        "meta": dict(sorted(run.meta.items())),
    }


def empty_metrics(k: int) -> RetrievalMetrics:
    """空结果（查询集为空时用），避免各处写一长串 0.0。"""
    return RetrievalMetrics(
        recall_at_k=0.0,
        ndcg_at_k=0.0,
        mrr=0.0,
        n_distance_computations=0.0,
        query_latency_ms=0.0,
        build_seconds=0.0,
        index_bytes=0,
        n_queries=0,
        n_chunks=0,
        k=k,
    )
