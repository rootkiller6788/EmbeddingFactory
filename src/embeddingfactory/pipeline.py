"""编排：把设计空间跑成一份结果。

    Documents → Chunk(×策略) → Embed(×模型) → Index(×类型×距离×搜索参数)
        → Query → Retrieve → Evaluate → Compare → Select → Export

三条实现纪律（都是本工作区反复踩出来的，见 skill 的「扇出-收拢型」一节）：

1. **中间结果必须缓存并共享给所有兄弟分支**。语料只加载一次、每个分块策略
   只切一次、每个 ``(embedding, chunking)`` 只拟合一次 —— 所有索引共用同一份向量。
   不缓存的话，「所有配置看的是同一份输入」这个前提会在实现层悄悄失效
   （每个 spec 各自重算一遍，结果可能因浮点顺序差异而不同）。测试用
   **对象同一性**（``is``）断言这一点。
2. **只有真正产出东西的阶段才建 DAG 节点**。``corpus`` / ``chunk`` / ``embed`` /
   ``idx`` / ``bench`` 各建一个；纯汇总的 ``compare`` 不建 ——
   它不产出新产物，建了会让图里出现一条没有对应产物的边。
3. **收拢点的输入恰好是前沿候选**。这是本形状的存在理由：看一张图就知道
   谁进了前沿。多一个（把落选者接进来）或少一个（前沿没接住）都算 bug。
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import compare, metrics as metrics_mod, report as report_mod  # noqa: F401
from .compare import assemble, frontier_key, frontier_pairs
from .corpus import CorpusBundle, chunk_profile, load_corpus
from .embed import EmbeddingModel, fit_embedding
from .index import BaseIndex, build_index
from .lineage import (
    CORPUS_NODE_ID,
    LineageDAG,
    LineageNode,
    artifact_node_id,
    bench_node_id,
    chunk_node_id,
    embed_node_id,
    index_node_id,
    select_node_id,
)
from .metrics import point_of
from .retrieve import RetrievalOutcome, evaluate, sample_of
from .schema import (
    CODE_VERSION,
    CorpusStats,
    RetrievalMetrics,
    RetrievalResult,
    RetrievalRun,
    RetrievalSpec,
    Version,
    sha256_text,
    utcnow,
)


@dataclass
class Bundle:
    """跑完之后**留在内存里、不落盘**的东西。

    刻意与 ``RetrievalResult`` 分开：``result`` 是可序列化的产物（进 JSON），
    ``bundle`` 是「活着」的对象（分词器 / 索引 / 向量）—— 它们要么太大、
    要么无法序列化。``write_outputs`` 需要两者，因为导出 artifact 时
    必须把入选配置的向量写出去。
    """

    corpus: CorpusBundle
    models: dict[str, EmbeddingModel] = field(default_factory=dict)
    vectors: dict[str, list[list[float]]] = field(default_factory=dict)
    indexes: dict[str, BaseIndex] = field(default_factory=dict)
    outcomes: dict[str, RetrievalOutcome] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def model_key(self, embedding: str, chunking: str) -> str:
        return f"{embedding}+{chunking}"


def _chunk_doc_ids(corpus: CorpusBundle, chunking: str) -> list[str]:
    return [c.doc_id for c in corpus.chunks[chunking]]


def _build_embed_node(
    dag: LineageDAG,
    embedding: str,
    chunking: str,
    model: EmbeddingModel,
) -> str:
    node_id = embed_node_id(embedding, chunking)
    dag.add_node(
        LineageNode(
            node_id=node_id,
            kind="embed",
            label=f"{embedding} × {chunking}",
            inputs=[chunk_node_id(chunking)],
            params={"embedding": embedding, "chunking": chunking, "dim": model.dim},
            stats={
                "vocab": len(model.vocab),
                **{k: v for k, v in model.meta.items()},
            },
        )
    )
    return node_id


def run_pipeline(
    cfg: dict[str, Any], *, verbose: bool = False
) -> tuple[RetrievalResult, Bundle]:
    """跑完整条流水线。返回 ``(result, bundle)``。"""
    started = time.perf_counter()
    seed = int(cfg["seed"])

    # ---- 1. 语料（只加载一次，所有分支共用） -------------------------
    corpus = load_corpus(cfg)
    bundle = Bundle(corpus=corpus)

    dag = LineageDAG()
    dag.add_node(
        LineageNode(
            node_id=CORPUS_NODE_ID,
            kind="corpus",
            label=corpus.name,
            params={
                "documents": cfg["corpus"]["documents"],
                "queries": cfg["corpus"]["queries"],
                "k": int(cfg["retrieval"]["k"]),
            },
            stats={
                "docs": len(corpus.documents),
                "queries": len(corpus.queries),
            },
        )
    )
    for strategy in [str(c) for c in cfg["representation"]["chunkings"]]:
        profile = chunk_profile(corpus.chunks[strategy])
        dag.add_node(
            LineageNode(
                node_id=chunk_node_id(strategy),
                kind="chunk",
                label=f"chunk × {strategy}",
                inputs=[CORPUS_NODE_ID],
                params={
                    "chunking": strategy,
                    **dict(cfg["representation"]["chunking_params"].get(strategy, {})),
                },
                stats=profile,
            )
        )

    # ---- 2. 表示空间：拟合（缓存 + 共享） ---------------------------
    for embedding in [str(e) for e in cfg["representation"]["embeddings"]]:
        for chunking in [str(c) for c in cfg["representation"]["chunkings"]]:
            key = bundle.model_key(embedding, chunking)
            model = fit_embedding(
                embedding,
                corpus.chunk_texts(chunking),
                cfg["representation"]["embedding_params"].get(embedding, {}),
                seed,
            )
            bundle.models[key] = model
            # 语料向量只算一次，所有索引共用同一份（对象同一性有测试守着）
            bundle.vectors[key] = model.matrix(corpus.chunk_texts(chunking))
            _build_embed_node(dag, embedding, chunking, model)

    # ---- 3. 检索空间：每个 spec 建索引 + 跑查询 ----------------------
    k = int(cfg["retrieval"]["k"])
    fetch_k = k * int(cfg["retrieval"]["fetch_factor"])
    latency_repeats = int(cfg["metrics"]["latency_repeats"])
    specs = _expand(cfg)

    runs: list[RetrievalRun] = []
    for i, spec in enumerate(specs, start=1):
        key = bundle.model_key(spec.embedding, spec.chunking)
        model = bundle.models[key]
        vectors = bundle.vectors[key]
        chunk_doc_ids = _chunk_doc_ids(corpus, spec.chunking)

        index_params = dict(cfg["retrieval"]["index_params"].get(spec.index, {}))
        index_params.update(spec.search)

        build_start = time.perf_counter()
        index = build_index(
            spec.index, vectors, spec.distance, index_params, seed
        )
        build_seconds = time.perf_counter() - build_start

        outcome = evaluate(
            index,
            model,
            corpus.queries,
            chunk_doc_ids=chunk_doc_ids,
            k=k,
            fetch_k=fetch_k,
            latency_repeats=latency_repeats,
        )
        bundle.indexes[spec.spec_id] = index
        bundle.outcomes[spec.spec_id] = outcome
        bundle.timings[spec.spec_id] = build_seconds

        m = RetrievalMetrics(
            recall_at_k=outcome.mean_recall,
            ndcg_at_k=outcome.mean_ndcg,
            mrr=outcome.mrr,
            n_distance_computations=outcome.mean_distance_computations,
            query_latency_ms=outcome.mean_latency_ms,
            build_seconds=build_seconds,
            index_bytes=index.size_bytes(),
            n_queries=len(outcome.outcomes),
            n_chunks=len(vectors),
            k=k,
            n_queries_zero_recall=outcome.n_zero_recall,
            mean_first_relevant_rank=outcome.mean_first_relevant_rank,
        )
        run = RetrievalRun(
            spec=spec,
            metrics=m,
            sample=sample_of(outcome, int(cfg["retrieval"]["sample_queries"])),
            meta=index.meta(),
        )
        runs.append(run)

        # DAG：idx ← embed，bench ← idx
        dag.add_node(
            LineageNode(
                node_id=index_node_id(spec.spec_id),
                kind="index",
                label=f"{spec.index} × {spec.distance} × {spec.search_tag}",
                inputs=[embed_node_id(spec.embedding, spec.chunking)],
                params={
                    "spec_id": spec.spec_id,
                    "index": spec.index,
                    "distance": spec.distance,
                    "search": dict(spec.search),
                },
                stats={
                    "vectors": len(vectors),
                    "index_bytes": index.size_bytes(),
                },
            )
        )
        dag.add_node(
            LineageNode(
                node_id=bench_node_id(spec.spec_id),
                kind="bench",
                label=f"bench {spec.spec_id}",
                inputs=[index_node_id(spec.spec_id)],
                stats={
                    f"recall@{k}": round(m.recall_at_k, 4),
                    "ndcg": round(m.ndcg_at_k, 4),
                    "dc": round(m.n_distance_computations, 1),
                },
            )
        )
        if verbose:
            print(f"[{i:>3}/{len(specs)}] {metrics_mod.metric_line(run)}")

    # ---- 4. 汇总（矩阵 / 前沿 / 消融 / 归因 / 选型） -----------------
    corpus_stats = CorpusStats(
        name=corpus.name,
        n_docs=len(corpus.documents),
        n_chunks=len(corpus.chunks[str(cfg["representation"]["chunkings"][0])]),
        n_queries=len(corpus.queries),
        n_topics=len({d.topic for d in corpus.documents if d.topic}),
        mean_chunk_chars=_mean(
            c.n_chars for c in corpus.chunks[str(cfg["representation"]["chunkings"][0])]
        ),
        mean_docs_per_query=_mean(len(q.relevant) for q in corpus.queries),
        topic_sizes=_topic_sizes(corpus),
        chunking={
            s: chunk_profile(corpus.chunks[s])
            for s in [str(c) for c in cfg["representation"]["chunkings"]]
        },
    )
    result = RetrievalResult(corpus=corpus_stats, runs=runs, cfg=cfg, dag=dag)
    assemble(result, cfg)

    # ---- 5. 收拢点：输入恰好是主前沿候选 ---------------------------
    main_key = frontier_key(cfg["pareto"]["objectives"])
    frontier_ids = sorted(result.frontiers.get(main_key, []))
    sel_id = select_node_id("main")
    dag.add_node(
        LineageNode(
            node_id=sel_id,
            kind="select",
            label=f"select {main_key}",
            inputs=[bench_node_id(sid) for sid in frontier_ids],
            params={
                "rule": cfg["select"]["rule"],
                "objectives": list(cfg["pareto"]["objectives"]),
            },
            stats={
                "candidates": len(frontier_ids),
                "selected": result.selected,
            },
        )
    )
    dag.add_node(
        LineageNode(
            node_id=artifact_node_id("main"),
            kind="artifact",
            label=f"artifact {result.selected}",
            inputs=[sel_id],
            params={"spec_id": result.selected},
            stats={
                "index_bytes": (
                    result.find(result.selected).metrics.index_bytes
                    if result.find(result.selected)
                    else 0
                )
            },
        )
    )

    # ---- 6. 版本（内容寻址：由入选配置决定） ------------------------
    selected_run = result.find(result.selected)
    digest_source = (
        json.dumps(
            {
                "spec": selected_run.spec.to_dict(),
                "recall": round(selected_run.metrics.recall_at_k, 6),
                "ndcg": round(selected_run.metrics.ndcg_at_k, 6),
                "n_chunks": selected_run.metrics.n_chunks,
            },
            sort_keys=True,
        )
        if selected_run
        else "empty"
    )
    digest = sha256_text(digest_source)
    result.version = Version(
        version_id=f"{cfg['version_tag']}-{digest[:8]}",
        created_at=utcnow(),
        digest=digest,
        tag=str(cfg["version_tag"]),
        counts={
            "specs": len(result.runs),
            "docs": corpus_stats.n_docs,
            "chunks": corpus_stats.n_chunks,
            "queries": corpus_stats.n_queries,
        },
    )
    result.counters["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    result.counters["code_version"] = CODE_VERSION
    return result, bundle


def _expand(cfg: dict[str, Any]) -> list[RetrievalSpec]:
    from .config import expand_specs

    return expand_specs(cfg)


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _topic_sizes(corpus: CorpusBundle) -> dict[str, int]:
    out: dict[str, int] = {}
    for doc in corpus.documents:
        out[doc.topic] = out.get(doc.topic, 0) + 1
    return out


# ----------------------------------------------------------------------
# 落盘
# ----------------------------------------------------------------------
MATRIX_HEADER: tuple[str, ...] = (
    "spec_id",
    "embedding",
    "chunking",
    "index",
    "distance",
    "search",
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "n_distance_computations",
    "query_latency_ms",
    "build_seconds",
    "index_bytes",
    "n_queries_zero_recall",
    "frontier",
)


def write_outputs(
    result: RetrievalResult, bundle: Bundle, out_dir: str | Path
) -> dict[str, Path]:
    """落盘全部产物。返回 ``{名字: 路径}``。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _w(name: str, text: str) -> Path:
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        written[name] = p
        return p

    # 1) 消融矩阵
    _w("matrix.csv", _csv(MATRIX_HEADER, [_matrix_csv_row(r) for r in result.matrix]))

    # 2) 前沿（每个目标对一张表，合并成一个 CSV，用 objective_pair 列区分）
    rows: list[list[str]] = []
    for pair in frontier_pairs(result.cfg):
        key = frontier_key(pair)
        for row in compare.frontier_rows(result, key):
            rows.append(
                [
                    key,
                    str(row["rank"]),
                    str(row["spec_id"]),
                    str(row["embedding"]),
                    str(row["chunking"]),
                    str(row["index"]),
                    str(row["distance"]),
                    str(row["search"]),
                    f"{row['recall_at_k']:.6f}",
                    f"{row['ndcg_at_k']:.6f}",
                    f"{row['mrr']:.6f}",
                    f"{row['n_distance_computations']:.3f}",
                    f"{row['query_latency_ms']:.4f}",
                    str(row["index_bytes"]),
                    str(row["dominated"]),
                    f"{row['hv_contribution']:.8f}",
                    "1" if row["selected"] else "0",
                ]
            )
    _w(
        "frontier.csv",
        _csv(
            (
                "objective_pair",
                "rank",
                "spec_id",
                "embedding",
                "chunking",
                "index",
                "distance",
                "search",
                "recall_at_k",
                "ndcg_at_k",
                "mrr",
                "n_distance_computations",
                "query_latency_ms",
                "index_bytes",
                "dominated",
                "hv_contribution",
                "selected",
            ),
            rows,
        ),
    )

    # 3) 每个 spec 一行（含样本切片）
    runs_lines = [
        json.dumps(
            {**run.to_dict(), "sample": _sample_payload(run)},
            ensure_ascii=False,
        )
        for run in result.runs
    ]
    _w("runs.jsonl", "\n".join(runs_lines) + ("\n" if runs_lines else ""))

    # 4) result.json —— **不含样本**（样本在 runs.jsonl 里）
    payload = {
        "code_version": CODE_VERSION,
        "corpus": result.corpus.to_dict(),
        "selected": result.selected,
        "matrix": result.matrix,
        "frontiers": result.frontiers,
        "ablations": result.ablations,
        "counters": result.counters,
        "version": result.version.to_dict() if result.version else None,
        "config": _config_payload(result.cfg),
    }
    _w("result.json", json.dumps(payload, ensure_ascii=False, indent=2))

    # 5) 消融 / 归因（单独一份，便于直接看）
    _w(
        "ablations.json",
        json.dumps(result.ablations, ensure_ascii=False, indent=2),
    )

    # 6) artifact：入选配置的**可载入**检索组件
    selected_run = result.find(result.selected)
    if selected_run is not None:
        artifact = _artifact_payload(selected_run, bundle, result.cfg)
        _w(
            f"artifacts/{selected_run.spec_id}.json",
            json.dumps(artifact, ensure_ascii=False),
        )

    # 7) 血缘
    if result.dag is not None:
        result.dag.save(out / "lineage.json")
        written["lineage.json"] = out / "lineage.json"
        _w("lineage.dot", result.dag.to_dot())
        _w("lineage_selected.dot", result.dag.to_dot(only_selected_ancestry=True))

    # 8) 版本
    if result.version is not None:
        _w("version.json", json.dumps(result.version.to_dict(), ensure_ascii=False, indent=2))

    return written


def _matrix_csv_row(row: dict[str, Any]) -> list[str]:
    frontier = row.get("frontier") or []
    return [
        str(row.get("spec_id", "")),
        str(row.get("embedding", "")),
        str(row.get("chunking", "")),
        str(row.get("index", "")),
        str(row.get("distance", "")),
        str(row.get("search", "")),
        f"{float(row.get('recall_at_k', 0.0)):.6f}",
        f"{float(row.get('ndcg_at_k', 0.0)):.6f}",
        f"{float(row.get('mrr', 0.0)):.6f}",
        f"{float(row.get('n_distance_computations', 0.0)):.3f}",
        f"{float(row.get('query_latency_ms', 0.0)):.4f}",
        f"{float(row.get('build_seconds', 0.0)):.5f}",
        str(int(row.get("index_bytes", 0))),
        str(int(row.get("n_queries_zero_recall", 0))),
        "|".join(frontier),
    ]


def _sample_payload(run: RetrievalRun) -> list[dict[str, Any]]:
    return [
        {
            "query_id": qid,
            "text": text,
            "top": [
                {"doc_id": doc, "rank": rank, "relevant": rel}
                for doc, rank, rel in top
            ],
        }
        for qid, text, top in run.sample
    ]


def _config_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    """只留可序列化的部分（去掉 ``_config_path`` / ``_root``）。"""
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def _artifact_payload(
    run: RetrievalRun, bundle: Bundle, cfg: dict[str, Any]
) -> dict[str, Any]:
    """导出检索组件。

    它必须**真的能用**：``cli.query`` 会载入它、嵌入一段新文本、在语料上搜索。
    所以里面既有嵌入模型的拟合参数（词表 / idf / ngram），
    也有索引需要的全部向量与 chunk→doc 映射。有测试守着
    「载入后重跑全部查询，召回率与 run 一致」。

    **``seed`` 必须一起落盘。** ivf 的 k-means++ 初始化由它派生：
    不存的话，载入方只能自己猜一个（``cli.query`` 原来传的是 0），
    于是重建出来的分簇与基准里的不同 —— 同一个 spec 的召回率会变
    （实测 ``tfidf-sentence-ivf-cos-np1``：种子 20260930 下 0.6806，
    种子 0 下 0.7917）。这种「产物载入后行为变了」是最难查的一类 bug：
    它不报错，只是数字对不上。
    """
    key = bundle.model_key(run.spec.embedding, run.spec.chunking)
    model = bundle.models[key]
    vectors = bundle.vectors[key]
    chunk_doc_ids = _chunk_doc_ids(bundle.corpus, run.spec.chunking)
    chunk_ids = [c.chunk_id for c in bundle.corpus.chunks[run.spec.chunking]]
    index_params = dict(cfg["retrieval"]["index_params"].get(run.spec.index, {}))
    index_params.update(run.spec.search)
    return {
        "code_version": CODE_VERSION,
        "spec": run.spec.to_dict(),
        # 重建索引要用的种子（ivf 的分簇、特征哈希的符号位都依赖它）
        "seed": int(cfg["seed"]),
        "k": run.metrics.k,
        "fetch_k": run.metrics.k * int(cfg["retrieval"]["fetch_factor"]),
        "embedding": {
            "name": model.name,
            "dim": model.dim,
            "vocab": list(model.vocab),
            # idf / avg_len / bucket_idf 一律**不截断** —— 它们进查询向量，
            # 末位差一点就可能翻转一个并列名次。
            "idf": {t: model.idf[t] for t in sorted(model.idf)},
            "bucket_idf": list(model.bucket_idf),
            "avg_len": model.avg_len,
            "params": {k2: model.params[k2] for k2 in sorted(model.params)},
        },
        "index": {
            "name": run.spec.index,
            "distance": run.spec.distance,
            "params": {k2: index_params[k2] for k2 in sorted(index_params)},
        },
        "corpus": {
            "chunk_ids": chunk_ids,
            "doc_ids": chunk_doc_ids,
            # 向量按**全精度**落盘，不做任何截断。
            #
            # 这里曾经写成 ``round(v, 6)``，理由是「再高的精度对结论没有影响，
            # 但会把文件撑大好几倍」—— 后半句是错的：实测全精度只比 6 位小数
            # 大 4%（0.0 占绝大多数，Python 的 repr 本来就短）。
            # 而 6 位小数会让 artifact **载入后行为改变**：ivf 的 k-means++
            # 按 D² 采样选簇心，向量末位差一点就换一组簇心，实测
            # ``tfidf-sentence-ivf-cos-np1`` 的召回率从 0.6806 变成 0.7708。
            # 导出物必须与基准逐位一致，所以这里不截断。
            "vectors": [list(vec) for vec in vectors],
        },
        "metrics": run.metrics.to_dict(),
    }


def _csv(header: tuple[str, ...] | list[str], rows: list[list[str]]) -> str:
    import io

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(list(header))
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()
