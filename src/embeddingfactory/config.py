"""配置加载与默认值合并。

配置为纯 JSON（避免引入 PyYAML），约定放在 ``<root>/configs/*.json``，
因此 ``_root`` 取配置文件的上两级目录，配置里的相对路径都相对 ``_root`` 解析。

配置分成四块，刻意的 —— **四块的切分直接对应 README 的「把两个变量分开」**：

- ``corpus``（指向 ``data/corpus/*.jsonl``）：**问题是什么** ——
  文档在哪、查询与 ground truth 在哪。换语料只改这一块，不动任何代码。
- ``representation``：**表示空间是什么** —— 分块策略、嵌入模型及其维度。
  这一块决定**召回率的上限**。
- ``retrieval``：**检索空间是什么** —— 索引类型、距离、搜索参数、k。
  这一块决定**达到该上限要花多少成本**。
- ``ablation`` / ``pareto`` / ``select``：**怎么比、怎么归因、怎么选** ——
  消融的参考配置、Pareto 用哪两个目标、最终按什么规则选出 Artifact。

把四块分开的好处很实际：回答「换个语料，结论还成立吗」时只需要替换 ``corpus``；
回答「换更好的 embedding 能不能救回召回率」时只需要改 ``representation``；
回答「同样的召回率能不能更快」时只需要改 ``retrieval``。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "pipeline_id": "embeddingfactory",
    "version_tag": "v1",
    "seed": 20260930,
    # 语料 —— 「问题本身」
    "corpus": {
        "documents": "data/corpus/documents.jsonl",
        "queries": "data/corpus/queries.jsonl",
        "name": "embeddingfactory-demo",
    },
    # 表示空间 —— 决定召回率上限
    "representation": {
        "chunkings": ["sentence", "fixed"],
        "chunking_params": {
            # sentence：按句号/换行切，超长句再按标点回退切
            "sentence": {"max_chars": 50},
            # fixed：定长窗口，overlap=0 时块之间不重叠
            "fixed": {"size": 40, "overlap": 0},
        },
        "embeddings": ["tfidf", "bm25", "hashing"],
        "embedding_params": {
            # dim 是「向量维度」，三个模型共用同一个量级才可比
            "tfidf": {"dim": 512, "min_df": 1},
            "bm25": {"dim": 512, "k1": 1.2, "b": 0.75},
            # hashing：字符 n-gram 特征哈希（对改写/错字更鲁棒，但没有词序）
            "hashing": {"dim": 512, "ngram": 3},
        },
    },
    # 检索空间 —— 决定达到上限的成本
    "retrieval": {
        "indexes": ["flat", "ivf", "hnsw"],
        "distances": ["cos", "l2"],
        # 每个索引的搜索参数档位。**每一档都会展开成一个 spec** ——
        # 「搜索参数」是 README 列的四个轴之一，不能只留一个默认值。
        # flat 没有参数（精确检索），留一个空档占位，这样展开逻辑只有一条路径。
        "search_params": {
            "flat": [{}],
            "ivf": [{"nprobe": 1}, {"nprobe": 3}],
            "hnsw": [{"ef_search": 16}, {"ef_search": 64}],
        },
        "index_params": {
            "flat": {},
            # nlist 是分簇数。取 6 是因为本语料 chunk 数只有一百多 ——
            # nlist 再大就会「一簇一个向量」，nprobe=1 的召回率会掉到随机水平，
            # 那时「索引变快」只是因为「什么都没扫」，结论会失真。
            "ivf": {"nlist": 6, "kmeans_iters": 12},
            # M 是每个节点的邻居数，ef_construction 是建图时的搜索宽度
            "hnsw": {"M": 6, "ef_construction": 24},
        },
        # top-k（文档级）
        "k": 5,
        # 索引一次返回多少个 chunk（= k × fetch_factor），再归并成 top-k 文档。
        # 为什么不是「直接返回 k 个 chunk」：一篇文档有多个块，只取 k 个 chunk
        # 很可能只覆盖 2~3 篇文档，top-k 文档根本填不满 —— 那会把所有配置的
        # 召回率一起压低到无法区分。fetch_factor 是**检索空间的一个真实参数**
        # （工业界叫 rerank depth），所以它进配置、进报告。
        "fetch_factor": 2,
        # 每个 run 存多少条查询样本，供 explain / query 使用
        "sample_queries": 6,
    },
    # 消融 —— 一次只动一个变量
    "ablation": {
        # 参考配置：消融表的基准行。它必须真的在设计空间里（validate 会检查）。
        # 选 flat + cos 是有理由的：flat 是精确检索，cos 是归一化后的余弦，
        # 于是参考行的召回率是「该表示空间的上限」—— 消融表读起来就是
        # 「从上限往下掉多少」，而不是「从一个任意起点往下掉多少」。
        # 注意 chunking 的全名是 ``sentence``（不是 ``sent``）—— spec_id 用的是
        # ``RetrievalSpec`` 的原始字段拼接，写缩写会让 validate 直接报「不在设计空间里」。
        "reference": "tfidf-sentence-flat-cos",
        # 「达到最优召回率的多少比例」算达标 —— 达额成本表用这个阈值
        "reach_thresholds": [0.8, 0.9, 0.95, 1.0],
    },
    # Pareto
    "pareto": {
        # 主前沿：**成本轴用「距离计算次数」而不是实测延迟**。
        # 距离计算次数完全由索引结构与搜索参数决定，逐位可复现；
        # 实测延迟受 CPU 降频与调度影响，两次运行不一致。
        # 这是本项目对「可复现的量化结果」这条验收标准的正面回应。
        "objectives": ["n_distance_computations", "recall_at_k"],
        # 额外再算几条前沿，用于回答「换个成本轴结论会不会变」：
        # - query_latency_ms：实测延迟（含实测轴，标注为不可逐位复现），
        #   用来验证「距离计算次数确实与真实延迟同向」；
        # - index_bytes：内存占用（README 的 RAM）；
        # - build_seconds：建索引耗时（离线成本，与在线成本是两回事）。
        "extra_objectives": [
            ["query_latency_ms", "recall_at_k"],
            ["index_bytes", "recall_at_k"],
            ["build_seconds", "recall_at_k"],
        ],
    },
    "select": {
        # max_recall_within_budget | knee
        "rule": "max_recall_within_budget",
        # 预算作用在哪个成本指标上（默认「每个查询平均距离计算次数」）
        "budget_metric": "n_distance_computations",
        # 预算值：真实部署里「每次查询扫多少个向量」是有硬上限的
        # （内存带宽 / 延迟 SLA），所以「预算内召回最好的那个」是最常用的选法。
        # 注意预算本身会改变赢家，见报告里的预算敏感性表。
        "budget": 40.0,
    },
    "metrics": {
        # 延迟基准重复几轮（取中位数）。单轮会被 CPU 降频与调度噪声带偏。
        # 注意：中位数只压小噪声，不能消掉它 —— 所以主前沿不用延迟。
        "latency_repeats": 3,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """递归合并：override 覆盖 base，dict 逐层深入，其余整体替换。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(root: str | Path, rel: str | Path) -> Path:
    """把配置里的相对路径解析为绝对路径（相对 ``root``）。

    签名收的是 ``root`` 而不是整个 ``cfg`` —— 子配置里没有 ``_root``，
    调用方请显式传 ``cfg["_root"]``。
    """
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 JSON 配置并与 DEFAULTS 合并，附带 ``_config_path`` / ``_root``。"""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    cfg = deep_merge(DEFAULTS, raw)
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(p.parent.parent)
    validate(cfg)
    return cfg


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读 JSONL。空行与 ``#`` 开头的注释行都跳过。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p}")
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def resolve_corpus_path(cfg: dict[str, Any], key: str) -> Path:
    """解析 ``corpus`` 里某个键指向的数据文件路径。"""
    rel = cfg["corpus"][key]
    return resolve_path(cfg["_root"], rel)


# ----------------------------------------------------------------------
# 校验 —— 尽量挡在运行之前
# ----------------------------------------------------------------------
def _validate_corpus(cfg: dict[str, Any]) -> None:
    for key in ("documents", "queries"):
        if not cfg["corpus"].get(key):
            raise ValueError(f"corpus.{key} 不能为空")


def _validate_representation(cfg: dict[str, Any]) -> None:
    from .corpus import CHUNKERS
    from .embed import EMBEDDINGS

    r = cfg["representation"]

    chunkings = [str(c) for c in r["chunkings"]]
    if not chunkings:
        raise ValueError("representation.chunkings 不能为空")
    if len(set(chunkings)) != len(chunkings):
        raise ValueError(f"representation.chunkings 有重复项：{chunkings}")
    unknown = [c for c in chunkings if c not in CHUNKERS]
    if unknown:
        raise ValueError(
            f"未知分块策略 {unknown}；可用：{', '.join(sorted(CHUNKERS))}"
        )
    for name in chunkings:
        params = r["chunking_params"].get(name, {})
        CHUNKERS[name].validate(params)

    embeddings = [str(e) for e in r["embeddings"]]
    if not embeddings:
        raise ValueError("representation.embeddings 不能为空")
    if len(set(embeddings)) != len(embeddings):
        raise ValueError(f"representation.embeddings 有重复项：{embeddings}")
    unknown = [e for e in embeddings if e not in EMBEDDINGS]
    if unknown:
        raise ValueError(
            f"未知嵌入模型 {unknown}；可用：{', '.join(sorted(EMBEDDINGS))}"
        )
    for name in embeddings:
        params = r["embedding_params"].get(name, {})
        EMBEDDINGS[name].validate(params)

    # 三个嵌入模型的维度必须同量级 —— 否则「换 embedding」这个消融里
    # 混进了「换向量维度」，一次动了两个变量（这是本工作区反复踩的坑）。
    dims = {
        name: int(r["embedding_params"].get(name, {}).get("dim", 0))
        for name in embeddings
    }
    nonzero = {k: v for k, v in dims.items() if v}
    if len(set(nonzero.values())) > 1:
        raise ValueError(
            f"各嵌入模型的 dim 不一致：{nonzero}。维度必须相同，否则消融表里"
            "「换 embedding」这一行混进了「换向量维度」这个变量，结论无法归因。"
        )


def _validate_retrieval(cfg: dict[str, Any]) -> None:
    from .index import DISTANCES, INDEXES

    r = cfg["retrieval"]

    indexes = [str(i) for i in r["indexes"]]
    if not indexes:
        raise ValueError("retrieval.indexes 不能为空")
    if len(set(indexes)) != len(indexes):
        raise ValueError(f"retrieval.indexes 有重复项：{indexes}")
    unknown = [i for i in indexes if i not in INDEXES]
    if unknown:
        raise ValueError(f"未知索引类型 {unknown}；可用：{', '.join(sorted(INDEXES))}")

    distances = [str(d) for d in r["distances"]]
    if not distances:
        raise ValueError("retrieval.distances 不能为空")
    if len(set(distances)) != len(distances):
        raise ValueError(f"retrieval.distances 有重复项：{distances}")
    unknown = [d for d in distances if d not in DISTANCES]
    if unknown:
        raise ValueError(
            f"未知距离 {unknown}；可用：{', '.join(sorted(DISTANCES))}"
        )

    for name in indexes:
        levels = r["search_params"].get(name)
        if not levels:
            raise ValueError(
                f"retrieval.search_params.{name} 至少要有一档 —— "
                "「搜索参数」是设计轴之一，空着等于这个索引不进设计空间"
            )
        INDEXES[name].validate_search(levels)
        INDEXES[name].validate_params(r["index_params"].get(name, {}))

    if int(r["k"]) < 1:
        raise ValueError("retrieval.k 至少为 1")
    if int(r["fetch_factor"]) < 1:
        raise ValueError(
            "retrieval.fetch_factor 至少为 1 —— 它必须 ≥ 1，否则索引返回的候选"
            "比 top-k 还少，top-k 文档永远填不满"
        )
    if int(r["sample_queries"]) < 0:
        raise ValueError("retrieval.sample_queries 不能为负")


def _validate_ablation(cfg: dict[str, Any]) -> None:
    from .metrics import METRIC_DIRECTIONS

    ab = cfg["ablation"]
    if not ab.get("reference"):
        raise ValueError("ablation.reference 不能为空（消融表需要一个基准行）")
    thresholds = [float(t) for t in ab["reach_thresholds"]]
    if not thresholds:
        raise ValueError("ablation.reach_thresholds 不能为空")
    if any(not 0.0 < t <= 1.0 for t in thresholds):
        raise ValueError(f"ablation.reach_thresholds 必须在 (0, 1]：{thresholds}")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError(f"ablation.reach_thresholds 有重复项：{thresholds}")
    for name in cfg["pareto"]["objectives"]:
        if name not in METRIC_DIRECTIONS:
            raise ValueError(f"pareto.objectives 里有未知指标 {name!r}")


def _validate_pareto(cfg: dict[str, Any]) -> None:
    from .metrics import METRIC_DIRECTIONS

    pairs = [cfg["pareto"]["objectives"]] + [
        list(p) for p in cfg["pareto"].get("extra_objectives", [])
    ]
    seen: list[list[str]] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"目标对必须是两个指标，得到 {pair}")
        for name in pair:
            if name not in METRIC_DIRECTIONS:
                raise ValueError(
                    f"未知指标 {name!r}；可用：{', '.join(sorted(METRIC_DIRECTIONS))}"
                )
        # 两个轴必须不同。同一个指标写两遍不会报错，只会让前沿表的表头
        # 出现两个同名列（而且前沿退化成「按这一个指标排序」）——
        # 这类「配置写错了但图还是画出来了」最费时间。
        if pair[0] == pair[1]:
            raise ValueError(f"目标对的两个轴不能是同一个指标：{pair}")
        if list(pair) in seen:
            raise ValueError(f"重复的目标对：{pair}")
        seen.append(list(pair))


def _validate_select(cfg: dict[str, Any]) -> None:
    from .metrics import METRIC_DIRECTIONS

    sel = cfg["select"]
    if sel["rule"] not in ("max_recall_within_budget", "knee"):
        raise ValueError(
            f"未知选型规则 {sel['rule']!r}；可用：max_recall_within_budget / knee"
        )
    if sel["rule"] == "max_recall_within_budget":
        metric = sel["budget_metric"]
        if metric not in METRIC_DIRECTIONS:
            raise ValueError(
                f"select.budget_metric 是未知指标 {metric!r}；"
                f"可用：{', '.join(sorted(METRIC_DIRECTIONS))}"
            )
        if float(sel["budget"]) <= 0:
            raise ValueError("select.budget 必须为正")


def expand_specs(cfg: dict[str, Any]) -> list[Any]:
    """把设计空间展开成全部 spec。

    **展开顺序 = 配置里的嵌套顺序**，而且是刻意的：表示空间在前、检索空间在后。
    报告里的消融表、矩阵、前沿都按这个顺序读 —— 顺序变了，
    「同一批向量换索引」这类行就不再相邻，读者要自己找。
    """
    from .schema import RetrievalSpec

    r = cfg["representation"]
    q = cfg["retrieval"]
    specs: list[RetrievalSpec] = []
    for embedding in [str(e) for e in r["embeddings"]]:
        for chunking in [str(c) for c in r["chunkings"]]:
            for index in [str(i) for i in q["indexes"]]:
                for distance in [str(d) for d in q["distances"]]:
                    for params in q["search_params"][index]:
                        specs.append(
                            RetrievalSpec(
                                embedding=embedding,
                                chunking=chunking,
                                index=index,
                                distance=distance,
                                search={str(k): v for k, v in (params or {}).items()},
                            )
                        )
    return specs


def validate(cfg: dict[str, Any]) -> None:
    """把配置错误尽量挡在运行之前。"""
    _validate_corpus(cfg)
    _validate_representation(cfg)
    _validate_retrieval(cfg)
    _validate_ablation(cfg)
    _validate_pareto(cfg)
    _validate_select(cfg)

    # 展开后 spec_id 必须唯一 —— DAG 节点 id、矩阵行名、前沿标签全用它
    specs = expand_specs(cfg)
    if not specs:
        raise ValueError("设计空间展开后为空，没有任何可跑的配置")
    ids = [s.spec_id for s in specs]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"设计空间展开后出现重复的 spec_id：{dupes}")

    # 消融参考配置必须真的在设计空间里 —— 否则消融表没有基准行，
    # 而报告会安静地少一整节（这类「静默少一块」最费时间）
    ref = str(cfg["ablation"]["reference"])
    if ref not in ids:
        raise ValueError(
            f"ablation.reference = {ref!r} 不在设计空间里。"
            f"相近的可选：{', '.join(i for i in ids if i[:3] == ref[:3]) or '（无）'}"
        )

    # 数据文件必须存在，且查询的 ground truth 必须指向真实文档
    documents = load_jsonl(resolve_corpus_path(cfg, "documents"))
    queries = load_jsonl(resolve_corpus_path(cfg, "queries"))
    if not documents:
        raise ValueError("文档集为空")
    if not queries:
        raise ValueError("查询集为空")
    doc_ids = {str(d["doc_id"]) for d in documents}
    for q in queries:
        relevant = [str(x) for x in q.get("relevant", [])]
        if not relevant:
            raise ValueError(
                f"查询 {q.get('query_id')!r} 没有 ground truth 相关文档 —— "
                "召回率算不出来，评测会安静地把它当 0 分"
            )
        missing = [x for x in relevant if x not in doc_ids]
        if missing:
            raise ValueError(
                f"查询 {q.get('query_id')!r} 的相关文档不存在：{missing}"
            )

    # k 不能大于文档数（否则 recall@k 恒等于「相关文档全召回」，毫无信息量）
    k = int(cfg["retrieval"]["k"])
    if k > len(doc_ids):
        raise ValueError(
            f"retrieval.k = {k} 大于文档总数 {len(doc_ids)}，"
            "召回率会恒为 1，度量失去意义"
        )
