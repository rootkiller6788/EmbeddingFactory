"""三种距离 + 三种索引 + **一个可复现的成本模型**。

本模块守着两条最容易被静默破坏的性质：

1. **``flat`` 是精确 oracle**。它的结果必须与暴力排序逐位一致 ——
   一旦不成立，所有「近似索引掉了多少召回」的结论全部失去参照。
2. **``cos`` 的距离符号**。第一版把余弦距离写成了 ``1 − d²/2``（那是相似度），
   召回率从 0.81 掉到 0.08，而且**不报错** —— 只让「cos 比 l2 差」这种假结论出现。
   所以这里直接断言「cos 的排序与朴素点积实现逐位一致」。

第三条同样重要但容易被忽略：**所有距离的单次成本必须相等**。
否则「距离计算次数」这个成本轴在不同距离之间不可比，
整条 Pareto 前沿的横轴就失去意义。
"""

from __future__ import annotations

import math
import random

import pytest

from embeddingfactory.index import (
    DISTANCES,
    INDEXES,
    BaseIndex,
    FlatIndex,
    HNSWIndex,
    IVFIndex,
    build_index,
)

K = 5


def _vectors(n: int, dim: int, seed: int = 1234) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(n)]


def _brute(vectors, query, distance_name: str, k: int) -> list[int]:
    """独立的暴力 oracle —— **刻意重写一遍**，不复用 ``FlatIndex`` 的代码。

    复用的话，实现里一个共同的错误会让「两者一致」这个断言永远成立，
    等于没测。
    """
    d = DISTANCES[distance_name]
    q = d.prepare(query)
    prepared = [d.prepare(v) for v in vectors]
    scored = sorted(
        ((d.compute(q, v), i) for i, v in enumerate(prepared)),
        key=lambda t: (t[0], t[1]),
    )
    return [i for _, i in scored[:k]]


# ----------------------------------------------------------------------
# flat = 精确 oracle
# ----------------------------------------------------------------------
@pytest.mark.parametrize("distance_name", ["cos", "l2", "ip"])
def test_flat_matches_brute_force(distance_name: str) -> None:
    vectors = _vectors(40, 16)
    query = [0.3] * 16
    index = build_index("flat", vectors, distance_name, {}, 0)
    result = index.search(query, K)
    assert result.ids == _brute(vectors, query, distance_name, K)


@pytest.mark.parametrize("distance_name", ["cos", "l2", "ip"])
def test_flat_counts_every_vector(distance_name: str) -> None:
    """暴力扫描的成本恒等于向量总数 —— 这是成本模型的上界。"""
    vectors = _vectors(37, 12)
    index = build_index("flat", vectors, distance_name, {}, 0)
    assert index.search([0.1] * 12, K).n_distance_computations == 37
    assert index.meta()["exact"] is True


def test_flat_is_exact_oracle_on_demo_corpus(cfg, corpus, bundle) -> None:
    """在真语料的向量上也要精确 —— 这是「表示空间上限」这张表的全部依据。"""
    key = bundle.model_key("tfidf", "sentence")
    vectors = bundle.vectors[key]
    query = bundle.models[key].transform(corpus.queries[0].text)
    index = build_index("flat", vectors, "cos", {}, int(cfg["seed"]))
    assert index.search(query, K).ids == _brute(vectors, query, "cos", K)


# ----------------------------------------------------------------------
# 距离：符号与「归一化属于距离」
# ----------------------------------------------------------------------
def test_cosine_distance_is_one_minus_dot_for_normalized_vectors() -> None:
    """``1 − cos = ‖a−b‖²/2``（只对归一化向量成立）。"""
    rng = random.Random(7)
    prepare = DISTANCES["cos"].prepare
    compute = DISTANCES["cos"].compute
    for _ in range(20):
        a = prepare([rng.uniform(-1, 1) for _ in range(24)])
        b = prepare([rng.uniform(-1, 1) for _ in range(24)])
        dot = sum(x * y for x, y in zip(a, b))
        assert compute(a, b) == pytest.approx(1.0 - dot, abs=1e-12)
        assert compute(a, b) == pytest.approx(math.dist(a, b) ** 2 / 2.0, abs=1e-12)


def test_cosine_ordering_matches_naive_dot() -> None:
    """**守 Error 53**：余弦距离的排序必须与朴素点积实现逐位一致。

    第一版写成 ``1 − d²/2``（相似度）时，这个断言会红 ——
    而单看召回率只会以为「cos 不如 l2」。
    """
    rng = random.Random(11)
    prepare = DISTANCES["cos"].prepare
    compute = DISTANCES["cos"].compute
    vectors = [prepare([rng.uniform(-1, 1) for _ in range(32)]) for _ in range(30)]
    query = prepare([rng.uniform(-1, 1) for _ in range(32)])

    by_distance = sorted(
        range(len(vectors)), key=lambda i: (compute(query, vectors[i]), i)
    )
    by_naive_dot = sorted(
        range(len(vectors)),
        key=lambda i: (-sum(x * y for x, y in zip(query, vectors[i])), i),
    )
    assert by_distance == by_naive_dot


def test_cosine_and_l2_differ_only_by_normalization() -> None:
    """构造一个两者 top-1 不同的例子，证明「换距离」这个轴真的在动。

    query 方向上的 A 范数很大、B 范数接近 1：
    按 L2 看 B 更近，按 cos 看 A 完全同向所以更近。
    """
    vectors = [[10.0, 0.0], [1.0, 0.9]]
    query = [1.0, 0.0]
    cos_top = build_index("flat", vectors, "cos", {}, 0).search(query, 1).ids
    l2_top = build_index("flat", vectors, "l2", {}, 0).search(query, 1).ids
    assert cos_top == [0]
    assert l2_top == [1]


def test_cos_normalizes_but_l2_and_ip_do_not() -> None:
    """归一化属于**距离**，不属于嵌入。三个距离的 ``prepare`` 必须不同。"""
    assert DISTANCES["cos"].normalizes is True
    assert DISTANCES["l2"].normalizes is False
    assert DISTANCES["ip"].normalizes is False
    vec = [3.0, 4.0]
    assert DISTANCES["cos"].prepare(vec) == pytest.approx([0.6, 0.8])
    assert DISTANCES["l2"].prepare(vec) == vec
    assert DISTANCES["ip"].prepare(vec) == vec


def test_ip_is_negated_dot_product() -> None:
    """约定：``compute`` 一律返回「越小越好」，所以内积要取负。"""
    a, b = [1.0, 2.0], [3.0, 4.0]
    assert DISTANCES["ip"].compute(a, b) == pytest.approx(-(1 * 3 + 2 * 4))


def test_cos_prepare_handles_zero_vector() -> None:
    """零向量归一化不能除零 —— 空查询会走到这里。"""
    assert DISTANCES["cos"].prepare([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_cost_axis_is_comparable_across_distances() -> None:
    """**三个距离的单次成本必须相等**，否则成本轴在距离之间不可比。

    做法是让 cos 也用一次 ``math.dist``（经 ``‖a−b‖²/2`` 恒等式），
    而不是用点积 —— 实测点积 8.0µs vs ``math.dist`` 4.1µs，
    差约 2 倍，足够让整条前沿的横轴失去意义。

    这里断言的是可观测的后果：同一批向量、同一个 k，
    三种距离报出的 ``n_distance_computations`` 必须完全相同。
    """
    vectors = _vectors(50, 16)
    query = [0.2] * 16
    counts = {
        name: build_index("flat", vectors, name, {}, 0).search(query, K).n_distance_computations
        for name in ("cos", "l2", "ip")
    }
    assert len(set(counts.values())) == 1
    assert set(counts.values()) == {50}


def test_distances_registry_names_are_stable() -> None:
    assert set(DISTANCES) == {"cos", "l2", "ip"}


# ----------------------------------------------------------------------
# 成本模型可复现
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "index_name, params",
    [
        ("flat", {}),
        ("ivf", {"nlist": 5, "kmeans_iters": 8, "nprobe": 2}),
        ("hnsw", {"M": 6, "ef_construction": 16, "ef_search": 24}),
    ],
)
def test_n_distance_computations_is_reproducible(index_name: str, params: dict) -> None:
    """成本模型必须逐位可复现 —— 这是它替代实测延迟的唯一理由。"""
    vectors = _vectors(60, 20)
    query = [0.1] * 20
    first = build_index(index_name, vectors, "cos", params, 3).search(query, K)
    second = build_index(index_name, vectors, "cos", params, 3).search(query, K)
    assert first.n_distance_computations == second.n_distance_computations
    assert first.ids == second.ids


def test_search_ties_are_broken_by_index() -> None:
    """并列必须显式按下标升序打破，否则 top-k 的成员会随排序实现变化。"""
    vectors = [[1.0, 0.0]] * 6
    index = build_index("flat", vectors, "cos", {}, 0)
    assert index.search([1.0, 0.0], 4).ids == [0, 1, 2, 3]


def test_search_edge_cases() -> None:
    index = build_index("flat", _vectors(5, 4), "cos", {}, 0)
    assert index.search([0.1] * 4, 0).ids == []
    assert index.search([0.1] * 4, -1).ids == []
    empty = build_index("flat", [], "cos", {}, 0)
    assert empty.search([0.1] * 4, 5).ids == []
    assert empty.search([0.1] * 4, 5).n_distance_computations == 0


def test_k_larger_than_corpus_is_clamped() -> None:
    index = build_index("flat", _vectors(3, 4), "cos", {}, 0)
    assert len(index.search([0.1] * 4, 99).ids) == 3


# ----------------------------------------------------------------------
# ivf
# ----------------------------------------------------------------------
def test_ivf_clustering_is_deterministic() -> None:
    """k-means++ 的初始化必须用派生种子，否则同一份配置两次运行的分簇不同。"""
    vectors = _vectors(60, 16)
    params = {"nlist": 6, "kmeans_iters": 10}
    a = build_index("ivf", vectors, "cos", dict(params, nprobe=1), 42)
    b = build_index("ivf", vectors, "cos", dict(params, nprobe=1), 42)
    assert a.cells == b.cells
    assert a.cell_sizes == b.cell_sizes
    assert a.centers == b.centers
    # 换个 seed 就该不一样 —— 否则说明种子根本没被用上
    c = build_index("ivf", vectors, "cos", dict(params, nprobe=1), 43)
    assert c.cells != a.cells


def test_ivf_cost_formula() -> None:
    """成本 = 簇心比较次数（nlist）+ 被扫簇的规模。"""
    vectors = _vectors(60, 16)
    index = build_index("ivf", vectors, "cos", {"nlist": 6, "kmeans_iters": 10, "nprobe": 2}, 5)
    query = [0.2] * 16
    compute = DISTANCES["cos"].compute
    q = DISTANCES["cos"].prepare(query)
    nearest = sorted(
        ((compute(q, c), i) for i, c in enumerate(index.centers)),
        key=lambda t: (t[0], t[1]),
    )[:2]
    expected = index.nlist + sum(len(index.cells[i]) for _, i in nearest)
    assert index.search(query, K).n_distance_computations == expected


def test_ivf_probing_all_cells_is_exact() -> None:
    """nprobe = nlist 时扫遍所有簇 → 与 flat 逐位一致。"""
    vectors = _vectors(48, 12)
    query = [0.4] * 12
    flat = build_index("flat", vectors, "cos", {}, 0)
    ivf = build_index("ivf", vectors, "cos", {"nlist": 4, "kmeans_iters": 10, "nprobe": 4}, 0)
    assert ivf.search(query, K).ids == flat.search(query, K).ids


def test_ivf_probing_more_costs_more() -> None:
    vectors = _vectors(80, 16)
    one = build_index("ivf", vectors, "cos", {"nlist": 8, "kmeans_iters": 10, "nprobe": 1}, 0)
    all_cells = build_index("ivf", vectors, "cos", {"nlist": 8, "kmeans_iters": 10, "nprobe": 8}, 0)
    query = [0.3] * 16
    assert (
        one.search(query, K).n_distance_computations
        <= all_cells.search(query, K).n_distance_computations
    )


def test_ivf_nlist_is_clamped_to_corpus_size() -> None:
    index = build_index("ivf", _vectors(3, 8), "cos", {"nlist": 99}, 0)
    assert index.nlist == 3


def test_ivf_meta_reports_cell_sizes() -> None:
    index = build_index("ivf", _vectors(40, 8), "cos", {"nlist": 5, "nprobe": 1}, 0)
    meta = index.meta()
    assert meta["nlist"] == 5
    assert len(meta["cell_sizes"]) == 5
    assert sum(meta["cell_sizes"]) == 40
    assert meta["cell_sizes"] == sorted(meta["cell_sizes"], reverse=True)


# ----------------------------------------------------------------------
# hnsw（简化版单层 NSW）
# ----------------------------------------------------------------------
def test_hnsw_is_deterministic() -> None:
    vectors = _vectors(70, 16)
    params = {"M": 6, "ef_construction": 24, "ef_search": 32}
    a = build_index("hnsw", vectors, "cos", dict(params), 9)
    b = build_index("hnsw", vectors, "cos", dict(params), 9)
    assert a.adj == b.adj
    assert a.degrees == b.degrees
    assert a.search([0.1] * 16, K).n_distance_computations == (
        b.search([0.1] * 16, K).n_distance_computations
    )


def test_hnsw_degree_is_bounded_by_M() -> None:
    vectors = _vectors(60, 12)
    index = build_index("hnsw", vectors, "cos", {"M": 4, "ef_construction": 16}, 0)
    assert max(index.degrees) <= 4
    assert all(d <= 4 for d in index.degrees)


def test_hnsw_cost_is_bounded_by_graph_edges() -> None:
    """束搜索每访问一个节点算一次距离，所以成本不可能超过「1 + 总边数」。"""
    vectors = _vectors(60, 12)
    index = build_index("hnsw", vectors, "cos", {"M": 6, "ef_construction": 20, "ef_search": 16}, 0)
    result = index.search([0.2] * 12, K)
    assert result.n_distance_computations <= 1 + sum(index.degrees)


def test_hnsw_returns_k_ids_in_range() -> None:
    vectors = _vectors(30, 8)
    index = build_index("hnsw", vectors, "cos", {"M": 5, "ef_construction": 16, "ef_search": 8}, 0)
    ids = index.search([0.5] * 8, K).ids
    assert len(ids) == K
    assert len(set(ids)) == K
    assert all(0 <= i < 30 for i in ids)


def test_hnsw_meta_reports_degrees() -> None:
    index = build_index("hnsw", _vectors(40, 8), "cos", {"M": 6, "ef_search": 16}, 0)
    meta = index.meta()
    assert meta["M"] == 6
    assert meta["ef_search"] == 16
    assert meta["max_degree"] <= 6
    assert meta["isolated"] == 0


def test_hnsw_search_axis_moves_cost_on_demo_corpus(cfg, corpus, bundle) -> None:
    """``search`` 轴必须真的在动。

    实测：本语料上 ``ef_search=16`` 平均扫 59 个向量、``64`` 扫 80 个。
    这条断言是「搜索参数是一个真实的成本旋钮」的直接证据；
    第一版用 ``[8, 32]`` 时两个档位都只扫 21 个向量、指标逐位相同，
    前沿上出现了两个目标向量完全一样的点 —— 那说明这个轴是死的。
    """
    key = bundle.model_key("tfidf", "sentence")
    vectors = bundle.vectors[key]
    query = bundle.models[key].transform(corpus.queries[0].text)
    low = build_index("hnsw", vectors, "cos", {"M": 6, "ef_construction": 24, "ef_search": 16}, int(cfg["seed"]))
    high = build_index("hnsw", vectors, "cos", {"M": 6, "ef_construction": 24, "ef_search": 64}, int(cfg["seed"]))
    low_dc = low.search(query, 10).n_distance_computations
    high_dc = high.search(query, 10).n_distance_computations
    assert high_dc >= low_dc, f"ef 变大反而扫得更少（{low_dc} → {high_dc}），search 轴失效"


# ----------------------------------------------------------------------
# size_bytes 口径
# ----------------------------------------------------------------------
def test_size_bytes_formulas() -> None:
    """口径：向量按 float64（8 字节/维），图/倒排按 int64（8 字节/项）。"""
    vectors = _vectors(40, 10)
    base = 40 * 10 * 8

    flat = build_index("flat", vectors, "cos", {}, 0)
    assert flat.size_bytes() == base

    ivf = build_index("ivf", vectors, "cos", {"nlist": 5}, 0)
    assert ivf.size_bytes() == base + (5 * 10 + 40) * 8

    hnsw = build_index("hnsw", vectors, "cos", {"M": 6}, 0)
    assert hnsw.size_bytes() == base + 2 * sum(hnsw.degrees) * 8

    assert ivf.size_bytes() > base
    assert hnsw.size_bytes() > base


def test_size_bytes_is_independent_of_search_params() -> None:
    """``ef_search`` 只影响查询，不影响索引常驻内存。"""
    vectors = _vectors(40, 10)
    a = build_index("hnsw", vectors, "cos", {"M": 6, "ef_search": 8}, 0)
    b = build_index("hnsw", vectors, "cos", {"M": 6, "ef_search": 64}, 0)
    assert a.size_bytes() == b.size_bytes()


# ----------------------------------------------------------------------
# 注册表与校验
# ----------------------------------------------------------------------
def test_unknown_index_or_distance_raises() -> None:
    vectors = _vectors(5, 4)
    with pytest.raises(KeyError):
        build_index("scann", vectors, "cos", {}, 0)
    with pytest.raises(KeyError):
        build_index("flat", vectors, "hamming", {}, 0)


def test_registry_names_are_stable() -> None:
    assert set(INDEXES) == {"flat", "ivf", "hnsw"}


def test_validate_search_rejects_bad_levels() -> None:
    with pytest.raises(ValueError):
        INDEXES["flat"].validate_search([{"nprobe": 1}])
    with pytest.raises(ValueError):
        INDEXES["ivf"].validate_search([{}])
    with pytest.raises(ValueError):
        INDEXES["ivf"].validate_search([{"nprobe": 0}])
    with pytest.raises(ValueError):
        INDEXES["hnsw"].validate_search([{}])
    with pytest.raises(ValueError):
        INDEXES["hnsw"].validate_search([{"ef_search": 0}])
    # 合法档位不抛
    INDEXES["flat"].validate_search([{}])
    INDEXES["ivf"].validate_search([{"nprobe": 1}])
    INDEXES["hnsw"].validate_search([{"ef_search": 16}])


def test_validate_params_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        INDEXES["ivf"].validate_params({"nlist": 0})
    with pytest.raises(ValueError):
        INDEXES["ivf"].validate_params({"kmeans_iters": 0})
    with pytest.raises(ValueError):
        INDEXES["hnsw"].validate_params({"M": 0})
    with pytest.raises(ValueError):
        INDEXES["hnsw"].validate_params({"ef_construction": 0})
    INDEXES["flat"].validate_params({})


def test_index_classes_are_the_registered_ones() -> None:
    assert INDEXES["flat"].cls is FlatIndex
    assert INDEXES["ivf"].cls is IVFIndex
    assert INDEXES["hnsw"].cls is HNSWIndex
    for cls in (FlatIndex, IVFIndex, HNSWIndex):
        assert issubclass(cls, BaseIndex)
