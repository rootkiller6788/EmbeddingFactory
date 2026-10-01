"""度量、支配、超体积、消融与归因。

这一层要回答 README 的**两条验收标准**，而两条都要求「可归因到具体变量」。
所以测试的重点不是「函数能跑」，而是**归因的方法论有没有被破坏**：

- 单变量消融必须真的「只动一个变量」（``ablation_groups`` 的 index / search 特判）；
- 空间归因的极差必须在「只差这一个轴」的组内算；
- 前沿点必须真的非支配，且目标向量逐位相同的重复项要被识别出来。
"""

from __future__ import annotations

import pytest

from embeddingfactory import compare
from embeddingfactory.metrics import (
    COST_METRICS,
    METRIC_DIRECTIONS,
    METRIC_LABELS,
    QUALITY_METRICS,
    TIMING_METRICS,
    empty_metrics,
    has_timing_axis,
    is_deterministic_axis,
    metric_line,
    point_of,
    summarize,
)
from embeddingfactory.retrieve import (
    dcg_at_k,
    first_relevant_rank,
    merge_to_docs,
    ndcg_at_k,
    recall_at_k,
)
from embeddingfactory.schema import DESIGN_AXES, RetrievalMetrics, RetrievalRun, RetrievalSpec


# ----------------------------------------------------------------------
# 度量定义
# ----------------------------------------------------------------------
def test_metric_tables_are_consistent() -> None:
    assert TIMING_METRICS == frozenset({"query_latency_ms", "build_seconds"})
    assert set(METRIC_LABELS) == set(METRIC_DIRECTIONS)
    assert COST_METRICS <= set(METRIC_DIRECTIONS)
    assert QUALITY_METRICS <= set(METRIC_DIRECTIONS)
    assert not (COST_METRICS & QUALITY_METRICS)
    for name in QUALITY_METRICS:
        assert METRIC_DIRECTIONS[name] == "max"
    for name in COST_METRICS:
        assert METRIC_DIRECTIONS[name] == "min"


def test_has_timing_axis() -> None:
    assert has_timing_axis(["n_distance_computations", "recall_at_k"]) is False
    assert has_timing_axis(["query_latency_ms", "recall_at_k"]) is True
    assert has_timing_axis(["build_seconds", "recall_at_k"]) is True
    assert is_deterministic_axis(["n_distance_computations", "recall_at_k"]) is True
    assert is_deterministic_axis(["query_latency_ms", "recall_at_k"]) is False


def _fake_run(spec_id: str, **metrics) -> RetrievalRun:
    spec = RetrievalSpec(
        embedding=metrics.pop("embedding", "tfidf"),
        chunking=metrics.pop("chunking", "sentence"),
        index=metrics.pop("index", "flat"),
        distance=metrics.pop("distance", "cos"),
        search=metrics.pop("search", {}),
    )
    base = dict(
        recall_at_k=0.5,
        ndcg_at_k=0.4,
        mrr=0.3,
        n_distance_computations=10.0,
        query_latency_ms=1.0,
        build_seconds=0.01,
        index_bytes=1024,
        n_queries=24,
        n_chunks=92,
        k=5,
    )
    base.update(metrics)
    return RetrievalRun(spec=spec, metrics=RetrievalMetrics(**base))


def test_point_of_covers_every_metric() -> None:
    point = point_of(_fake_run("x"))
    assert set(point) == set(METRIC_DIRECTIONS)
    assert all(isinstance(v, float) for v in point.values())


def test_empty_metrics_is_all_zero() -> None:
    m = empty_metrics(k=5)
    assert m.recall_at_k == 0.0
    assert m.n_distance_computations == 0.0
    assert m.k == 5


def test_metric_line_and_summarize() -> None:
    run = _fake_run("tfidf-sentence-flat-cos")
    assert "tfidf-sentence-flat-cos" in metric_line(run)
    payload = summarize(run)
    assert payload["spec_id"] == "tfidf-sentence-flat-cos"
    assert payload["representation"] == {"embedding": "tfidf", "chunking": "sentence"}
    assert payload["retrieval"]["index"] == "flat"


# ----------------------------------------------------------------------
# 检索指标
# ----------------------------------------------------------------------
def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b"], ["a", "b"]) == 1.0
    assert recall_at_k(["a", "x"], ["a", "b"]) == 0.5
    assert recall_at_k(["x", "y"], ["a"]) == 0.0
    assert recall_at_k(["a"], []) == 0.0


def test_recall_deduplicates_ranked() -> None:
    """同一个文档不该因为出现两次就把召回率算成 2 倍。"""
    assert recall_at_k(["a", "a"], ["a", "b"]) == 0.5


def test_dcg_and_ndcg_perfect_ranking() -> None:
    ranked = ["a", "b", "c"]
    assert ndcg_at_k(ranked, ["a", "b", "c"], k=3) == pytest.approx(1.0)
    assert dcg_at_k(ranked, ["a", "b", "c"]) == pytest.approx(
        1.0 + 1 / 1.5849625007211563 + 1 / 2.0, abs=1e-9
    )


def test_ndcg_penalises_late_hits() -> None:
    """相关文档排在第 2 位，就该低于满分。"""
    score = ndcg_at_k(["x", "a"], ["a"], k=2)
    assert score == pytest.approx((1 / 1.5849625007211563) / 1.0, abs=1e-9)
    assert score < 1.0


def test_ndcg_idcg_is_capped_at_k() -> None:
    """IDCG 的分母上限是 ``min(k, |relevant|)``。

    相关文档比 k 还多时，理想排序也只能放 k 个 —— 不设这个上限会让
    nDCG 永远小于 1，读者会以为排序还有改进空间。
    """
    ranked = ["a", "b"]
    relevant = ["a", "b", "c", "d"]
    assert ndcg_at_k(ranked, relevant, k=2) == pytest.approx(1.0)


def test_ndcg_empty_relevant() -> None:
    assert ndcg_at_k(["a"], [], k=1) == 0.0


def test_first_relevant_rank() -> None:
    assert first_relevant_rank(["x", "a"], ["a"]) == 2
    assert first_relevant_rank(["a"], ["a"]) == 1
    assert first_relevant_rank(["x"], ["a"]) == 0


def test_merge_to_docs_keeps_best_rank_per_doc() -> None:
    """同一篇文档只保留第一次出现的位置 —— 它是「分块策略会改变召回率」
    这个结论的全部来源。"""
    chunk_docs = ["d1", "d1", "d2", "d3", "d2"]
    assert merge_to_docs([0, 1, 2, 3, 4], chunk_docs, k=3) == ["d1", "d2", "d3"]
    assert merge_to_docs([0, 1, 2, 3, 4], chunk_docs, k=2) == ["d1", "d2"]
    assert merge_to_docs([], chunk_docs, k=5) == []


# ----------------------------------------------------------------------
# 支配 / 超体积 / 拐点
# ----------------------------------------------------------------------
def test_dominates() -> None:
    assert compare.dominates((1.0, 1.0), (2.0, 2.0)) is True
    assert compare.dominates((1.0, 3.0), (2.0, 2.0)) is False
    assert compare.dominates((1.0, 1.0), (1.0, 1.0)) is False, "相等不算支配"
    assert compare.dominates((1.0, 2.0), (2.0, 2.0)) is True, "一维严格更好即可"


def test_to_min_form_respects_directions() -> None:
    point = {"recall_at_k": 0.8, "n_distance_computations": 20.0}
    assert compare.to_min_form(point, ["n_distance_computations", "recall_at_k"]) == (
        20.0,
        -0.8,
    )


def test_non_dominated_picks_the_tradeoff_curve() -> None:
    points = {
        "a": {"x": 1.0, "y": 3.0},
        "b": {"x": 2.0, "y": 2.0},
        "c": {"x": 3.0, "y": 1.0},
        "d": {"x": 5.0, "y": 5.0},
    }
    directions = dict(METRIC_DIRECTIONS)
    # 借 x/y 走一遍 to_min_form：把两个轴都当 min 用
    import embeddingfactory.compare as cmp

    original = cmp.METRIC_DIRECTIONS
    cmp.METRIC_DIRECTIONS = {**directions, "x": "min", "y": "min"}
    try:
        assert compare.non_dominated(["a", "b", "c", "d"], points, ["x", "y"]) == [
            "a",
            "b",
            "c",
        ]
    finally:
        cmp.METRIC_DIRECTIONS = original


def test_hypervolume_of_unit_square() -> None:
    """一个点 (0,0)、参考点 (1,1) → 面积 1。"""
    assert compare.hypervolume_2d([(0.0, 0.0)], (1.0, 1.0)) == pytest.approx(1.0)


def test_hypervolume_ignores_dominated_points() -> None:
    """调用方传进来的点若含被支配点，阶梯会算错而**不会报错** ——
    所以 ``hypervolume_2d`` 自己先重算一遍非支配集。"""
    keep = compare.hypervolume_2d([(0.0, 0.0)], (1.0, 1.0))
    with_dominated = compare.hypervolume_2d([(0.0, 0.0), (0.5, 0.5)], (1.0, 1.0))
    assert with_dominated == pytest.approx(keep)


def test_hv_contributions_sum_equals_total_hypervolume() -> None:
    """前沿点的增量之和恒等于总超体积 —— 这条恒等式守着一整套面积实现。"""
    points = {
        "a": {"x": 1.0, "y": 4.0},
        "b": {"x": 2.0, "y": 3.0},
        "c": {"x": 3.0, "y": 2.0},
        "d": {"x": 4.0, "y": 1.0},
        "e": {"x": 6.0, "y": 6.0},
    }
    import embeddingfactory.compare as cmp

    original = cmp.METRIC_DIRECTIONS
    cmp.METRIC_DIRECTIONS = {**original, "x": "min", "y": "min"}
    try:
        ids = sorted(points)
        contributions = compare.hv_contributions(ids, points, ["x", "y"])
        total = compare.hypervolume_of(ids, points, ["x", "y"])
        assert sum(contributions.values()) == pytest.approx(total, abs=1e-9)
        # 被支配的点不产生箱体，所以它根本不出现在增量表里
        assert "e" not in contributions
        assert set(contributions) == {"a", "b", "c", "d"}
    finally:
        cmp.METRIC_DIRECTIONS = original


def test_reference_point_adds_margin() -> None:
    points = {"a": {"x": 2.0, "y": 4.0}}
    import embeddingfactory.compare as cmp

    original = cmp.METRIC_DIRECTIONS
    cmp.METRIC_DIRECTIONS = {**original, "x": "min", "y": "min"}
    try:
        ref = compare.reference_point(["a"], points, ["x", "y"])
        assert ref == pytest.approx(
            (2.0 * (1 + compare.REFERENCE_MARGIN), 4.0 * (1 + compare.REFERENCE_MARGIN))
        )
    finally:
        cmp.METRIC_DIRECTIONS = original


def test_knee_point_picks_the_middle_of_a_curve() -> None:
    points = {
        "a": {"x": 0.0, "y": 0.0},
        "b": {"x": 1.0, "y": 5.0},
        "c": {"x": 10.0, "y": 10.0},
    }
    import embeddingfactory.compare as cmp

    original = cmp.METRIC_DIRECTIONS
    cmp.METRIC_DIRECTIONS = {**original, "x": "min", "y": "max"}
    try:
        assert compare.knee_point(["a", "b", "c"], points, ["x", "y"]) == "b"
    finally:
        cmp.METRIC_DIRECTIONS = original


def test_knee_point_small_frontier() -> None:
    points = {"a": {"x": 0.0, "y": 1.0}, "b": {"x": 1.0, "y": 0.0}}
    import embeddingfactory.compare as cmp

    original = cmp.METRIC_DIRECTIONS
    cmp.METRIC_DIRECTIONS = {**original, "x": "min", "y": "max"}
    try:
        assert compare.knee_point(["a", "b"], points, ["x", "y"]) == "a"
        assert compare.knee_point([], points, ["x", "y"]) is None
    finally:
        cmp.METRIC_DIRECTIONS = original


def test_pareto_representatives_dedupes_identical_objectives() -> None:
    """目标向量逐位相同的两个配置互不支配（支配要求至少一维**严格**更好），
    所以会一起留在前沿上，读者会以为那是 bug。必须按向量去重。"""
    points = {
        "b": {"x": 1.0, "y": 1.0},
        "a": {"x": 1.0, "y": 1.0},  # 与 b 逐位相同
        "c": {"x": 2.0, "y": 0.5},
    }
    import embeddingfactory.compare as cmp

    original = cmp.METRIC_DIRECTIONS
    cmp.METRIC_DIRECTIONS = {**original, "x": "min", "y": "min"}
    try:
        reps, dropped = compare.pareto_representatives(["b", "a", "c"], points, ["x", "y"])
        # 代表取 spec_id 字典序最小的那个（确定，不依赖输入顺序）
        assert reps == ["a", "c"]
        assert dropped == ["b"]
        # 换个传入顺序，结果必须一样 —— 否则「字典序最小的代表」这句话不成立
        reps2, dropped2 = compare.pareto_representatives(["c", "a", "b"], points, ["x", "y"])
        assert (reps2, dropped2) == (reps, dropped)
        assert len(reps) + len(dropped) == len(
            compare.non_dominated(["b", "a", "c"], points, ["x", "y"])
        )
    finally:
        cmp.METRIC_DIRECTIONS = original


# ----------------------------------------------------------------------
# 前沿（真结果）
# ----------------------------------------------------------------------
def test_every_frontier_member_is_non_dominated(result) -> None:
    points = {r.spec_id: point_of(r) for r in result.runs}
    ids = sorted(points)
    for key, members in result.frontiers.items():
        pair = key.split("+")
        assert members, f"{key} 的前沿是空的"
        assert set(members) <= set(compare.non_dominated(ids, points, pair))


def test_main_frontier_has_expected_shape(result, cfg) -> None:
    """主前沿：成本轴是「距离计算次数」，且**可复现**（不含实测轴）。

    前沿成员按 spec_id 排序（不是按成本排序），所以不能断言 rows[0] 最便宜。
    该断言的是**端点性质**：全空间最便宜的点与召回最高的点都在前沿上
    （否则「最左端 = 成本最低、最右端 = 召回最高」这句话就不成立）。
    """
    key = compare.frontier_key(cfg["pareto"]["objectives"])
    members = result.frontiers[key]
    assert len(members) == 5, f"实测前沿成员：{members}"
    assert has_timing_axis(key.split("+")) is False

    rows = compare.frontier_rows(result, key)
    assert len(rows) == len(members)
    assert min(r["n_distance_computations"] for r in rows) == min(
        r.metrics.n_distance_computations for r in result.runs
    ), "全空间最便宜的点不在前沿上"
    # frontier_rows 里的数值是四舍五入到 6 位的，所以用 approx 比
    assert max(r["recall_at_k"] for r in rows) == pytest.approx(
        max(r.metrics.recall_at_k for r in result.runs), abs=1e-6
    ), "召回最高的点不在前沿上"
    # 前沿按成本严格递增时，召回也必须严格递增（否则中间点会被支配）
    by_cost = sorted(rows, key=lambda r: r["n_distance_computations"])
    recalls = [r["recall_at_k"] for r in by_cost]
    assert recalls == sorted(recalls)


def test_timing_frontiers_are_marked(result, cfg) -> None:
    deterministic = set(result.counters["deterministic_frontiers"])
    timing = set(result.counters["timing_frontiers"])
    assert deterministic & timing == set()
    assert deterministic | timing == set(result.frontiers)
    assert compare.frontier_key(cfg["pareto"]["objectives"]) in deterministic
    assert "query_latency_ms+recall_at_k" in timing


def test_frontier_duplicates_are_genuinely_identical(result) -> None:
    """被去重的配置，其目标向量必须与某个代表项**逐位相同**。"""
    points = {r.spec_id: point_of(r) for r in result.runs}
    duplicates = result.counters["frontier_duplicates"]
    assert duplicates, "本语料实测存在重复项（hnsw 的 ef16 / ef64 在 l2 上等价）"
    for key, dropped in duplicates.items():
        pair = key.split("+")
        reps = result.frontiers[key]
        for sid in dropped:
            assert sid not in reps
            assert any(
                compare.to_min_form(points[sid], pair) == compare.to_min_form(points[rep], pair)
                for rep in reps
            ), f"{sid} 与任何代表项都不逐位相同，那它就不该被去掉"


def test_all_objective_pairs_is_cost_times_quality(cfg) -> None:
    pairs = compare.all_objective_pairs(cfg)
    for pair in pairs:
        assert len(pair) == 2
        assert (pair[0] in COST_METRICS and pair[1] in QUALITY_METRICS) or (
            pair[1] in COST_METRICS and pair[0] in QUALITY_METRICS
        ) or pair in compare.frontier_pairs(cfg)
    for pair in compare.frontier_pairs(cfg):
        assert pair in pairs


def test_timing_axis_pairs(cfg) -> None:
    pairs = compare.timing_axis_pairs(compare.frontier_pairs(cfg))
    assert ["query_latency_ms", "recall_at_k"] in pairs
    assert ["build_seconds", "recall_at_k"] in pairs
    assert ["n_distance_computations", "recall_at_k"] not in pairs


# ----------------------------------------------------------------------
# 消融：一次只动一个变量
# ----------------------------------------------------------------------
def _differing_axes(a: RetrievalSpec, b: RetrievalSpec) -> set[str]:
    return {axis for axis in DESIGN_AXES if a.axis(axis) != b.axis(axis)}


def test_ablation_groups_cover_every_axis(result, cfg) -> None:
    groups = compare.ablation_groups(result, cfg)
    assert [g["axis"] for g in groups] == list(DESIGN_AXES)
    for group in groups:
        assert len(group["members"]) >= 2, f"{group['axis']} 组内只有一个成员 = 没比"
        assert group["baseline"] in group["members"]
        assert group["space"] in ("representation", "retrieval")


def test_ablation_groups_vary_exactly_one_axis(result, cfg) -> None:
    """**这是本项目的核心方法论**：一次动两个变量，Δ 就无法归因。

    三个「普通」轴（embedding / chunking / distance）的组内成员必须与基准
    只差这一个轴。``index`` 与 ``search`` 两个轴是特判（参考配置是 flat，
    而 flat 没有搜索参数），但特判本身也是确定的：
    换索引时给每个索引配它的**第一档**搜索参数，换搜索参数时索引固定。
    """
    by_spec = {r.spec_id: r for r in result.runs}
    for group in compare.ablation_groups(result, cfg):
        axis = group["axis"]
        base = by_spec[group["baseline"]].spec
        for sid in group["members"]:
            if sid == group["baseline"]:
                continue  # 基准与自身当然不差任何轴
            spec = by_spec[sid].spec
            diff = _differing_axes(base, spec)
            if axis in ("embedding", "chunking", "distance"):
                assert diff == {axis}, f"{axis} 组里的 {sid} 同时动了 {diff}"
            elif axis == "index":
                assert diff == {"index", "search"}, f"{sid} 动了 {diff}"
                first_level = cfg["retrieval"]["search_params"][spec.index][0]
                assert spec.search == first_level
            elif axis == "search":
                assert spec.index == base.index or diff == {"index", "search"}
                assert diff <= {"index", "search"}


def test_index_ablation_group_covers_every_index(result, cfg) -> None:
    """第一版这里静默少算：flat 没有搜索参数，所以「只把 index 换成 ivf」
    造出的 spec_id 缺 nprobe，那一行根本不存在 —— 消融表看起来正常，
    实际 5 个轴里有 2 个什么都没比。"""
    group = next(g for g in compare.ablation_groups(result, cfg) if g["axis"] == "index")
    assert len(group["members"]) == len(cfg["retrieval"]["indexes"])
    assert group["note"], "特判口径必须写进 note，否则读者不知道它和别的轴不一样"
    by_spec = {r.spec_id: r for r in result.runs}
    assert {by_spec[m].spec.index for m in group["members"]} == set(cfg["retrieval"]["indexes"])


def test_search_ablation_group_uses_richest_index(result, cfg) -> None:
    """``search`` 轴在 flat 上无从消融（只有一档），所以挑档位最多的索引当实验台。

    档位并列时取名字最小的那个（这里 ivf 与 hnsw 都是 2 档 → 选 hnsw）——
    这条并列打破规则必须确定，否则同一个设计空间会得到两张不同的消融表。
    """
    group = next(g for g in compare.ablation_groups(result, cfg) if g["axis"] == "search")
    host = sorted(
        cfg["retrieval"]["indexes"],
        key=lambda i: (-len(cfg["retrieval"]["search_params"][i]), i),
    )[0]
    assert len(group["members"]) == len(cfg["retrieval"]["search_params"][host])
    by_spec = {r.spec_id: r for r in result.runs}
    assert {by_spec[m].spec.index for m in group["members"]} == {host}
    assert group["note"]


def test_ablation_rows_are_relative_to_group_baseline(result, cfg) -> None:
    rows = compare.ablation_rows(result, cfg)
    assert len(rows) == 7, f"实测 7 行，得到 {len(rows)}"
    by_spec = {r.spec_id: r for r in result.runs}
    for row in rows:
        base = by_spec[row["baseline"]]
        run = by_spec[row["spec_id"]]
        assert row["from"] == base.spec.axis(row["axis"])
        assert row["to"] == run.spec.axis(row["axis"])
        assert row["from"] != row["to"]
        assert row["d_recall"] == pytest.approx(
            run.metrics.recall_at_k - base.metrics.recall_at_k, abs=1e-6
        )
        assert row["d_cost"] == pytest.approx(
            run.metrics.n_distance_computations - base.metrics.n_distance_computations,
            abs=1e-3,
        )


def test_single_axis_ablation_headline_numbers(result, cfg) -> None:
    """实测结论（这些数字进了报告，改了就说明口径或语料动了）：
    embedding 从 tfidf 换到 hashing 掉 0.2639；
    chunking 从 sentence 换到 fixed 掉 0.5694；
    distance 从 cos 换到 l2 掉 0.5764。"""
    rows = {(r["axis"], r["to"]): r for r in result.ablations["single_axis"]}
    assert rows[("embedding", "hashing")]["d_recall"] == pytest.approx(-0.2639, abs=1e-4)
    assert rows[("chunking", "fixed")]["d_recall"] == pytest.approx(-0.5694, abs=1e-4)
    assert rows[("distance", "l2")]["d_recall"] == pytest.approx(-0.5764, abs=1e-4)
    # 换 l2 时成本不变（两者单次成本相等，扫的向量数也一样）
    assert rows[("distance", "l2")]["d_cost"] == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# 空间归因
# ----------------------------------------------------------------------
def test_space_attribution_sorted_by_spread(result, cfg) -> None:
    rows = compare.space_attribution(result, cfg)
    assert len(rows) == len(DESIGN_AXES)
    spreads = [r["spread"] for r in rows]
    assert spreads == sorted(spreads, reverse=True)
    for row in rows:
        assert row["n_compared"] >= 2, f"{row['axis']} 组内只比了 1 个 = 极差恒为 0"
        assert row["recall_max"] >= row["recall_min"]
        # 两个字段各自四舍五入到 6 位小数，所以差值不能要求逐位相等
        assert row["spread"] == pytest.approx(row["recall_max"] - row["recall_min"], abs=1e-5)


def test_space_attribution_reports_n_compared_vs_n_values(result, cfg) -> None:
    """「比了几个」小于「取值数」时，那是**没比**，不是**没影响**。"""
    rows = {r["axis"]: r for r in compare.space_attribution(result, cfg)}
    for axis, row in rows.items():
        assert row["n_values"] == len(result.axis_values(axis))
        assert row["n_compared"] <= row["n_values"]
    # search 轴的取值数是 3（exact / np1 / np3 …），但只能在自己的实验台上比
    assert rows["search"]["n_compared"] >= 2


def test_space_attribution_headline_order(result, cfg) -> None:
    """实测：distance 0.5764 > chunking 0.5694 > embedding 0.2639
    > index 0.1528 > search 0.0625。

    注意这是**局部量**（其余轴固定在参考配置上），报告里有警告。
    """
    rows = {r["axis"]: r for r in compare.space_attribution(result, cfg)}
    assert rows["distance"]["spread"] == pytest.approx(0.5764, abs=1e-4)
    assert rows["chunking"]["spread"] == pytest.approx(0.5694, abs=1e-4)
    assert rows["embedding"]["spread"] == pytest.approx(0.2639, abs=1e-4)
    order = [r["axis"] for r in compare.space_attribution(result, cfg)]
    assert order[0] == "distance"
    assert order[-1] == "search"


# ----------------------------------------------------------------------
# 表示空间上限 / 达额成本 / 预算敏感性
# ----------------------------------------------------------------------
def test_representation_ceiling_uses_only_flat(result, cfg) -> None:
    """上限必须只用精确检索算 —— 近似索引的召回率里混着「索引掉了多少」。"""
    ceiling = compare.representation_ceiling(result, cfg)
    assert ceiling["rows"]
    for row in ceiling["rows"]:
        run = result.find(row["spec_id"])
        assert run.spec.index == "flat"
        assert run.spec.distance == result.reference_run().spec.distance
    assert ceiling["ceiling_recall"] == max(r["recall_at_k"] for r in ceiling["rows"])
    assert ceiling["rows"][0]["gap_to_ceiling"] <= 0
    assert sum(1 for r in ceiling["rows"] if r["is_ceiling"]) == 1


def test_representation_ceiling_headline(result, cfg) -> None:
    """实测：tfidf/sentence 在精确检索下 0.8333；tfidf/fixed 只有 0.2639。
    分块粒度对召回率上限的影响比换 embedding 还大。"""
    ceiling = compare.representation_ceiling(result, cfg)
    by_key = {(r["embedding"], r["chunking"]): r for r in ceiling["rows"]}
    assert ceiling["ceiling_recall"] == pytest.approx(0.8333, abs=1e-4)
    assert ceiling["ceiling_spec"] == "tfidf-sentence-flat-cos"
    assert by_key[("tfidf", "fixed")]["recall_at_k"] == pytest.approx(0.2639, abs=1e-4)


def test_best_embedding_flips_with_chunking(result, cfg) -> None:
    """「哪个 embedding 最好」必须连着分块一起说。

    实测：``sentence`` 下 tfidf 最优（0.8333）、hashing 最差（0.5694）；
    ``fixed`` 下**正好反过来** —— hashing 0.5208 成了最好的，tfidf 掉到 0.2639。
    原因是字符 n-gram 不依赖词边界，块被切断时损失小；词袋则整词丢失。

    这是报告第五节「极差是局部量」那条警告的实证依据 ——
    如果哪天这个翻转消失了，那句警告就该跟着改，而不是留在那里骗人。
    """
    rows = compare.representation_ceiling(result, cfg)["rows"]
    best: dict[str, str] = {}
    for chunking in ("sentence", "fixed"):
        sub = [r for r in rows if r["chunking"] == chunking]
        best[chunking] = max(sub, key=lambda r: (r["recall_at_k"], r["embedding"]))["embedding"]
    assert best["sentence"] == "tfidf"
    assert best["fixed"] == "hashing"


def test_reach_cost_is_monotone_in_threshold(result, cfg) -> None:
    """目标比例越高，需要的成本不会更低（更严的约束不可能更便宜）。"""
    reach = compare.reach_cost(result, cfg)
    rows = [r for r in reach["rows"] if r["reachable"]]
    assert rows
    for row in rows:
        assert row["recall_at_k"] >= row["target_recall"] - 1e-9
        assert row["cost_vs_exact"] is not None
        assert 0.0 < row["cost_vs_exact"] <= 1.0 + 1e-9
    costs = [r["n_distance_computations"] for r in rows]
    assert costs == sorted(costs)


def test_reach_cost_at_100pct_is_still_cheaper_than_exact(result, cfg) -> None:
    """达到「上限的 100%」也不需要真的跑精确检索。

    实测：上限 0.8333 由 ``tfidf-sentence-flat-cos`` 取得（92 次距离/查询），
    但 ``tfidf-sentence-ivf-cos-np3`` 用 61.7 次就拿到了同样的 0.8333 ——
    成本只有精确检索的 0.67 倍。

    这条断言把「近似索引的价值」落成一个具体数字：**它不只是掉一点召回换成本，
    而是在这个语料上能免费拿到同样的召回**。如果哪天 np3 拿不到了，
    说明 ivf 的分簇质量退化了。
    """
    reach = compare.reach_cost(result, cfg)
    full = next(r for r in reach["rows"] if r["threshold"] == 1.0)
    assert full["reachable"] is True
    assert full["recall_at_k"] == pytest.approx(reach["ceiling_recall"], abs=1e-6)
    assert full["spec_id"] != reach["ceiling_spec"], "最省的 100% 配置不该是精确检索本身"
    assert full["index"] != "flat"
    assert 0.0 < full["cost_vs_exact"] < 1.0


def test_budget_sensitivity_has_two_separate_gaps(result, cfg) -> None:
    """两列差距必须分开：预算约束造成的损失 vs 向量质量造成的损失。"""
    rows = compare.budget_sensitivity(result, cfg)
    assert rows
    for row in rows:
        assert row["gap_to_ceiling"] <= 1e-9
        assert row["embedding_spread"] >= 0.0
        assert row["n_within"] >= 1
    budgets = [r["budget"] for r in rows]
    assert budgets == sorted(budgets)
    # 预算放宽后，可用配置数只增不减
    counts = [r["n_within"] for r in rows]
    assert counts == sorted(counts)


def test_budget_sensitivity_gap_shrinks_with_budget(result, cfg) -> None:
    rows = compare.budget_sensitivity(result, cfg)
    assert rows[-1]["gap_to_ceiling"] == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# 选型与基线
# ----------------------------------------------------------------------
def test_select_respects_budget(result, cfg) -> None:
    chosen, reason = compare.select(result, cfg)
    assert chosen in result.known_spec_ids()
    assert result.selected == chosen
    run = result.find(chosen)
    budget = float(cfg["select"]["budget"])
    assert run.metrics.n_distance_computations <= budget
    assert "max_recall_within_budget" in reason
    # 预算内没有更好的
    within = [
        r for r in result.runs if r.metrics.n_distance_computations <= budget
    ]
    assert run.metrics.recall_at_k == max(r.metrics.recall_at_k for r in within)


def test_select_rule_knee_is_available(result, cfg) -> None:
    import copy

    alt = copy.deepcopy(cfg)
    alt["select"]["rule"] = "knee"
    chosen, reason = compare.select(result, alt)
    assert chosen in result.known_spec_ids()
    assert "knee" in reason


def test_baseline_reports_saving_at_same_recall(result, cfg) -> None:
    baseline = result.counters["baseline"]
    assert baseline["spec_id"] == result.reference_run().spec_id
    saving = baseline["cheapest_at_same_recall"]
    assert 0.0 < saving["cost_ratio"] <= 1.0
    assert saving["saving_pct"] == pytest.approx(100.0 * (1.0 - saving["cost_ratio"]), abs=0.01)
    assert saving["recall_at_k"] >= baseline["recall_at_k"] - 1e-9


def test_reference_run_exists_and_is_in_design_space(result, cfg) -> None:
    ref = result.reference_run()
    assert ref is not None
    assert ref.spec_id == cfg["ablation"]["reference"]
    assert ref.spec.index == "flat", "基准必须是精确检索，召回率才等于上限"
