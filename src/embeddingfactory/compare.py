"""对比、前沿、消融与归因。

这一层要回答 README 的**两条验收标准**，而且每一条都要求「可归因到具体变量」：

1. **给出 Embedding × Index × Distance × Search 的消融矩阵**
   → :func:`ablation_rows`（单变量消融）+ :func:`build_matrix`（全量矩阵）
2. **输出 Latency–Recall Pareto 前沿，且每个结论可归因到具体变量**
   → :func:`assemble`（前沿）+ :func:`space_attribution`（归因）

坐标约定（踩过坑，写清楚）
--------------------------
**只保留一套坐标约定：``min`` 形式**（两个方向都转成「越小越好」）。
同一份代码里并存两套约定是 DataMixture 项目真实踩过的坑
（超体积函数吃最大形式的点却内部调 min 形式的 ``dominates``，静默削掉半个前沿）。
本项目只有一套，并且 ``hypervolume_2d`` **自己重算非支配集**，不信任调用方。

归因怎么做才不是自说自话
------------------------
「可归因到具体变量」很容易变成一句空话。这里的做法是**两个互补的量**：

- **单变量消融**（:func:`ablation_rows`）：固定其余四个轴，只动一个，
  读出 Δrecall 与 Δ成本。这是**受控实验**，因果性最强。
- **空间归因**（:func:`space_attribution`）：在「只差这一个轴」的配置组内，
  算 recall 的极差（max − min）。它回答「这个轴最多能移动多少召回率」，
  于是五个轴可以**按影响力排序** —— 这才是「归因」的可操作形式。

两者缺一不可：消融给的是「参考点附近的局部斜率」，空间归因给的是
「整个轴上的最大摆幅」。只报前者，会漏掉「这个轴在别处影响巨大」的情况；
只报后者，会说不清「从当前配置出发该先动哪个」。
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable, Sequence

from .metrics import (
    METRIC_DIRECTIONS,
    TIMING_METRICS,
    has_timing_axis,
    is_deterministic_axis,
    point_of,
)
from .schema import DESIGN_AXES, RetrievalResult, RetrievalRun


def frontier_key(pair: Sequence[str]) -> str:
    """目标对的稳定键名（也是 ``result.frontiers`` 的键）。"""
    return "+".join(pair)


def frontier_pairs(cfg: dict[str, Any]) -> list[list[str]]:
    """主目标对 + 额外目标对。主目标对排在第一个。"""
    main = list(cfg["pareto"]["objectives"])
    extras = [list(p) for p in cfg["pareto"].get("extra_objectives", [])]
    out = [main]
    for pair in extras:
        if pair not in out:
            out.append(pair)
    return out


# ----------------------------------------------------------------------
# 支配
# ----------------------------------------------------------------------
def to_min_form(point: dict[str, float], objectives: Sequence[str]) -> tuple[float, ...]:
    """按指标方向把点转成「越小越好」的形式。"""
    return tuple(
        point[o] if METRIC_DIRECTIONS[o] == "min" else -point[o] for o in objectives
    )


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """``a`` 是否支配 ``b``（全不差，且至少一维更好）。

    注意「支配」不要求某一维**严格**更好：只要不全差且至少一维更好即可。
    """
    not_worse = all(x <= y for x, y in zip(a, b))
    strictly_better = any(x < y for x, y in zip(a, b))
    return not_worse and strictly_better


def non_dominated(
    ids: Sequence[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> list[str]:
    """非支配集。返回顺序与输入一致，便于稳定输出。"""
    values = {i: to_min_form(points[i], objectives) for i in ids}
    out: list[str] = []
    for i in ids:
        vi = values[i]
        if any(j != i and dominates(values[j], vi) for j in ids):
            continue
        out.append(i)
    return out


def dominators(
    target: str,
    ids: Sequence[str],
    points: dict[str, dict[str, float]],
    objectives: Sequence[str],
) -> list[str]:
    """谁支配了 ``target``。报告里用来解释「基线为什么没进前沿」。"""
    vt = to_min_form(points[target], objectives)
    return [
        i
        for i in ids
        if i != target and dominates(to_min_form(points[i], objectives), vt)
    ]


def dominated_count(
    target: str,
    ids: Sequence[str],
    points: dict[str, dict[str, float]],
    objectives: Sequence[str],
) -> int:
    """``target`` 支配了多少个 spec。

    前沿上每个点都值得给出这个数：它回答「这个取舍轴上的位置，
    在真实的设计空间里排掉了多少个别的配置」。
    """
    vt = to_min_form(points[target], objectives)
    return sum(
        1
        for i in ids
        if i != target and dominates(vt, to_min_form(points[i], objectives))
    )


def pareto_representatives(
    ids: Sequence[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> tuple[list[str], list[str]]:
    """非支配集，并**按目标向量去重**。

    为什么需要去重：设计空间里不同配置可能给出**逐位相同**的目标向量
    （实测 ``hnsw`` 的 ``ef_search=16`` 与 ``64`` 在 ``l2`` 上结果完全一致 ——
    因为 l2 的图是退化的，两个档位都只扫 21 个向量）。此时两者互不支配
    （支配要求至少一维**严格**更好），于是会一起留在前沿上，
    读者会以为那是 bug。

    去重保留 spec_id 字典序最小的那个当代表（确定，不依赖输入顺序），
    返回 ``(代表集, 被去掉的重复项)`` —— 被去掉的项要进 counters，
    因为「这两个档位完全等价」本身就是一个有信息量的结果。
    """
    front = non_dominated(ids, points, objectives)
    seen: dict[tuple[float, ...], str] = {}
    reps: list[str] = []
    dropped: list[str] = []
    # 按 spec_id 排序后再去重。``non_dominated`` 返回的是**输入顺序**，
    # 而输入顺序取决于调用方 —— 直接用它的话「保留字典序最小的代表」
    # 这句话就不成立（同一批点换个传入顺序会留下不同的代表）。
    for sid in sorted(front):
        key = to_min_form(points[sid], objectives)
        if key in seen:
            dropped.append(sid)
            continue
        seen[key] = sid
        reps.append(sid)
    return reps, dropped


def hv_contributions(
    ids: Sequence[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> dict[str, float]:
    """每个前沿点对超体积的**增量**贡献（阶梯面积）。

    前沿点的增量之和恒等于总超体积 —— 这条恒等式有测试守着。
    贡献接近 0 的点说明它只是勉强没被支配。
    """
    ref = reference_point(points.keys(), points, objectives)
    front = non_dominated(ids, points, objectives)
    ordered = sorted((to_min_form(points[i], objectives), i) for i in front)
    out: dict[str, float] = {}
    prev_y = ref[1]
    for value, i in ordered:
        x, y = value
        if y >= prev_y:
            out[i] = 0.0
            continue
        out[i] = (ref[0] - x) * (prev_y - y)
        prev_y = y
    return out


# ----------------------------------------------------------------------
# 超体积（二维，min 形式）
# ----------------------------------------------------------------------
#: 参考点外扩比例。参考点取全部候选点的最差值再外扩 —— 不外扩的话，
#: 「恰好在成本上限」那个点的箱体宽度为 0、贡献 0 面积。
REFERENCE_MARGIN = 0.05


def reference_point(
    ids: Iterable[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> tuple[float, float]:
    """参考点 = 全部候选点在两个方向上的最差值，再外扩 ``REFERENCE_MARGIN``。

    取「全部候选点」而不是「前沿点」：这样超体积可以在
    「同一候选集的不同前沿子集」之间比较。
    """
    mins = [to_min_form(points[i], objectives) for i in ids]
    if not mins:
        return (1.0, 1.0)
    return tuple(  # type: ignore[return-value]
        max(v[k] for v in mins) * (1.0 + REFERENCE_MARGIN) for k in range(2)
    )


def hypervolume_2d(
    frontier: Iterable[tuple[float, ...]], reference: tuple[float, float]
) -> float:
    """二维 min 形式的超体积：``∪ [x_i, R0] × [y_i, R1]`` 的面积。

    自己先重算一遍非支配集再走阶梯求和 —— 调用方传进来的点若含有被支配点，
    阶梯会算错（多算面积）而**不会报错**。多算一遍很便宜，静默错一次很贵。
    """
    pts = list(frontier)
    keep = [p for p in pts if not any(dominates(q, p) for q in pts if q != p)]
    if not keep:
        return 0.0
    ordered = sorted(keep, key=lambda p: (p[0], p[1]))
    hv = 0.0
    prev_y = reference[1]
    for x, y in ordered:
        if y >= prev_y:
            continue
        hv += (reference[0] - x) * (prev_y - y)
        prev_y = y
    return hv


def hypervolume_of(
    ids: Sequence[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> float:
    """一组点的超体积（先筛非支配，再用全部候选点定参考点）。"""
    ref = reference_point(points.keys(), points, objectives)
    front = non_dominated(ids, points, objectives)
    return hypervolume_2d([to_min_form(points[i], objectives) for i in front], ref)


# ----------------------------------------------------------------------
# 拐点
# ----------------------------------------------------------------------
def knee_point(
    ids: Sequence[str], points: dict[str, dict[str, float]], objectives: Sequence[str]
) -> str | None:
    """前沿的「拐点」：离首尾连线最远的那个点。

    每个轴先按方向归一化到 ``[0, 1]``（越大越好）再算距离。

    归一化的作用要说准（TokenizerFactory 项目的 docstring 曾被测试纠正）：
    逐轴仿射缩放**不会**改变「点到直线距离」的排序 —— 归一化前后两个距离
    只差一个常数因子。真正依赖归一化的是 ``start`` / ``end`` 的选取：
    它们按 ``(x, y)`` 做**字典序**比较，用原始量时量级大的那个轴会单独决定选谁
    （本项目里「距离计算次数」是几十到几百，「召回率」是 0~1，
    不归一化的话拐点永远由成本轴决定）。
    """
    ordered = sorted(ids)
    if not ordered:
        return None
    if len(ordered) <= 2:
        return ordered[0]

    def axis(name: str) -> dict[str, float]:
        values = [points[i][name] for i in ordered]
        lo, hi = min(values), max(values)
        span = hi - lo
        out: dict[str, float] = {}
        for i in ordered:
            if span == 0:
                out[i] = 1.0
            elif METRIC_DIRECTIONS[name] == "min":
                out[i] = 1.0 - (points[i][name] - lo) / span
            else:
                out[i] = (points[i][name] - lo) / span
        return out

    o0, o1 = objectives
    x, y = axis(o0), axis(o1)
    start = max(ordered, key=lambda i: (x[i], y[i]))
    end = max(ordered, key=lambda i: (y[i], x[i]))
    dx, dy = x[end] - x[start], y[end] - y[start]
    length = math.hypot(dx, dy)
    if length == 0:
        return ordered[0]

    best: str | None = None
    best_distance = -1.0
    for i in ordered:
        distance = abs(dy * (x[i] - x[start]) - dx * (y[i] - y[start])) / length
        if distance > best_distance:
            best_distance, best = distance, i
    return best


# ----------------------------------------------------------------------
# 选型
# ----------------------------------------------------------------------
def select(result: RetrievalResult, cfg: dict[str, Any]) -> tuple[str, str]:
    """返回 ``(spec_id, 理由)``。

    理由必须一并返回：选型规则是**一个价值判断**，不是从数据里推出来的结论。
    """
    if not result.runs:
        return "", "没有可选的 run"

    rule = cfg["select"]["rule"]
    if rule == "knee":
        key = frontier_key(cfg["pareto"]["objectives"])
        ids = result.frontiers.get(key, [])
        chosen = knee_point(
            ids, {r.spec_id: point_of(r) for r in result.runs}, cfg["pareto"]["objectives"]
        )
        if chosen:
            return chosen, f"knee：主前沿 {key} 上离首尾连线最远的点"
        return "", "主前沿为空，knee 规则无法选型"

    metric = str(cfg["select"]["budget_metric"])
    budget = float(cfg["select"]["budget"])
    within = [r for r in result.runs if r.metrics.get(metric) <= budget]
    if not within:
        within = list(result.runs)
        reason = (
            f"max_recall_within_budget：没有任何 spec 的 {metric} ≤ {budget:g}，"
            f"退化为在全部 {len(within)} 个 spec 里选召回最好的"
        )
    else:
        reason = (
            f"max_recall_within_budget：在 {metric} ≤ {budget:g} 的 "
            f"{len(within)} 个 spec 里选 recall_at_k 最大的"
        )
    chosen = max(
        within,
        key=lambda r: (
            r.metrics.recall_at_k,
            -r.metrics.get(metric),
            r.spec_id,
        ),
    )
    return chosen.spec_id, reason


# ----------------------------------------------------------------------
# 矩阵
# ----------------------------------------------------------------------
def build_matrix(result: RetrievalResult, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """消融矩阵：每个 spec 一行，含**五个设计轴** + 全部度量。

    行顺序 = 设计空间的展开顺序（embedding → chunking → index → distance → search）。
    这不是随手排的：它让「同一批向量换索引」的几行天然相邻，
    读者顺着往下看就是一次受控实验。
    """
    rows: list[dict[str, Any]] = []
    for run in result.runs:
        rows.append(
            {
                "spec_id": run.spec_id,
                "embedding": run.spec.embedding,
                "chunking": run.spec.chunking,
                "index": run.spec.index,
                "distance": run.spec.distance,
                "search": run.spec.search_tag,
                "space": "representation"
                if run.spec_id == str(cfg["ablation"]["reference"])
                else "",
                "frontier": [],
                **run.metrics.to_dict(),
                **{f"meta_{k}": v for k, v in sorted(run.meta.items())},
            }
        )
    return rows


def assemble(result: RetrievalResult, cfg: dict[str, Any]) -> None:
    """就地填好 ``matrix`` / ``frontiers`` / ``ablations`` / ``selected`` / ``counters``。"""
    points = {r.spec_id: point_of(r) for r in result.runs}
    ids = sorted(points)

    frontiers: dict[str, list[str]] = {}
    dropped: dict[str, list[str]] = {}
    for pair in frontier_pairs(cfg):
        key = frontier_key(pair)
        reps, gone = pareto_representatives(ids, points, pair)
        frontiers[key] = reps
        if gone:
            dropped[key] = gone
    result.frontiers = frontiers

    matrix = build_matrix(result, cfg)
    # 矩阵的 ``frontier`` 列**只列可复现的前沿**。含实测轴的副前沿会在两次运行
    # 之间换成员，把它写进矩阵会让 ``matrix.csv`` 整体失去可复现性 ——
    # 而那正是这张表存在的意义（验收标准①要的是一份能引用的消融矩阵）。
    # 副前沿的完整成员在 ``frontier.csv`` 与报告第七节里，一个都没丢。
    deterministic_keys = {
        frontier_key(pair) for pair in frontier_pairs(cfg) if is_deterministic_axis(pair)
    }
    for row in matrix:
        row["frontier"] = sorted(
            key
            for key, members in frontiers.items()
            if key in deterministic_keys and row["spec_id"] in members
        )
    result.matrix = matrix

    result.ablations = {
        "single_axis": ablation_rows(result, cfg),
        "attribution": space_attribution(result, cfg),
        "representation_ceiling": representation_ceiling(result, cfg),
        "reach_cost": reach_cost(result, cfg),
        "budget_sensitivity": budget_sensitivity(result, cfg),
    }

    selected, reason = select(result, cfg)
    result.selected = selected

    main_pair = list(cfg["pareto"]["objectives"])
    result.counters = {
        "n_specs": len(result.runs),
        "n_axes": len(DESIGN_AXES),
        "axis_values": {a: result.axis_values(a) for a in DESIGN_AXES},
        "selected_reason": reason,
        "frontier_sizes": {k: len(v) for k, v in sorted(frontiers.items())},
        # 目标向量逐位相同、被去重掉的配置。它们与代表项在指标上完全等价 ——
        # 「某个搜索档位完全无效」就是这么被发现的，所以必须留下记录。
        "frontier_duplicates": {k: v for k, v in sorted(dropped.items())},
        "hypervolume": {
            key: round(hypervolume_of(frontiers[key], points, pair), 8)
            for key, pair in ((frontier_key(p), p) for p in frontier_pairs(cfg))
        },
        "reference_point": {
            frontier_key(p): [round(v, 6) for v in reference_point(ids, points, p)]
            for p in frontier_pairs(cfg)
        },
        "deterministic_frontiers": sorted(
            key for key, pair in ((frontier_key(p), p) for p in frontier_pairs(cfg))
            if not has_timing_axis(pair)
        ),
        "timing_frontiers": sorted(
            key for key, pair in ((frontier_key(p), p) for p in frontier_pairs(cfg))
            if has_timing_axis(pair)
        ),
        "best_by": {
            name: {
                "spec_id": run.spec_id,
                "value": round(run.metrics.get(name), 6),
            }
            for name in (
                "recall_at_k",
                "ndcg_at_k",
                "mrr",
                "n_distance_computations",
                "query_latency_ms",
                "index_bytes",
            )
            if (run := result.best_by(name, METRIC_DIRECTIONS[name])) is not None
        },
        "baseline": _baseline(result, cfg, main_pair),
    }


# ----------------------------------------------------------------------
# 单变量消融
# ----------------------------------------------------------------------
def _spec_varying(ref: RetrievalRun, axis: str, value: str) -> RetrievalSpec:
    """构造「参考配置只改 ``axis`` 为 ``value``」的 spec。"""
    from .schema import RetrievalSpec

    kwargs: dict[str, Any] = {
        "embedding": ref.spec.embedding,
        "chunking": ref.spec.chunking,
        "index": ref.spec.index,
        "distance": ref.spec.distance,
        "search": dict(ref.spec.search),
    }
    kwargs[axis] = value
    return RetrievalSpec(**kwargs)


def _index_spec(
    ref: RetrievalRun, index: str, level: dict[str, Any] | None
) -> RetrievalSpec:
    """参考配置的其余轴不动，只把索引换成 ``index`` 并给它一个搜索档位。"""
    from .schema import RetrievalSpec

    return RetrievalSpec(
        embedding=ref.spec.embedding,
        chunking=ref.spec.chunking,
        index=index,
        distance=ref.spec.distance,
        search={str(k): v for k, v in (level or {}).items()},
    )


def ablation_groups(result: RetrievalResult, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """每个设计轴一组「**只差这一个轴**」的配置，外加组内基准。

    这是本项目最重要的方法论的可执行形式（见 skill 的
    「扇出-收拢型的额外约定」）：一次动两个变量，Δ 就无法归因。

    **``index`` 与 ``search`` 两个轴需要特别处理**（第一版在这里静默少算）：

    - 参考配置是 ``flat``，而 flat **没有搜索参数**。于是
      「只把 index 从 flat 换成 ivf」造出来的 spec_id 是 ``…-ivf-cos``（缺 nprobe），
      它根本不存在，那一行就被静默跳过了 —— 消融表看起来正常，实际少了两个轴。
      修法：``index`` 轴给每个索引配**它的第一档**搜索参数，并把这个口径写进 ``note``。
    - ``search`` 轴在 flat 上无从消融（只有一档）。修法：挑**档位最多**的那个索引
      当这个轴的实验台（其余轴仍固定为参考值），同样写进 ``note``。

    两个修法都保证「一次只动一个变量」这条纪律不被破坏：换索引时搜索档位是
    该索引的**最低档**（一个确定的选择，不是最优选择），换搜索档位时索引固定。
    """
    ref = result.reference_run()
    if ref is None:
        return []
    by_spec = {r.spec_id: r for r in result.runs}
    groups: list[dict[str, Any]] = []

    for axis in DESIGN_AXES:
        note = ""
        if axis == "index":
            members = [
                _index_spec(ref, index, cfg["retrieval"]["search_params"][index][0]).spec_id
                for index in cfg["retrieval"]["indexes"]
            ]
            note = "每个索引取它的第一档搜索参数（否则 ivf/hnsw 的 spec 不存在）"
        elif axis == "search":
            # 挑档位最多的索引当实验台；并列时取名字最小的，保证确定
            candidates = sorted(
                cfg["retrieval"]["indexes"],
                key=lambda i: (-len(cfg["retrieval"]["search_params"][i]), i),
            )
            host = candidates[0]
            levels = cfg["retrieval"]["search_params"][host]
            members = [
                _index_spec(ref, host, level).spec_id for level in levels
            ]
            note = (
                f"在 {host} 上扫（参考索引 {ref.spec.index} 只有 "
                f"{len(cfg['retrieval']['search_params'][ref.spec.index])} 档，无从消融）"
                if len(cfg["retrieval"]["search_params"][ref.spec.index]) < 2
                else f"在 {host} 上扫"
            )
        else:
            members = [
                _spec_varying(ref, axis, value).spec_id
                for value in result.axis_values(axis)
            ]

        members = [m for m in members if m in by_spec]
        if len(members) < 2:
            continue
        baseline = ref.spec_id if ref.spec_id in members else members[0]
        groups.append(
            {
                "axis": axis,
                "baseline": baseline,
                "members": members,
                "note": note,
                "space": "representation"
                if axis in ("embedding", "chunking")
                else "retrieval",
            }
        )
    return groups


def ablation_rows(result: RetrievalResult, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """单变量消融表：每个轴、每个非基准取值一行，Δ 相对**该轴的组内基准**。

    组内基准通常是参考配置本身（``flat`` + ``cos``）。因为 flat 是精确检索，
    基准行的召回率就是该表示空间的上限，所以这张表读起来是
    「从上限往下掉多少」，而不是「从一个任意起点往下掉多少」。
    """
    by_spec = {r.spec_id: r for r in result.runs}
    rows: list[dict[str, Any]] = []
    for group in ablation_groups(result, cfg):
        axis = group["axis"]
        baseline_run = by_spec.get(group["baseline"])
        if baseline_run is None:
            continue
        base = point_of(baseline_run)
        for sid in group["members"]:
            if sid == group["baseline"]:
                continue
            run = by_spec[sid]
            p = point_of(run)
            rows.append(
                {
                    "axis": axis,
                    "space": group["space"],
                    "baseline": group["baseline"],
                    "from": baseline_run.spec.axis(axis),
                    "to": run.spec.axis(axis),
                    "spec_id": sid,
                    "recall_at_k": round(p["recall_at_k"], 6),
                    "d_recall": round(p["recall_at_k"] - base["recall_at_k"], 6),
                    "ndcg_at_k": round(p["ndcg_at_k"], 6),
                    "d_ndcg": round(p["ndcg_at_k"] - base["ndcg_at_k"], 6),
                    "n_distance_computations": round(p["n_distance_computations"], 3),
                    "d_cost": round(
                        p["n_distance_computations"] - base["n_distance_computations"], 3
                    ),
                    "query_latency_ms": round(p["query_latency_ms"], 4),
                    "d_latency_ms": round(
                        p["query_latency_ms"] - base["query_latency_ms"], 4
                    ),
                    "note": group["note"],
                }
            )
    return rows


# ----------------------------------------------------------------------
# 空间归因
# ----------------------------------------------------------------------
def space_attribution(
    result: RetrievalResult, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """每个设计轴「最多能移动多少召回率」—— 归因的可操作形式。

    口径：对每个轴，取「只差这一个轴」的配置组（见 :func:`ablation_groups`），
    算该组内 Recall@k 的极差（max − min）。极差越大，说明这个轴越值得调。

    **同时报「组内比了几个配置」与「这个轴一共有几个取值」**：
    组内只有一个成员时极差恒为 0，那不是「这个轴不重要」，
    而是「从当前参考配置出发没法消融它」。两者必须分开读，
    否则读者会把「没比」当成「没影响」。
    """
    by_spec = {r.spec_id: r for r in result.runs}
    rows: list[dict[str, Any]] = []
    for group in ablation_groups(result, cfg):
        axis = group["axis"]
        members = [by_spec[m] for m in group["members"] if m in by_spec]
        if not members:
            continue
        recalls = [r.metrics.recall_at_k for r in members]
        costs = [r.metrics.n_distance_computations for r in members]
        best = max(members, key=lambda r: (r.metrics.recall_at_k, r.spec_id))
        worst = min(members, key=lambda r: (r.metrics.recall_at_k, r.spec_id))
        rows.append(
            {
                "axis": axis,
                "space": group["space"],
                "n_values": len(result.axis_values(axis)),
                "n_compared": len(members),
                "baseline": group["baseline"],
                "recall_min": round(min(recalls), 6),
                "recall_max": round(max(recalls), 6),
                "spread": round(max(recalls) - min(recalls), 6),
                "cost_ratio": round(
                    (max(costs) / min(costs)) if min(costs) > 0 else 0.0, 3
                ),
                "best_value": best.spec.axis(axis),
                "worst_value": worst.spec.axis(axis),
                "note": group["note"],
            }
        )
    rows.sort(key=lambda r: (-r["spread"], r["axis"]))
    return rows


def representation_ceiling(
    result: RetrievalResult, cfg: dict[str, Any]
) -> dict[str, Any]:
    """表示空间的上限：**精确检索**下每个 (embedding, chunking) 的召回率。

    只用 ``flat`` + 参考距离的 spec 来算。理由：近似索引的召回率里混着
    「索引掉了多少」，而这里要问的是「这批向量本身能把答案排多前」。
    flat 是精确的，所以它就是该表示空间在 top-k 下的**上限**。

    这张表是「表示空间决定上限」这个命题的直接证据：
    如果某个 embedding 在 flat 下就只有 0.5，那么任何索引都救不回来。
    """
    ref = result.reference_run()
    if ref is None:
        return {}
    flat_specs = [
        r
        for r in result.runs
        if r.spec.index == "flat" and r.spec.distance == ref.spec.distance
    ]
    if not flat_specs:
        return {}
    best = max(flat_specs, key=lambda r: (r.metrics.recall_at_k, r.spec_id))
    rows: list[dict[str, Any]] = []
    for r in sorted(flat_specs, key=lambda r: (r.spec.embedding, r.spec.chunking)):
        rows.append(
            {
                "embedding": r.spec.embedding,
                "chunking": r.spec.chunking,
                "spec_id": r.spec_id,
                "recall_at_k": round(r.metrics.recall_at_k, 6),
                "ndcg_at_k": round(r.metrics.ndcg_at_k, 6),
                "mrr": round(r.metrics.mrr, 6),
                "n_queries_zero_recall": r.metrics.n_queries_zero_recall,
                "gap_to_ceiling": round(
                    r.metrics.recall_at_k - best.metrics.recall_at_k, 6
                ),
                "is_ceiling": r.spec_id == best.spec_id,
            }
        )
    return {
        "ceiling_spec": best.spec_id,
        "ceiling_recall": round(best.metrics.recall_at_k, 6),
        "rows": rows,
    }


def reach_cost(result: RetrievalResult, cfg: dict[str, Any]) -> dict[str, Any]:
    """达到「最优召回率的 X%」需要的成本 —— 前沿的可决策版本。

    对每个阈值，在设计空间里找**成本最低**的那个 spec。这张表把
    「Latency–Recall 前沿」翻译成一句人话：**要达到 95% 的最好效果，
    最少要扫多少个向量 / 花多少毫秒**。

    ``best_recall`` 取全部 spec 的最大召回率（即 ``flat`` + 最好 embedding
    的召回率）—— 这是该语料上的可达上限。

    并列时的取舍：**优先挑 ``flat`` 的那个配置**。因为上限这个概念的定义就是
    「精确检索下能达到多少」，报告第六节的 ``representation_ceiling`` 也是这么
    算的 —— 两处若给出不同的 ``ceiling_spec``，读者会以为哪里算错了。
    （实测确实会并列：``tfidf-sentence-ivf-cos-np3`` 用 61.7 次距离就拿到了
    与 ``flat`` 相同的 0.8333，此时该报的是「上限是 flat 的 0.8333」，
    而不是把那个更便宜的近似配置当成上限。）
    """
    if not result.runs:
        return {}
    best = min(
        result.runs,
        key=lambda r: (
            -r.metrics.recall_at_k,
            0 if r.spec.index == "flat" else 1,
            r.spec_id,
        ),
    )
    ceiling = best.metrics.recall_at_k
    thresholds = sorted(float(t) for t in cfg["ablation"]["reach_thresholds"])
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        need = t * ceiling
        eligible = [
            r
            for r in result.runs
            if r.metrics.recall_at_k >= need - 1e-12
        ]
        if not eligible:
            rows.append(
                {
                    "threshold": t,
                    "target_recall": round(need, 6),
                    "spec_id": "",
                    "n_distance_computations": None,
                    "query_latency_ms": None,
                    "recall_at_k": None,
                    "cost_vs_exact": None,
                    "reachable": False,
                }
            )
            continue
        cheapest = min(
            eligible,
            key=lambda r: (r.metrics.n_distance_computations, r.metrics.recall_at_k, r.spec_id),
        )
        exact = min(
            (r for r in result.runs if r.spec.index == "flat"),
            key=lambda r: (r.metrics.n_distance_computations, r.spec_id),
            default=None,
        )
        ratio = (
            cheapest.metrics.n_distance_computations / exact.metrics.n_distance_computations
            if exact and exact.metrics.n_distance_computations > 0
            else None
        )
        rows.append(
            {
                "threshold": t,
                "target_recall": round(need, 6),
                "spec_id": cheapest.spec_id,
                "index": cheapest.spec.index,
                "distance": cheapest.spec.distance,
                "n_distance_computations": round(
                    cheapest.metrics.n_distance_computations, 3
                ),
                "query_latency_ms": round(cheapest.metrics.query_latency_ms, 4),
                "recall_at_k": round(cheapest.metrics.recall_at_k, 6),
                "cost_vs_exact": round(ratio, 4) if ratio is not None else None,
                "reachable": True,
            }
        )
    return {
        "ceiling_spec": best.spec_id,
        "ceiling_recall": round(ceiling, 6),
        "rows": rows,
    }


def budget_sensitivity(
    result: RetrievalResult, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """成本预算敏感性：预算越小，能拿到的最好召回率掉多少。

    与 TokenizerFactory 的预算敏感性表同构，但**拆成两列差距**
    （那边踩过「把两条决策轴合成一列」的坑）：

    - **该档最优 vs 全局最优**：预算约束本身造成的损失（含所有轴）；
    - **该档内表示空间差距**：固定预算下，换 embedding 能拿回多少。

    两列必须分开：只报第一列会把「预算不够」与「向量不好」混成一个数，
    而这两种情况的处置完全不同（一个加预算，一个换 embedding）。
    """
    if not result.runs:
        return []
    ceiling = max(r.metrics.recall_at_k for r in result.runs)
    budgets = sorted({round(r.metrics.n_distance_computations, 3) for r in result.runs})
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        within = [
            r for r in result.runs if r.metrics.n_distance_computations <= budget + 1e-9
        ]
        if not within:
            continue
        winner = max(within, key=lambda r: (r.metrics.recall_at_k, -r.metrics.n_distance_computations, r.spec_id))
        # 该档内按 embedding 分组，看「换 embedding」能拉开多少
        by_emb: dict[str, float] = {}
        for r in within:
            by_emb[r.spec.embedding] = max(
                by_emb.get(r.spec.embedding, float("-inf")), r.metrics.recall_at_k
            )
        emb_spread = (
            (max(by_emb.values()) - min(by_emb.values())) if len(by_emb) > 1 else 0.0
        )
        rows.append(
            {
                "budget": budget,
                "n_within": len(within),
                "best_spec": winner.spec_id,
                "best_recall": round(winner.metrics.recall_at_k, 6),
                "gap_to_ceiling": round(winner.metrics.recall_at_k - ceiling, 6),
                "gap_pct": round(
                    100.0 * (winner.metrics.recall_at_k / ceiling - 1.0), 2
                )
                if ceiling
                else 0.0,
                "embedding_spread": round(emb_spread, 6),
                "best_embedding": max(sorted(by_emb), key=lambda k: by_emb[k])
                if by_emb
                else "",
            }
        )
    return rows


# ----------------------------------------------------------------------
# 基线
# ----------------------------------------------------------------------
def _baseline(
    result: RetrievalResult, cfg: dict[str, Any], objectives: Sequence[str]
) -> dict[str, Any]:
    """基线 = **不比较就直接上的那个配置**。

    具体就是参考配置本身（``flat`` + ``cos`` + 最小 embedding/chunking）：
    它是「人会怎么写第一版」的答案 —— 精确检索，不折腾索引。
    这不是从数据里推出来的，而是一个关于「人默认会怎么做」的假设，
    必须写出来，否则「优化省了多少」这个说法没有意义。

    真正该回答的是：**如果你不比较，你会多花多少成本。**
    所以基线报告里给的是「同一召回率下最省成本的配置 vs 基线」。
    """
    ref = result.reference_run()
    if ref is None:
        return {}
    ids = sorted(r.spec_id for r in result.runs)
    points = {r.spec_id: point_of(r) for r in result.runs}
    dom = dominators(ref.spec_id, ids, points, objectives)

    # 同召回率下最省的配置 —— 「不牺牲效果，光靠选型能省多少」
    same_or_better = [
        r
        for r in result.runs
        if r.metrics.recall_at_k >= ref.metrics.recall_at_k - 1e-12
        and r.spec_id != ref.spec_id
    ]
    payload: dict[str, Any] = {
        "spec_id": ref.spec_id,
        "why": (
            "不比较就直接上的默认选择：精确检索（flat）+ 余弦距离 + "
            f"{ref.spec.embedding} + {ref.spec.chunking} 分块"
        ),
        "values": {o: round(points[ref.spec_id][o], 6) for o in objectives},
        "recall_at_k": round(ref.metrics.recall_at_k, 6),
        "n_distance_computations": round(ref.metrics.n_distance_computations, 3),
        "query_latency_ms": round(ref.metrics.query_latency_ms, 4),
        "dominated_by": len(dom),
        "dominated_by_sample": dom[:8],
    }
    if same_or_better:
        cheapest = min(
            same_or_better,
            key=lambda r: (
                r.metrics.n_distance_computations,
                r.metrics.query_latency_ms,
                r.spec_id,
            ),
        )
        ratio = (
            cheapest.metrics.n_distance_computations
            / ref.metrics.n_distance_computations
            if ref.metrics.n_distance_computations
            else 0.0
        )
        payload["cheapest_at_same_recall"] = {
            "spec_id": cheapest.spec_id,
            "recall_at_k": round(cheapest.metrics.recall_at_k, 6),
            "n_distance_computations": round(
                cheapest.metrics.n_distance_computations, 3
            ),
            "query_latency_ms": round(cheapest.metrics.query_latency_ms, 4),
            "cost_ratio": round(ratio, 4),
            "saving_pct": round(100.0 * (1.0 - ratio), 2),
            "note": "同一召回率水平下，成本只是基线的这个比例",
        }
    return payload


# ----------------------------------------------------------------------
# 前沿导出（给 CSV / 报告用）
# ----------------------------------------------------------------------
def frontier_rows(
    result: RetrievalResult, key: str
) -> list[dict[str, Any]]:
    """前沿点的完整信息，供 CSV 与报告表格共用。"""
    pair = key.split("+")
    ids = result.frontiers.get(key, [])
    points = {r.spec_id: point_of(r) for r in result.runs}
    contributions = hv_contributions(ids, points, pair) if ids else {}
    rows: list[dict[str, Any]] = []
    for rank, sid in enumerate(ids, start=1):
        run = result.find(sid)
        if run is None:
            continue
        p = point_of(run)
        rows.append(
            {
                "rank": rank,
                "spec_id": sid,
                "embedding": run.spec.embedding,
                "chunking": run.spec.chunking,
                "index": run.spec.index,
                "distance": run.spec.distance,
                "search": run.spec.search_tag,
                "objectives": {o: round(p[o], 6) for o in pair},
                "recall_at_k": round(p["recall_at_k"], 6),
                "ndcg_at_k": round(p["ndcg_at_k"], 6),
                "mrr": round(p["mrr"], 6),
                "n_distance_computations": round(p["n_distance_computations"], 3),
                "query_latency_ms": round(p["query_latency_ms"], 4),
                "index_bytes": int(p["index_bytes"]),
                "dominated": dominated_count(sid, sorted(points), points, pair),
                "hv_contribution": round(contributions.get(sid, 0.0), 8),
                "selected": sid == result.selected,
            }
        )
    return rows


def all_objective_pairs(cfg: dict[str, Any]) -> list[list[str]]:
    """全部「成本 × 质量」目标对（报告里逐对给前沿用）。

    只组合「一个成本指标 + 一个质量指标」，不做成本×成本 ——
    成本轴之间高度相关，画出来的前沿没有取舍含义。
    """
    from .metrics import COST_METRICS, QUALITY_METRICS

    pairs = [list(p) for p in frontier_pairs(cfg)]
    for cost, quality in itertools.product(sorted(COST_METRICS), sorted(QUALITY_METRICS)):
        pair = [cost, quality]
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def timing_axis_pairs(pairs: Sequence[Sequence[str]]) -> list[list[str]]:
    """这些目标对里哪些含实测轴（报告要标注）。"""
    return [list(p) for p in pairs if any(o in TIMING_METRICS for o in p)]
