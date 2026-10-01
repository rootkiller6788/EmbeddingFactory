"""检索空间的核心：三种距离 + 三种索引结构。

本模块最重要的设计不是「三种索引」，而是**一个可复现的成本模型**。

为什么需要成本模型
------------------
本项目的验收标准要一条 **Latency–Recall 前沿**。但延迟是**实测值**：
它受 CPU 降频、调度、缓存命中的影响，同一份配置跑两次不会逐位相同
（TokenizerFactory 项目实测过这条前沿会在两次运行之间换成员）。

于是本模块让每个 ``search()`` **同时返回「算了多少次距离」**：

    距离计算次数 = f(索引结构, 搜索参数, 数据规模)   ← 完全确定，逐位可复现
    实测延迟     = f(上面的东西, 机器当时的负载)      ← 不可复现

两者的关系是「成本模型 ↔ 真实成本」：报告里主前沿用前者（可复现），
另算一条用后者的副前沿来验证**两者确实同向**。
如果实测延迟与距离计算次数不同向，那说明成本模型选错了 ——
这是一个能被证伪的断言，比单报一个延迟数字诚实得多。

三种距离（统一约定：``compute`` 返回**越小越好**）
---------------------------------------------------
| 距离 | 预处理 | compute | 说明 |
| --- | --- | --- | --- |
| ``cos`` | 向量 L2 归一化 | ``1 - dot`` | 归一化后内积 = 余弦；归一化让「长度」不再影响排序 |
| ``l2`` | 无 | 平方欧氏距离 | 与欧氏同序但省一次开方 |
| ``ip`` | 无 | ``-dot`` | 内积越大越好，取负变「越小越好」 |

**归一化属于距离，不属于嵌入**（见 embed.py 模块文档第 1 条）：
``cos`` 归一化、``l2`` / ``ip`` 不归一化，三者才有区别。

三种索引
--------
| 索引 | 结构 | 精确？ | 成本 |
| --- | --- | --- | --- |
| ``flat`` | 全量暴力扫描 | **精确（oracle）** | 每个查询扫全部向量 |
| ``ivf`` | 倒排：k-means 分簇 + 只扫 ``nprobe`` 簇 | 近似 | ``nlist`` + 被扫簇的规模 |
| ``hnsw`` | 单层 NSW 图 + ``ef_search`` 束搜索 | 近似 | 图上的距离计算次数 |

``flat`` 是**精确 oracle**：它的召回率就是「该表示空间在 top-k 下的上限」。
这条性质有测试守着（``flat`` 的结果必须与暴力内积排序逐位一致）——
它一旦不成立，后面所有「近似索引掉了多少召回」的结论全部失去参照。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from operator import mul
from typing import Any, Callable, Sequence

from .rng import make_rng


# ----------------------------------------------------------------------
# 距离
# ----------------------------------------------------------------------
#: 这两个原语是**热路径**（每个 spec 要算几万次），所以刻意用 C 层实现：
#: - ``math.dist`` 是 C 实现，比手写循环快约 **7 倍**（实测 dim=512：
#:   26µs → 4.1µs）；
#: - ``math.sumprod``（3.12+）是 C 实现的点积，比 ``sum(map(mul, a, b))``
#:   快约 1.8 倍。
#: 它们不改变任何语义，只让「扫 100 个向量 × 24 条查询 × 60 个配置」
#: 从几十秒降到几秒。
def _l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = 0.0
    for v in vec:
        norm += v * v
    norm = norm**0.5
    if norm == 0.0:
        return [0.0] * len(vec)
    return [v / norm for v in vec]


try:  # Python 3.12+
    _sumprod = math.sumprod
except AttributeError:  # pragma: no cover - 3.10 / 3.11 回退
    def _sumprod(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(map(mul, a, b))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return _sumprod(a, b)


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(a, b)


def _cosine_from_dist(a: Sequence[float], b: Sequence[float]) -> float:
    """**归一化向量**的余弦距离，用一次 ``math.dist`` 算出来。

    为什么要绕这一圈？因为成本模型要求「每次距离计算等成本」。
    若 cos 用点积、l2 用 ``math.dist``，两者单次成本会差约 2 倍
    （实测 dim=512：点积 8.0µs vs ``math.dist`` 4.1µs），
    于是「距离计算次数」这个成本轴在不同距离之间就不可比了 ——
    而那会让整条 Pareto 前沿的横轴失去意义。

    恒等式（仅对归一化向量成立）：``‖a−b‖² = 2 − 2·cos(a,b)``，
    于是 ``cos(a,b) = 1 − ‖a−b‖²/2``，而**余弦距离**是

        ``1 − cos(a,b) = ‖a−b‖² / 2``

    也就是下面这一行。第一版把它写成了 ``1 − d²/2``（那是**相似度**），
    结果召回率从 0.81 掉到 0.08 —— 距离的符号写反了，排序就整个反过来。
    这个 bug 不会报错，只会让「cos 比 l2 差」这种假结论出现，
    所以有一条测试直接断言「cos 的排序与朴素点积实现逐位一致」。

    于是 cos 与 l2 的**唯一区别**就回到「要不要归一化」这一件事上 ——
    这正好是本项目想研究的那个变量。
    """
    d = math.dist(a, b)
    return d * d / 2.0


@dataclass(frozen=True)
class Distance:
    """一个距离。``compute`` 返回**越小越好**的值。"""

    name: str
    prepare: Callable[[Sequence[float]], list[float]]
    compute: Callable[[Sequence[float], Sequence[float]], float]
    #: 这个距离是否已经归一化过（报告里解释「cos 与 l2 结果为何接近/远离」要用）
    normalizes: bool

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "normalizes": self.normalizes}


DISTANCES: dict[str, Distance] = {
    "cos": Distance(
        name="cos",
        prepare=lambda v: _l2_normalize(v),
        compute=_cosine_from_dist,
        normalizes=True,
    ),
    "l2": Distance(
        name="l2",
        prepare=lambda v: list(v),
        compute=_l2,
        normalizes=False,
    ),
    # ``ip`` 保留在注册表里（配置想用就能用），但它的单次成本与上面两个不同
    # （点积 vs math.dist），所以**不要**把 ip 和 cos/l2 放进同一条成本轴做比较。
    "ip": Distance(
        name="ip",
        prepare=lambda v: list(v),
        compute=lambda a, b: -_dot(a, b),
        normalizes=False,
    ),
}


# ----------------------------------------------------------------------
# 索引基类
# ----------------------------------------------------------------------
@dataclass
class SearchResult:
    """一次检索的产物。``n_distance_computations`` 是本项目的成本模型。"""

    ids: list[int]
    n_distance_computations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ids": list(self.ids),
            "n_distance_computations": self.n_distance_computations,
        }


class BaseIndex:
    """索引基类。

    子类必须实现 :meth:`_search`，它返回 ``(ids, n_distance_computations)``。
    ``ids`` 按距离升序；并列时按下标升序（**必须显式打破并列**，
    否则 top-k 的成员会随排序实现变化，召回率跟着抖）。
    """

    name = "base"

    def __init__(
        self,
        vectors: Sequence[Sequence[float]],
        distance: Distance,
        params: dict[str, Any],
        seed: int,
    ) -> None:
        self.distance = distance
        self.params = dict(params or {})
        self.seed = seed
        # 语料向量按距离的约定预处理一次（cos 会归一化）
        self.vectors: list[list[float]] = [distance.prepare(v) for v in vectors]
        self.n_vectors = len(self.vectors)
        self.dim = len(self.vectors[0]) if self.vectors else 0
        self._build()

    # -- 子类接口 ----------------------------------------------------
    def _build(self) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError

    def _search(self, query: list[float], k: int) -> tuple[list[int], int]:
        raise NotImplementedError  # pragma: no cover - 抽象

    def size_bytes(self) -> int:
        """索引常驻内存的估算值（README 的 RAM 维度）。

        口径：向量按 float64 计（8 字节/维），图/倒排的邻接表按 int64 计。
        它是**估算**而不是实测 RSS —— 报告里会写明这一点，
        并且它只用于「同一批向量下不同索引的相对比较」，不做绝对承诺。
        """
        base = self.n_vectors * self.dim * 8
        return base + self._extra_bytes()

    def _extra_bytes(self) -> int:
        return 0

    # -- 公共入口 ----------------------------------------------------
    def search(self, query: Sequence[float], k: int) -> SearchResult:
        if self.n_vectors == 0 or k <= 0:
            return SearchResult(ids=[], n_distance_computations=0)
        prepared = self.distance.prepare(query)
        ids, n_dc = self._search(prepared, min(k, self.n_vectors))
        return SearchResult(ids=ids, n_distance_computations=n_dc)

    def meta(self) -> dict[str, Any]:
        return {"n_vectors": self.n_vectors, "dim": self.dim}


# ----------------------------------------------------------------------
# flat：全量暴力（精确 oracle）
# ----------------------------------------------------------------------
class FlatIndex(BaseIndex):
    """暴力扫描。**精确** —— 它的召回率是「表示空间的上限」。"""

    name = "flat"

    def _build(self) -> None:
        pass

    def _search(self, query: list[float], k: int) -> tuple[list[int], int]:
        compute = self.distance.compute
        scored = [(compute(query, vec), i) for i, vec in enumerate(self.vectors)]
        scored.sort(key=lambda t: (t[0], t[1]))
        return [i for _, i in scored[:k]], len(scored)

    def meta(self) -> dict[str, Any]:
        return {**super().meta(), "exact": True, "scans_all": True}


# ----------------------------------------------------------------------
# ivf：倒排（k-means 分簇 + nprobe）
# ----------------------------------------------------------------------
class IVFIndex(BaseIndex):
    """倒排索引：先对簇心做一轮比较，再只扫最近 ``nprobe`` 个簇。

    簇心用**确定性 k-means**（k-means++ 初始化 + 固定轮数）：
    随机初始化会让同一份配置在两次运行里产生不同的分簇，
    于是召回率与距离计算次数都变 —— 而本项目要求成本模型逐位可复现。
    """

    name = "ivf"

    def _build(self) -> None:
        nlist = int(self.params.get("nlist", 6))
        iters = int(self.params.get("kmeans_iters", 12))
        self.nlist = max(1, min(nlist, self.n_vectors))
        compute = self.distance.compute

        # ---- 确定性 k-means++ 初始化 ----
        rng = make_rng(self.seed, "ivf-init", self.nlist, self.n_vectors)
        first = rng.randrange(self.n_vectors)
        centers: list[list[float]] = [list(self.vectors[first])]
        chosen = {first}
        while len(centers) < self.nlist:
            # 每个点取「到最近已有中心的距离」，按 D² 采样
            weights: list[float] = []
            for vec in self.vectors:
                best = min(compute(vec, c) for c in centers)
                weights.append(max(best, 0.0))
            total = sum(weights)
            if total <= 0.0:
                # 所有点都与某个中心重合：补一个尚未被选中的点即可
                rest = [i for i in range(self.n_vectors) if i not in chosen]
                pick = rest[0] if rest else 0
            else:
                r = rng.random() * total
                acc = 0.0
                pick = self.n_vectors - 1
                for i, w in enumerate(weights):
                    acc += w
                    if acc >= r:
                        pick = i
                        break
            centers.append(list(self.vectors[pick]))
            chosen.add(pick)

        # ---- Lloyd 迭代 ----
        assign = [0] * self.n_vectors
        for _ in range(max(1, iters)):
            changed = False
            for i, vec in enumerate(self.vectors):
                best_c, best_d = 0, float("inf")
                for c_i, center in enumerate(centers):
                    d = compute(vec, center)
                    if d < best_d:
                        best_c, best_d = c_i, d
                if assign[i] != best_c:
                    assign[i] = best_c
                    changed = True
            sums = [[0.0] * self.dim for _ in range(self.nlist)]
            counts = [0] * self.nlist
            for i, vec in enumerate(self.vectors):
                c = assign[i]
                counts[c] += 1
                s = sums[c]
                for d_i, v in enumerate(vec):
                    s[d_i] += v
            for c_i in range(self.nlist):
                if counts[c_i]:
                    centers[c_i] = [x / counts[c_i] for x in sums[c_i]]
            if not changed:
                break

        self.centers = centers
        self.cells: list[list[int]] = [[] for _ in range(self.nlist)]
        for i, c in enumerate(assign):
            self.cells[c].append(i)
        self.cell_sizes = [len(c) for c in self.cells]

    def _search(self, query: list[float], k: int) -> tuple[list[int], int]:
        nprobe = max(1, min(int(self.params.get("nprobe", 1)), self.nlist))
        compute = self.distance.compute
        # 第一轮：对簇心算距离（也算成本 —— 不扫簇心就不知道扫哪个簇）
        centroid_hits = [(compute(query, c), i) for i, c in enumerate(self.centers)]
        centroid_hits.sort(key=lambda t: (t[0], t[1]))
        n_dc = len(centroid_hits)
        # 第二轮：扫被选中的簇
        scored: list[tuple[float, int]] = []
        for _, c_i in centroid_hits[:nprobe]:
            for idx in self.cells[c_i]:
                scored.append((compute(query, self.vectors[idx]), idx))
                n_dc += 1
        scored.sort(key=lambda t: (t[0], t[1]))
        return [i for _, i in scored[:k]], n_dc

    def _extra_bytes(self) -> int:
        return (self.nlist * self.dim + self.n_vectors) * 8

    def meta(self) -> dict[str, Any]:
        sizes = sorted(self.cell_sizes, reverse=True)
        return {
            **super().meta(),
            "nlist": self.nlist,
            "nprobe": int(self.params.get("nprobe", 1)),
            "cell_sizes": sizes,
            "max_cell": sizes[0] if sizes else 0,
            "min_cell": sizes[-1] if sizes else 0,
            "empty_cells": sum(1 for s in self.cell_sizes if s == 0),
        }


# ----------------------------------------------------------------------
# hnsw：单层 NSW 图 + ef_search 束搜索
# ----------------------------------------------------------------------
class HNSWIndex(BaseIndex):
    """**简化版** HNSW：单层 NSW 图 + 束搜索。

    刻意如实命名：完整 HNSW 是多层跳表 + 启发式邻居选择，
    本项目实现的是它的**核心机制**（近邻图 + ``ef_search`` 宽度控制），
    去掉了分层。理由有两条，都是诚实的：

    1. 本语料只有一百多个向量，分层带来的「对数级跳转」收益体现不出来
       （层数会是 1），留着分层只是让代码更难懂；
    2. 本项目要研究的**不是 HNSW 本身**，而是「索引结构 × 搜索参数
       如何换取召回率」—— 单层 NSW 已经完整具备这个取舍
       （``ef_search`` 越大 → 距离计算越多 → 召回越高）。

    报告与文档里一律叫它「hnsw（简化：单层 NSW）」，不假装是完整 HNSW。
    """

    name = "hnsw"

    def _build(self) -> None:
        self.M = max(1, min(int(self.params.get("M", 6)), max(1, self.n_vectors - 1)))
        ef_construction = max(self.M, int(self.params.get("ef_construction", 24)))
        compute = self.distance.compute
        n = self.n_vectors
        # 邻接表。每插入一个节点就与已插入的近邻双向连边，并各自裁剪到 M 个最近。
        self.adj: list[list[int]] = [[] for _ in range(n)]
        self.entry = 0
        for i in range(n):
            if i == 0:
                continue
            # 建图阶段用暴力找最近邻（N 只有一百多，代价可忽略，
            # 而且这样建出来的图与插入顺序无关地「更接近真近邻图」）
            cands = sorted(
                ((compute(self.vectors[i], self.vectors[j]), j) for j in range(i)),
                key=lambda t: (t[0], t[1]),
            )[: max(self.M, ef_construction)]
            picked = [j for _, j in cands[: self.M]]
            for j in picked:
                self.adj[i].append(j)
                self.adj[j].append(i)
            # 裁剪：双向连边后可能超过 M，各留最近的 M 个
            for node in [i] + picked:
                if len(self.adj[node]) > self.M:
                    self.adj[node] = [
                        j
                        for _, j in sorted(
                            (
                                (compute(self.vectors[node], self.vectors[j]), j)
                                for j in set(self.adj[node])
                                if j != node
                            ),
                            key=lambda t: (t[0], t[1]),
                        )[: self.M]
                    ]
            # 去重（双向连边可能重复加入）
            self.adj[i] = sorted(set(self.adj[i]))
            for j in picked:
                self.adj[j] = sorted(set(self.adj[j]))
        self.degrees = [len(a) for a in self.adj]

    def _search(self, query: list[float], k: int) -> tuple[list[int], int]:
        ef = max(k, int(self.params.get("ef_search", 16)))
        ef = min(ef, self.n_vectors)
        compute = self.distance.compute
        n_dc = 0

        # 入口点
        entry_d = compute(query, self.vectors[self.entry])
        n_dc += 1
        visited = {self.entry}
        # candidates：待扩展（最小堆，按距离）
        candidates: list[tuple[float, int]] = [(entry_d, self.entry)]
        # results：当前最优 ef 个（最大堆，用负距离）
        results: list[tuple[float, int]] = [(-entry_d, self.entry)]

        while candidates:
            dist, node = heapq.heappop(candidates)
            # 束搜索的终止条件：当前最好候选已经比结果集里最差的还差
            if dist > -results[0][0] and len(results) >= ef:
                break
            for nb in self.adj[node]:
                if nb in visited:
                    continue
                visited.add(nb)
                d = compute(query, self.vectors[nb])
                n_dc += 1
                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(results, (-d, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
        # 结果集按距离升序（并列按下标升序 —— 必须显式打破并列）
        ordered = sorted(((-nd, i) for nd, i in results), key=lambda t: (t[0], t[1]))
        return [i for _, i in ordered[:k]], n_dc

    def _extra_bytes(self) -> int:
        return 2 * sum(self.degrees) * 8

    def meta(self) -> dict[str, Any]:
        deg = sorted(self.degrees, reverse=True)
        return {
            **super().meta(),
            "M": self.M,
            "ef_search": int(self.params.get("ef_search", 16)),
            "max_degree": deg[0] if deg else 0,
            "min_degree": deg[-1] if deg else 0,
            "mean_degree": round(sum(deg) / len(deg), 3) if deg else 0.0,
            "isolated": sum(1 for d in deg if d == 0),
        }


# ----------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------
class _IndexRegistry:
    def __init__(self, cls: type[BaseIndex]) -> None:
        self.cls = cls
        self.name = cls.name

    def validate_search(self, levels: Sequence[dict[str, Any]]) -> None:
        """校验搜索参数档位。**每一档都要能真的建出索引**。"""
        for i, level in enumerate(levels):
            params = dict(level or {})
            if self.name == "flat":
                if params:
                    raise ValueError(
                        f"retrieval.search_params.flat[{i}] 必须为空 —— "
                        "flat 是精确检索，没有可调的搜索参数"
                    )
                continue
            if self.name == "ivf":
                if "nprobe" not in params:
                    raise ValueError(
                        f"retrieval.search_params.ivf[{i}] 缺少 nprobe"
                    )
                if int(params["nprobe"]) < 1:
                    raise ValueError(f"ivf[{i}].nprobe 至少为 1")
            if self.name == "hnsw":
                if "ef_search" not in params:
                    raise ValueError(
                        f"retrieval.search_params.hnsw[{i}] 缺少 ef_search"
                    )
                if int(params["ef_search"]) < 1:
                    raise ValueError(f"hnsw[{i}].ef_search 至少为 1")

    def validate_params(self, params: dict[str, Any]) -> None:
        p = params or {}
        if self.name == "ivf":
            if int(p.get("nlist", 6)) < 1:
                raise ValueError("ivf.nlist 至少为 1")
            if int(p.get("kmeans_iters", 12)) < 1:
                raise ValueError("ivf.kmeans_iters 至少为 1")
        if self.name == "hnsw":
            if int(p.get("M", 6)) < 1:
                raise ValueError("hnsw.M 至少为 1")
            if int(p.get("ef_construction", 24)) < 1:
                raise ValueError("hnsw.ef_construction 至少为 1")

    def build(
        self,
        vectors: Sequence[Sequence[float]],
        distance: Distance,
        params: dict[str, Any],
        seed: int,
    ) -> BaseIndex:
        return self.cls(vectors, distance, params, seed)


INDEXES: dict[str, _IndexRegistry] = {
    "flat": _IndexRegistry(FlatIndex),
    "ivf": _IndexRegistry(IVFIndex),
    "hnsw": _IndexRegistry(HNSWIndex),
}


def build_index(
    name: str,
    vectors: Sequence[Sequence[float]],
    distance_name: str,
    params: dict[str, Any],
    seed: int,
) -> BaseIndex:
    """建一个索引。``params`` 里应当同时含索引参数（nlist / M）与搜索参数
    （nprobe / ef_search）—— 两者都在建索引时定下来，检索时不再改。"""
    if name not in INDEXES:
        raise KeyError(f"未知索引类型 {name!r}；可用：{', '.join(sorted(INDEXES))}")
    if distance_name not in DISTANCES:
        raise KeyError(
            f"未知距离 {distance_name!r}；可用：{', '.join(sorted(DISTANCES))}"
        )
    return INDEXES[name].build(vectors, DISTANCES[distance_name], params, seed)
