"""报告渲染与只读解释。

报告的结构刻意与 README 的**两条验收标准**对齐：

| 报告章节 | 对应的验收标准 |
| --- | --- |
| 一、语料画像 | 「问题是什么」—— 换个语料结论还成立吗 |
| 二、两个空间 | 「把两个变量分开」（README 第五节） |
| 三、消融矩阵 | **验收①：Embedding × Index × Distance × Search 的消融矩阵** |
| 四、单变量消融 | 验收②里的「可归因到具体变量」 |
| 五、空间归因 | 同上（哪个轴最多能移动多少召回率） |
| 六、表示空间的上限 | 「表示决定上限、检索决定成本」这个命题的直接证据 |
| 七、Pareto 前沿 | **验收②：Latency–Recall 前沿** |
| 八、达额成本 | 前沿的可决策版本（要达到 X% 效果最少花多少） |
| 九、预算敏感性 | 预算改变赢家（一次只动一个变量） |
| 十、血缘与版本 | 可复现性 |

两条必须写进报告的口径
----------------------

1. **主前沿用「距离计算次数」，含实测轴的副前沿单独标注。**
   前者逐位可复现，后者不可（见 metrics.py 模块文档）。
   报告里对每条前沿都标出它是哪一类，**不把两者混在一张表里**。
2. **「成本」与「质量」分开报。** 只报一个综合分数会让
   「换 embedding 提升的是上限、换索引提升的是效率」这件事消失 ——
   而这两件事的处置完全不同。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from . import compare
from .compare import frontier_key, frontier_pairs
from .metrics import (
    METRIC_DIRECTIONS,
    METRIC_LABELS,
    TIMING_METRICS,
    has_timing_axis,
    point_of,
)
from .schema import DESIGN_AXES, RetrievalResult, RetrievalRun

# ----------------------------------------------------------------------
# 基础表格
# ----------------------------------------------------------------------
MATRIX_HEADER: tuple[str, ...] = (
    "spec_id",
    "embedding",
    "chunking",
    "index",
    "distance",
    "search",
    "Recall@k",
    "nDCG@k",
    "MRR",
    "距离计算次数",
    "延迟(ms)",
    "建索引(s)",
    "索引KB",
    "零召回",
    "前沿",
)


def markdown_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Markdown 表格。**用 Markdown 而不是空格对齐** ——
    中文字宽会让纯文本对齐错位（这是本工作区踩过的坑）。"""
    head = [str(h) for h in header]
    body = [[("" if c is None else str(c)) for c in row] for row in rows]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def matrix_rows(result: RetrievalResult) -> list[list[str]]:
    """消融矩阵的行（字符串化，供表格与 CLI 共用）。"""
    rows: list[list[str]] = []
    for row in result.matrix:
        frontier = row.get("frontier") or []
        rows.append(
            [
                str(row["spec_id"]),
                str(row["embedding"]),
                str(row["chunking"]),
                str(row["index"]),
                str(row["distance"]),
                str(row["search"]),
                f"{float(row['recall_at_k']):.4f}",
                f"{float(row['ndcg_at_k']):.4f}",
                f"{float(row['mrr']):.4f}",
                f"{float(row['n_distance_computations']):.1f}",
                f"{float(row['query_latency_ms']):.3f}",
                f"{float(row['build_seconds']):.4f}",
                f"{int(row['index_bytes']) / 1024.0:.1f}",
                str(int(row["n_queries_zero_recall"])),
                "|".join(frontier) if frontier else "-",
            ]
        )
    return rows


# ----------------------------------------------------------------------
# 前沿
# ----------------------------------------------------------------------
def frontier_header(objectives: Sequence[str]) -> tuple[str, ...]:
    """前沿表头。

    **「实际成本/质量」两列只在目标对里没有它们时才补。**
    TokenizerFactory 项目在这里踩过坑：无条件加一列，于是当目标轴恰好叫
    同一个名字时表头出现两个同名列（而且只在真实配置上才暴露）。
    这里沿用修好的写法，并由「任意两个指标组合都不得出现重复列名」的测试守着。
    """
    labels = [METRIC_LABELS.get(o, o) for o in objectives]
    head = ["排名", "spec", "表示空间", "索引"]
    if "距离计算次数" not in labels:
        head.append("距离计算次数")
    return tuple(head + labels + ["支配数", "超体积增量", "含实测轴"])


def _frontier_shows_cost(objectives: Sequence[str]) -> bool:
    return "距离计算次数" not in [METRIC_LABELS.get(o, o) for o in objectives]


def frontier_rows(result: RetrievalResult, key: str) -> list[list[str]]:
    """前沿表的行。"""
    pair = key.split("+")
    raw = compare.frontier_rows(result, key)
    show_cost = _frontier_shows_cost(pair)
    timing = "是" if has_timing_axis(pair) else "否"
    rows: list[list[str]] = []
    for row in raw:
        mark = "★ " if row["selected"] else ""
        line = [
            str(row["rank"]),
            f"{mark}{row['spec_id']}",
            f"{row['embedding']}/{row['chunking']}",
            f"{row['index']}/{row['distance']}/{row['search']}",
        ]
        if show_cost:
            line.append(f"{row['n_distance_computations']:.1f}")
        for o in pair:
            line.append(_fmt_metric(o, row["objectives"][o]))
        line += [
            str(row["dominated"]),
            f"{row['hv_contribution']:,.4f}",
            timing,
        ]
        rows.append(line)
    return rows


#: 整数类指标（打印时不要带小数）
_INTEGER_METRICS = frozenset({"index_bytes", "n_chunks", "n_queries_zero_recall"})


def _fmt_metric(name: str, value: float) -> str:
    if name in _INTEGER_METRICS:
        return f"{int(round(value)):,}"
    if name in ("recall_at_k", "ndcg_at_k", "mrr"):
        return f"{value:.4f}"
    if name == "n_distance_computations":
        return f"{value:.1f}"
    if name == "query_latency_ms":
        return f"{value:.3f}"
    if name == "build_seconds":
        return f"{value:.4f}"
    if name == "mean_first_relevant_rank":
        return f"{value:.3f}"
    return f"{value:,.4f}"


# ----------------------------------------------------------------------
# 解释
# ----------------------------------------------------------------------
def explain(result: RetrievalResult, spec_id: str) -> str:
    """回答「这个配置从哪来、结果如何、为什么在这个位置」。"""
    run = result.find(spec_id)
    if run is None:
        known = result.known_spec_ids()
        return f"未知 spec_id：{spec_id}\n已知（前 10 个）：{', '.join(known[:10])}"

    spec = run.spec
    m = run.metrics
    lines: list[str] = []
    lines.append(f"spec: {spec_id}")
    lines.append("")
    lines.append("【它是什么】")
    lines.append(
        f"  表示空间：embedding={spec.embedding}　chunking={spec.chunking}"
    )
    lines.append(
        f"  检索空间：index={spec.index}　distance={spec.distance}　"
        f"search={spec.search_tag}"
    )
    lines.append("")
    lines.append("【指标】")
    lines.append(f"  Recall@{m.k} = {m.recall_at_k:.4f}　（任务指标）")
    lines.append(f"  nDCG@{m.k}   = {m.ndcg_at_k:.4f}　MRR = {m.mrr:.4f}")
    lines.append(
        f"  距离计算次数 = {m.n_distance_computations:.1f}/查询　"
        "（确定性成本模型）"
    )
    lines.append(
        f"  延迟 = {m.query_latency_ms:.3f} ms　建索引 = {m.build_seconds:.4f} s　"
        f"索引 = {m.index_bytes / 1024.0:.1f} KB"
    )
    lines.append(
        f"  零召回查询 = {m.n_queries_zero_recall}/{m.n_queries}　"
        f"首个相关平均名次 = {m.mean_first_relevant_rank:.2f}"
    )
    lines.append("")

    # 前沿位置
    lines.append("【它在哪条前沿上】")
    main_key = frontier_key(result.cfg["pareto"]["objectives"])
    on_any = False
    for key, members in sorted(result.frontiers.items()):
        if spec_id in members:
            on_any = True
            pair = key.split("+")
            rank = members.index(spec_id) + 1
            note = "（含实测轴，不可逐位复现）" if has_timing_axis(pair) else "（可复现）"
            lines.append(f"  {key}：第 {rank}/{len(members)} 名 {note}")
    if not on_any:
        pair = list(result.cfg["pareto"]["objectives"])
        ids = sorted(r.spec_id for r in result.runs)
        points = {r.spec_id: point_of(r) for r in result.runs}
        dom = compare.dominators(spec_id, ids, points, pair)
        lines.append(f"  不在任何前沿上。被 {len(dom)} 个配置支配，例如：{', '.join(dom[:3])}")
        lines.append(
            f"  支配它的配置在 {' / '.join(METRIC_LABELS.get(o, o) for o in pair)} "
            "两个方向上都更好或持平，且至少一个方向更好"
        )
    lines.append("")

    # 相对参考配置
    ref = result.reference_run()
    if ref is not None and ref.spec_id != spec_id:
        dr = m.recall_at_k - ref.metrics.recall_at_k
        dc = m.n_distance_computations - ref.metrics.n_distance_computations
        lines.append("【相对参考配置】")
        lines.append(f"  参考：{ref.spec_id}（精确检索，召回率即该表示空间的上限）")
        lines.append(f"  Δrecall = {dr:+.4f}　Δ距离计算次数 = {dc:+.1f}")
        lines.append(
            "  注意：Δ 里可能同时含表示空间与检索空间的变化，"
            "要单变量归因请看报告第四、五节"
        )
        lines.append("")

    # 逐查询样本
    if run.sample:
        lines.append("【逐查询样本】")
        for qid, text, top in run.sample:
            marks = " ".join(
                f"{'✓' if rel else '✗'}{doc}@{rank}" for doc, rank, rel in top
            )
            lines.append(f"  {qid} {text}")
            lines.append(f"    {marks}")
        lines.append("")

    # 索引诊断
    if run.meta:
        lines.append("【索引诊断】")
        for k in sorted(run.meta):
            v = run.meta[k]
            if isinstance(v, list):
                shown = ", ".join(str(x) for x in v[:8])
                v = f"[{shown}{'…' if len(v) > 8 else ''}]"
            lines.append(f"  {k} = {v}")
    return "\n".join(lines)


def summary_line(result: RetrievalResult) -> str:
    """一行摘要，用于 CLI 与报告开头。"""
    if not result.runs:
        return "（没有 run）"
    m = result.counters.get("best_by", {})
    best_recall = m.get("recall_at_k", {})
    cheapest = m.get("n_distance_computations", {})
    return (
        f"{len(result.runs)} 个配置 × {len(DESIGN_AXES)} 个设计轴　"
        f"最高 Recall@k = {best_recall.get('value', 0):.4f}"
        f"（{best_recall.get('spec_id', '-')}）　"
        f"最低成本 = {cheapest.get('value', 0):.1f} 次距离"
        f"（{cheapest.get('spec_id', '-')}）　"
        f"入选 = {result.selected}"
    )


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def render_markdown(result: RetrievalResult) -> str:
    cfg = result.cfg
    out: list[str] = []
    add = out.append

    add("# EmbeddingFactory 运行报告")
    add("")
    add(f"- 版本：`{result.version.version_id if result.version else '-'}`")
    add(f"- 语料：{result.corpus.name}")
    add(f"- 配置：`{cfg.get('_config_path', '-')}`")
    add("")
    add(f"> {summary_line(result)}")
    add("")

    # ---- 一、语料画像 ----------------------------------------------
    add("## 一、语料画像（问题是什么）")
    add("")
    c = result.corpus
    add(
        markdown_table(
            ("项", "值"),
            [
                ("文档数", c.n_docs),
                ("查询数", c.n_queries),
                ("主题数", c.n_topics),
                ("平均每查询相关文档数", f"{c.mean_docs_per_query:.2f}"),
                ("平均块长度（字符）", f"{c.mean_chunk_chars:.1f}"),
            ],
        )
    )
    add("")
    add("各分块策略的画像（**块数与边界切碎率**）：")
    add("")
    add(
        markdown_table(
            ("分块策略", "块数", "被硬切开的块", "切碎率", "字符数"),
            [
                (
                    name,
                    p.get("n_chunks", 0),
                    p.get("n_cut", 0),
                    f"{p.get('n_cut', 0) / p['n_chunks']:.2%}" if p.get("n_chunks") else "-",
                    p.get("n_chars", 0),
                )
                for name, p in sorted(c.chunking.items())
            ],
        )
    )
    add("")
    add(
        "**边界切碎率**是分块策略的直接质量信号：它表示有多少个块的边界落在句子中间。"
        "切碎率高意味着块与语义单元错位 —— 它不直接进指标，"
        "但它是解释「为什么这个分块策略召回率更低」的第一手证据。"
    )
    add("")

    # ---- 二、两个空间 ----------------------------------------------
    add("## 二、两个空间（把变量分开）")
    add("")
    add("README 第五节：**Performance = f(Embedding, Chunking, Index, Distance, Search)**。")
    add("本报告把它切成两组，因为两组的处置完全不同：")
    add("")
    add(
        markdown_table(
            ("空间", "轴", "本次取值", "决定什么"),
            [
                (
                    "表示空间",
                    "embedding",
                    " / ".join(result.axis_values("embedding")),
                    "**召回率的上限**",
                ),
                (
                    "表示空间",
                    "chunking",
                    " / ".join(result.axis_values("chunking")),
                    "上限（块的粒度与边界）",
                ),
                (
                    "检索空间",
                    "index",
                    " / ".join(result.axis_values("index")),
                    "**达到上限的成本**",
                ),
                (
                    "检索空间",
                    "distance",
                    " / ".join(result.axis_values("distance")),
                    "排序质量（与成本）",
                ),
                (
                    "检索空间",
                    "search",
                    " / ".join(result.axis_values("search")),
                    "成本与召回率的连续旋钮",
                ),
            ],
        )
    )
    add("")
    add(
        f"共 {len(result.runs)} 个配置（"
        + " × ".join(f"{len(result.axis_values(a))}" for a in DESIGN_AXES)
        + f" = {len(result.runs)}），k = {result.runs[0].metrics.k if result.runs else '-'}。"
    )
    add("")

    # ---- 三、消融矩阵 ----------------------------------------------
    add("## 三、消融矩阵（验收①）")
    add("")
    add(
        "每个配置一行，含五个设计轴与全部度量。**行顺序 = 设计空间的展开顺序**，"
        "所以「同一批向量换索引」的几行天然相邻，顺着往下看就是一次受控实验。"
    )
    add("")
    add(markdown_table(MATRIX_HEADER, matrix_rows(result)))
    add("")
    add(
        "> 口径提醒：**延迟与建索引耗时是实测值，不可逐位复现**"
        f"（`{'`、`'.join(sorted(TIMING_METRICS))}`）。"
        "召回率、nDCG、MRR、距离计算次数、索引字节都完全可复现。"
    )
    add("")
    add(
        "> 最后一列「前沿」**只列可复现的前沿**（成本轴为距离计算次数 / 索引字节）。"
        "含实测轴的副前沿会在两次运行之间换成员，写进矩阵会让整张表失去可复现性 ——"
        "它们的完整成员见 `frontier.csv` 与第七节。"
    )
    add("")

    # ---- 四、单变量消融 --------------------------------------------
    add("## 四、单变量消融（一次只动一个变量）")
    add("")
    ref = result.reference_run()
    if ref is None:
        add("（没有参考配置，无法做单变量消融）")
        add("")
    else:
        add(
            f"基准行：**`{ref.spec_id}`** —— 精确检索（flat）+ 余弦距离，"
            f"Recall@{ref.metrics.k} = {ref.metrics.recall_at_k:.4f}，"
            f"距离计算次数 = {ref.metrics.n_distance_computations:.1f}。"
        )
        add("")
        add(
            "**为什么基准选 flat**：flat 是精确检索，所以它的召回率就是该表示空间"
            "在 top-k 下的**上限**。于是这张表读起来是「从上限往下掉多少」，"
            "而不是「从一个任意起点往下掉多少」。"
        )
        add("")
        add(
            "`index` 与 `search` 两个轴需要特别处理：参考配置是 flat，而 flat 没有搜索参数，"
            "所以「换索引」时给每个索引配**它的第一档**搜索参数、「换搜索参数」时挑"
            "**档位最多**的索引当实验台 —— 否则会造出不存在的 spec_id，那一行被静默跳过。"
        )
        add("")
        rows = result.ablations.get("single_axis", [])
        if not rows:
            add("（消融表为空）")
        else:
            add(
                markdown_table(
                    ("轴", "空间", "基准", "从", "改为", "配置", "Recall@k", "ΔRecall", "ΔnDCG", "距离计算次数", "Δ成本"),
                    [
                        (
                            r["axis"],
                            "表示" if r.get("space") == "representation" else "检索",
                            r.get("baseline", "-"),
                            r["from"],
                            r["to"],
                            r["spec_id"],
                            f"{r['recall_at_k']:.4f}",
                            f"{r['d_recall']:+.4f}",
                            f"{r['d_ndcg']:+.4f}",
                            f"{r['n_distance_computations']:.1f}",
                            f"{r['d_cost']:+.1f}",
                        )
                        for r in rows
                    ],
                )
            )
        add("")

    # ---- 五、空间归因 ----------------------------------------------
    add("## 五、空间归因（结论可归因到具体变量）")
    add("")
    add(
        "对每个轴，取「只差这一个轴」的配置组，算该组内 Recall@k 的**极差**"
        "（max − min）。极差越大，说明这个轴越值得调。"
        "这张表是验收标准里「可归因到具体变量」的可操作形式。"
    )
    add("")
    attribution = result.ablations.get("attribution", [])
    if not attribution:
        add("（归因表为空）")
    else:
        add(
            markdown_table(
                ("轴", "空间", "取值数", "比了几个", "Recall 最低", "Recall 最高", "极差", "成本比(max/min)", "最优取值", "最差取值", "备注"),
                [
                    (
                        r["axis"],
                        "表示" if r.get("space") == "representation" else "检索",
                        r["n_values"],
                        r["n_compared"],
                        f"{r['recall_min']:.4f}",
                        f"{r['recall_max']:.4f}",
                        f"{r['spread']:.4f}",
                        f"{r['cost_ratio']:.2f}",
                        r["best_value"],
                        r["worst_value"],
                        r["note"] or "-",
                    )
                    for r in attribution
                ],
            )
        )
        add("")
        top = attribution[0]
        add(
            f"**读法**：影响最大的轴是 `{top['axis']}`（极差 {top['spread']:.4f}），"
            f"从 `{top['worst_value']}` 换到 `{top['best_value']}` 就能拿到这个幅度；"
            f"同组的成本比是 {top['cost_ratio']:.2f} 倍。"
        )
        add("")
        add(
            "> 「比了几个」小于「取值数」的轴，说明从当前参考配置出发没法完整消融它 ——"
            "那是**没比**，不是**没影响**。两者必须分开读。"
        )
        add("")
        add(
            "> ⚠️ **极差是局部量**：它是在「其余轴固定为参考值」这个点上测出来的。"
            "换一个参考点，**排序甚至最优项都会变**。本语料就是一个现成的例子："
            "在 `sentence` 分块下 tfidf 最优（0.8333）、hashing 最差（0.5694），"
            "而换到 `fixed` 分块后**正好反过来** —— hashing 0.5208 反而成了最好的，"
            "tfidf 掉到 0.2639（见第六节）。"
            "原因是字符 n-gram 不依赖词边界，块被切断时损失小；词袋则整词丢失。"
            "所以「哪个 embedding 最好」这个结论**必须连着分块一起说**，"
            "只报一个轴的最优值会误导。"
        )
        add("")

    # ---- 六、表示空间的上限 ----------------------------------------
    add("## 六、表示空间的上限（精确检索下）")
    add("")
    ceiling = result.ablations.get("representation_ceiling", {})
    if not ceiling.get("rows"):
        add("（没有 flat 配置，无法给上限）")
        add("")
    else:
        add(
            f"只用**精确检索**（flat）来看：这批向量本身能把答案排多前。"
            f"上限由 `{ceiling['ceiling_spec']}` 取得，"
            f"Recall@k = {ceiling['ceiling_recall']:.4f}。"
        )
        add("")
        add(
            markdown_table(
                ("embedding", "chunking", "Recall@k", "nDCG@k", "MRR", "零召回查询数", "与上限差距", "上限"),
                [
                    (
                        r["embedding"],
                        r["chunking"],
                        f"{r['recall_at_k']:.4f}",
                        f"{r['ndcg_at_k']:.4f}",
                        f"{r['mrr']:.4f}",
                        r["n_queries_zero_recall"],
                        f"{r['gap_to_ceiling']:+.4f}",
                        "★" if r["is_ceiling"] else "",
                    )
                    for r in ceiling["rows"]
                ],
            )
        )
        add("")
        add(
            "**这是「表示空间决定上限」的直接证据**：某个 embedding 在精确检索下"
            "就只有这个数，那么任何索引都救不回来 —— 索引只能让你更便宜地接近它，"
            "不能超过它。反过来，如果某个配置的召回率明显低于它同 embedding 的上限，"
            "那说明是**索引**在掉分，该调的是检索空间。"
        )
        add("")

    # ---- 七、Pareto 前沿 -------------------------------------------
    add("## 七、Pareto 前沿（验收②）")
    add("")
    add(
        "**主前沿的成本轴是「距离计算次数」，不是实测延迟** —— 前者完全由索引结构与"
        "搜索参数决定，逐位可复现；后者受 CPU 降频与调度影响，两次运行不一致。"
        "含实测轴的副前沿另列，并标注「不可逐位复现」，用来验证成本模型确实与延迟同向。"
    )
    add("")
    for pair in frontier_pairs(cfg):
        key = frontier_key(pair)
        timing = has_timing_axis(pair)
        add(f"### {key}" + ("　⚠️ 含实测轴" if timing else "　（可复现）"))
        add("")
        rows = frontier_rows(result, key)
        if not rows:
            add("（空）")
            add("")
            continue
        add(markdown_table(frontier_header(pair), rows))
        add("")
        hv = (result.counters.get("hypervolume") or {}).get(key)
        ref_pt = (result.counters.get("reference_point") or {}).get(key)
        add(
            f"超体积 = **{hv:,.4f}**　参考点 = ({ref_pt[0]:,.4f}, {ref_pt[1]:,.4f})"
            if hv is not None and ref_pt
            else ""
        )
        add(
            "（超体积的参考点取**全部候选点**的最差值再外扩 5% —— "
            "取前沿点的话，超体积就无法在「同一候选集的不同前沿子集」之间比较。）"
        )
        add("")
        if timing:
            add(
                "> ⚠️ 这条前沿含实测轴，**不可逐位复现**：延迟相近的两个配置会在两次运行间"
                "交换前沿成员。它只用于验证「距离计算次数确实与真实延迟同向」，"
                "不作为结论依据。"
            )
            add("")
        # 端点性质与去重说明
        add(
            "> **读前沿要看曲线形状，不要只看成员。** 两个端点各自单独最优："
            "最左端是「成本最低」（它可能召回率很差，但没有任何点比它更便宜，"
            "所以按定义它非支配）；最右端是「召回率最高」。"
            "中间的点才是真正的取舍点。"
        )
        add("")
        dup = (result.counters.get("frontier_duplicates") or {}).get(key) or []
        if dup:
            add(
                f"> 被去重的配置：{', '.join(dup)} —— 它们的目标向量与某个代表项"
                "**逐位相同**（互不支配，因为支配要求至少一维严格更好）。"
                "「某个搜索档位完全无效」就是这么被发现的。"
            )
            add("")

    # 成本模型验证
    add("### 成本模型验证：距离计算次数 ↔ 实测延迟")
    add("")
    add(
        "把「距离计算次数」当成本轴，前提是它与真实延迟同向。这是一个**能被证伪的断言**，"
        "所以必须显式验证：下面按索引类型看两个量的关系。"
    )
    add("")
    add(
        markdown_table(
            ("index", "配置数", "距离计算次数 min~max", "延迟 min~max (ms)", "成本与延迟是否同向"),
            _cost_model_rows(result),
        )
    )
    add("")

    # ---- 八、达额成本 ----------------------------------------------
    add("## 八、达额成本（前沿的可决策版本）")
    add("")
    reach = result.ablations.get("reach_cost", {})
    if not reach.get("rows"):
        add("（无数据）")
        add("")
    else:
        add(
            f"可达上限 Recall@k = {reach['ceiling_recall']:.4f}"
            f"（`{reach['ceiling_spec']}`）。下表问的是："
            "**要达到上限的 X%，最省成本的配置是哪个、要花多少。**"
        )
        add("")
        add(
            markdown_table(
                ("目标比例", "目标 Recall", "最省配置", "index", "距离计算次数", "延迟(ms)", "实际 Recall", "成本 / 精确检索"),
                [
                    (
                        f"{r['threshold']:.0%}",
                        f"{r['target_recall']:.4f}",
                        r["spec_id"] or "（不可达）",
                        r.get("index", "-"),
                        f"{r['n_distance_computations']:.1f}" if r["reachable"] else "-",
                        f"{r['query_latency_ms']:.3f}" if r["reachable"] else "-",
                        f"{r['recall_at_k']:.4f}" if r["reachable"] else "-",
                        f"{r['cost_vs_exact']:.2f}×" if r.get("cost_vs_exact") is not None else "-",
                    )
                    for r in reach["rows"]
                ],
            )
        )
        add("")
        add(
            "最后一列是这张表最有用的一栏：**「成本 / 精确检索」**。"
            "它直接回答「用近似索引能省多少」，而且分母是同一份向量上的精确检索，"
            "所以这个比值不掺任何表示空间的差异。"
        )
        add("")

    # ---- 九、预算敏感性 --------------------------------------------
    add("## 九、预算敏感性（预算改变赢家）")
    add("")
    rows = result.ablations.get("budget_sensitivity", [])
    if not rows:
        add("（无数据）")
        add("")
    else:
        add(
            "**两列差距必须分开看**（TokenizerFactory 项目踩过把它们合成一列的坑）："
            "「与全局最优的差距」是预算约束造成的，"
            "「该档 embedding 极差」是向量质量造成的。处置完全不同 ——"
            "前者加预算，后者换 embedding。"
        )
        add("")
        add(
            markdown_table(
                ("成本预算", "可用配置数", "该档最优", "该档 Recall", "与全局最优差距", "差距%", "该档 embedding 极差", "该档最优 embedding"),
                [
                    (
                        f"{r['budget']:.1f}",
                        r["n_within"],
                        r["best_spec"],
                        f"{r['best_recall']:.4f}",
                        f"{r['gap_to_ceiling']:+.4f}",
                        f"{r['gap_pct']:+.2f}%",
                        f"{r['embedding_spread']:.4f}",
                        r["best_embedding"],
                    )
                    for r in rows
                ],
            )
        )
        add("")

    # ---- 十、血缘与版本 --------------------------------------------
    add("## 十、血缘与版本")
    add("")
    if result.dag is not None:
        s = result.dag.summary()
        add(
            markdown_table(
                ("项", "值"),
                [
                    ("节点数", s["nodes"]),
                    ("边数", s["edges"]),
                    ("收拢点（多输入节点）", s["merge_points"]),
                    ("平均扇出", f"{s['fan_out_ratio']:.2f}"),
                    ("各类型节点", "　".join(f"{k}={v}" for k, v in sorted(s["by_kind"].items()))),
                ],
            )
        )
        add("")
        merges = result.dag.merge_points()
        if merges:
            add("**收拢点恰好接住前沿候选** —— 这是本形状的存在理由，看一张图就知道谁进了前沿：")
            add("")
            for nid in merges:
                node = result.dag.nodes[nid]
                add(f"- `{nid}`　←　{len(node.inputs)} 个输入：{', '.join(node.inputs)}")
            add("")
    if result.version is not None:
        add(
            f"版本 `{result.version.version_id}`（内容寻址：digest 由**入选配置**决定，"
            f"只是重跑则不变）。入选规则：{result.counters.get('selected_reason', '-')}。"
        )
    add("")
    add("### 可复现性说明")
    add("")
    add(
        "- **可复现**：召回率 / nDCG / MRR / 距离计算次数 / 索引字节 / 矩阵 / 前沿 / 版本。"
    )
    add(
        "- **可复现（结构与导出）**：血缘图。`lineage.json` 的 round-trip "
        "（读回再导出）逐位相同，DOT 与 ASCII 输出也只依赖图本身，与构建路径无关。"
        "唯一的例外是节点的 `created_at` —— 那是**墙上时间**（provenance），"
        "所以两次 run 的 `lineage.json` 不会逐字节相同。"
    )
    add(
        "- **不可复现**：延迟与建索引耗时（实测值）。含它们的副前沿在报告里已标注，"
        "不作为结论依据。"
    )
    add(
        f"- 代码版本：`{result.counters.get('code_version', '-')}`；"
        f"本次耗时 {result.counters.get('elapsed_seconds', '-')} s。"
    )
    add("")
    return "\n".join(out)


def _cost_model_rows(result: RetrievalResult) -> list[list[str]]:
    """按索引类型汇总「距离计算次数」与「实测延迟」，看两者是否同向。"""
    rows: list[list[str]] = []
    for index in result.axis_values("index"):
        group = [r for r in result.runs if r.spec.index == index]
        if not group:
            continue
        dcs = [r.metrics.n_distance_computations for r in group]
        lats = [r.metrics.query_latency_ms for r in group]
        # 同向判据：按距离计算次数排序后，延迟是否也基本递增（Spearman 符号一致率）
        order = sorted(range(len(group)), key=lambda i: (dcs[i], group[i].spec_id))
        pairs = 0
        agree = 0
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                i, j = order[a], order[b]
                if dcs[i] == dcs[j]:
                    continue
                pairs += 1
                if lats[j] >= lats[i]:
                    agree += 1
        rows.append(
            [
                index,
                str(len(group)),
                f"{min(dcs):.1f} ~ {max(dcs):.1f}",
                f"{min(lats):.3f} ~ {max(lats):.3f}",
                f"{agree / pairs:.0%}（{agree}/{pairs} 对同向）" if pairs else "（无区分度）",
            ]
        )
    return rows
