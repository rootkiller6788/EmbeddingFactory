"""命令行入口。

设计约定（与同工作区其它项目共用一套，见 python-pipeline-scaffold skill）：

- **``--out`` 只属于 ``run``**。其余子命令是**只读观察者**，它们只读 ``out/``
  里的产物，不写任何东西 —— 观察者会改状态是最容易把人绕晕的一类设计。
- **只读子命令不做检索**。它们从 ``result.json`` + ``runs.jsonl`` 重建结果对象；
  两处例外都是**载入**而不是重算：``query`` 从 ``artifacts/<spec>.json``
  载入一个检索组件并真的跑一次查询（那正好是「导出的产物真的能用」这条承诺的
  可执行检查），``dag`` 从 ``lineage.json`` 重建血缘图（否则 ``--dot``
  只能打印一句提示）。
- **未知 ``spec_id`` 返回 1 并给相近建议**，不抛栈。
- 刻意**没有** ``rounds`` 子命令：本项目不是一轮一轮迭代的。
- 刻意**没有** ``searchers``：本项目没有可插拔搜索器，配置里的交叉积就是全部候选。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import report
from .compare import frontier_key, frontier_pairs
from .config import load_config
from .index import build_index
from .lineage import LineageDAG
from .metrics import METRIC_DIRECTIONS
from .pipeline import run_pipeline, write_outputs
from .schema import (
    CorpusStats,
    RetrievalMetrics,
    RetrievalResult,
    RetrievalRun,
    RetrievalSpec,
)


# ----------------------------------------------------------------------
# 读取产物
# ----------------------------------------------------------------------
def _load_result(out_dir: Path) -> RetrievalResult:
    result_path = out_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"找不到 {result_path}；先跑一次 run（例如 python main.py run --out {out_dir}）"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = RetrievalResult(corpus=CorpusStats(**payload["corpus"]))
    result.matrix = payload.get("matrix", [])
    result.frontiers = payload.get("frontiers", {})
    result.ablations = payload.get("ablations", {})
    result.selected = payload.get("selected", "")
    result.counters = payload.get("counters", {})
    result.cfg = payload.get("config", {})

    runs_path = out_dir / "runs.jsonl"
    if runs_path.exists():
        runs: list[RetrievalRun] = []
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            runs.append(_run_from_dict(json.loads(line)))
        result.runs = runs
    return result


def _run_from_dict(payload: dict[str, Any]) -> RetrievalRun:
    spec_dict = dict(payload["spec"])
    spec_dict.pop("spec_id", None)
    spec_dict.pop("search_tag", None)
    spec_dict.pop("representation", None)
    spec_dict.pop("retrieval", None)
    return RetrievalRun(
        spec=RetrievalSpec(**spec_dict),
        metrics=RetrievalMetrics(**payload["metrics"]),
        sample=[
            (
                str(item["query_id"]),
                str(item["text"]),
                [(str(t["doc_id"]), int(t["rank"]), bool(t["relevant"])) for t in item.get("top", [])],
            )
            for item in payload.get("sample", [])
        ],
        meta=dict(payload.get("meta", {})),
    )


def _print_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    print(report.markdown_table(header, rows))


def _suggest(result: RetrievalResult, spec_id: str) -> str:
    known = result.known_spec_ids()
    parts = spec_id.split("-")
    prefix = [k for k in known if k.startswith(parts[0])] if parts else []
    pool = prefix or known
    return ", ".join(pool[:8]) + ("…" if len(pool) > 8 else "")


# ----------------------------------------------------------------------
# 子命令
# ----------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    result, bundle = run_pipeline(cfg, verbose=not args.quiet)
    written = write_outputs(result, bundle, args.out)
    report_path = Path(args.out) / "report.md"
    report_path.write_text(report.render_markdown(result), encoding="utf-8")
    written["report.md"] = report_path

    print()
    print(report.summary_line(result))
    print()
    print(f"产物目录：{Path(args.out).resolve()}")
    for name in sorted(written):
        print(f"  {name}")
    print()
    print(f"下一步：python main.py compare  --out {args.out}")
    print(f"        python main.py frontier --out {args.out}")
    print(f"        python main.py ablation --out {args.out}")
    print(f"        python main.py query {result.selected} \"你的查询文本\" --out {args.out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    if not result.runs:
        print("没有可对比的 run。", file=sys.stderr)
        return 1
    _print_table(report.MATRIX_HEADER, report.matrix_rows(result))
    print()
    print(report.summary_line(result))
    return 0


def _cmd_frontier(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    pairs = frontier_pairs(result.cfg) if result.cfg else []
    if args.pair:
        wanted = args.pair.replace("+", " ")
        pairs = [p for p in pairs if " ".join(p) == wanted]
        if not pairs:
            print(
                f"未知目标对 {args.pair!r}；可用："
                + ", ".join(frontier_key(p) for p in frontier_pairs(result.cfg)),
                file=sys.stderr,
            )
            return 1
    if not pairs:
        print("结果里没有记录目标对。", file=sys.stderr)
        return 1

    for pair in pairs:
        key = frontier_key(pair)
        rows = report.frontier_rows(result, key)
        print(f"## {key}")
        print()
        if not rows:
            print("（空）")
        else:
            _print_table(report.frontier_header(pair), rows)
        hv = (result.counters.get("hypervolume") or {}).get(key)
        if hv is not None:
            print()
            print(f"超体积：{hv:,.4f}　（参考点口径见 report.md 第七节）")
        print()
    print("★ = 最终入选")
    return 0


def _cmd_ablation(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    ab = result.ablations or {}
    ref = result.reference_run()
    if ref is not None:
        print(f"基准行：{ref.spec_id}　Recall@k = {ref.metrics.recall_at_k:.4f}")
        print()

    print("## 单变量消融（一次只动一个变量）")
    print()
    rows = ab.get("single_axis", [])
    if rows:
        _print_table(
            ("轴", "从", "改为", "配置", "Recall@k", "ΔRecall", "距离计算次数", "Δ成本"),
            [
                (
                    r["axis"],
                    r["from"],
                    r["to"],
                    r["spec_id"],
                    f"{r['recall_at_k']:.4f}",
                    f"{r['d_recall']:+.4f}",
                    f"{r['n_distance_computations']:.1f}",
                    f"{r['d_cost']:+.1f}",
                )
                for r in rows
            ],
        )
        # 脚注：index / search 两个轴是特判（参考配置是 flat，而 flat 没有搜索参数）。
        # 不写出来的话，读者会困惑「为什么 flat→hnsw 这一行的 hnsw 带了个 ef16」。
        seen_notes: dict[str, str] = {}
        for r in rows:
            if r.get("note"):
                seen_notes.setdefault(str(r["axis"]), str(r["note"]))
        if seen_notes:
            print()
            for axis, note in seen_notes.items():
                print(f"注（{axis} 轴）：{note}")
    else:
        print("（空）")
    print()

    print("## 空间归因（极差 = 该轴最多能移动多少召回率）")
    print()
    attribution = ab.get("attribution", [])
    if attribution:
        _print_table(
            ("轴", "空间", "取值数", "Recall 最低", "Recall 最高", "极差", "最优取值", "最差取值"),
            [
                (
                    r["axis"],
                    "表示" if r["axis"] in ("embedding", "chunking") else "检索",
                    r["n_values"],
                    f"{r['recall_min']:.4f}",
                    f"{r['recall_max']:.4f}",
                    f"{r['spread']:.4f}",
                    r["best_value"],
                    r["worst_value"],
                )
                for r in attribution
            ],
        )
    else:
        print("（空）")
    print()

    print("## 达额成本")
    print()
    reach = ab.get("reach_cost", {})
    if reach.get("rows"):
        _print_table(
            ("目标比例", "最省配置", "距离计算次数", "实际 Recall", "成本 / 精确检索"),
            [
                (
                    f"{r['threshold']:.0%}",
                    r["spec_id"] or "（不可达）",
                    f"{r['n_distance_computations']:.1f}" if r["reachable"] else "-",
                    f"{r['recall_at_k']:.4f}" if r["reachable"] else "-",
                    f"{r['cost_vs_exact']:.2f}×" if r.get("cost_vs_exact") is not None else "-",
                )
                for r in reach["rows"]
            ],
        )
    else:
        print("（空）")
    return 0


def _cmd_spaces(args: argparse.Namespace) -> int:
    """把「表示空间 / 检索空间」两组轴的取值与效果并排打出来。"""
    result = _load_result(Path(args.out))
    if not result.runs:
        print("没有可用的 run。", file=sys.stderr)
        return 1

    print("## 表示空间（决定召回率上限）")
    print()
    ceiling = (result.ablations or {}).get("representation_ceiling", {})
    if ceiling.get("rows"):
        _print_table(
            ("embedding", "chunking", "Recall@k（精确检索）", "nDCG@k", "零召回", "与上限差距"),
            [
                (
                    r["embedding"],
                    r["chunking"],
                    f"{r['recall_at_k']:.4f}",
                    f"{r['ndcg_at_k']:.4f}",
                    r["n_queries_zero_recall"],
                    f"{r['gap_to_ceiling']:+.4f}",
                )
                for r in ceiling["rows"]
            ],
        )
    else:
        print("（没有 flat 配置，无法给上限）")
    print()

    print("## 检索空间（决定达到上限的成本）")
    print()
    rows: list[list[str]] = []
    for index in result.axis_values("index"):
        for distance in result.axis_values("distance"):
            for search in result.axis_values("search"):
                group = [
                    r
                    for r in result.runs
                    if r.spec.index == index
                    and r.spec.distance == distance
                    and r.spec.search_tag == search
                ]
                if not group:
                    continue
                best = max(group, key=lambda r: (r.metrics.recall_at_k, r.spec_id))
                rows.append(
                    [
                        index,
                        distance,
                        search,
                        str(len(group)),
                        f"{best.metrics.recall_at_k:.4f}",
                        f"{min(r.metrics.n_distance_computations for r in group):.1f} ~ "
                        f"{max(r.metrics.n_distance_computations for r in group):.1f}",
                        f"{best.metrics.query_latency_ms:.3f}",
                    ]
                )
    _print_table(
        ("index", "distance", "search", "配置数", "最好 Recall@k", "距离计算次数范围", "该档延迟(ms)"),
        rows,
    )
    print()
    print("提示：`flat` 是精确检索，它的召回率就是「该表示空间的上限」；")
    print("      近似索引（ivf / hnsw）的价值在于用更少的距离计算次数接近它。")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    if result.find(args.spec_id) is None:
        print(f"未知 spec_id：{args.spec_id}", file=sys.stderr)
        print(f"相近的可选项：{_suggest(result, args.spec_id)}", file=sys.stderr)
        return 1
    print(report.explain(result, args.spec_id))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    main_key = (
        frontier_key(result.cfg["pareto"]["objectives"])
        if result.cfg.get("pareto")
        else ""
    )
    print(f"共 {len(result.runs)} 个配置（★ 入选，▲ 在主前沿上，· 其它）")
    print()
    for run in result.runs:
        mark = (
            "★"
            if run.spec_id == result.selected
            else ("▲" if run.spec_id in result.frontiers.get(main_key, []) else "·")
        )
        m = run.metrics
        print(
            f"  {mark} {run.spec_id:<32} recall={m.recall_at_k:.4f} "
            f"ndcg={m.ndcg_at_k:.4f} dc={m.n_distance_computations:>7.1f} "
            f"lat={m.query_latency_ms:>7.3f}ms"
        )
    return 0


def _load_dag(out_dir: Path) -> Any:
    """从 ``lineage.json`` 重建血缘图。

    ``dag`` 是只读子命令，不能依赖「同一次 run 的内存对象」；但 ``--dot`` /
    ``--selected-only`` / ``--limit`` 都需要图对象。不重建的话 ``--dot``
    会静默退化成「打印摘要 + 一句提示」—— 用户以为拿到了 DOT，却拿到别的东西。
    """
    path = out_dir / "lineage.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("nodes"):
        return None
    return LineageDAG.from_dict(payload)


def _cmd_dag(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.out))
    dag = result.dag if result.dag is not None else _load_dag(Path(args.out))
    if dag is None:
        print(f"找不到 {Path(args.out) / 'lineage.json'}", file=sys.stderr)
        return 1
    if args.dot:
        print(dag.to_dot(only_selected_ancestry=args.selected_only))
    else:
        print(dag.to_ascii(limit=args.limit))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    """载入导出的检索组件，真的跑一次查询 —— 「产物真的能用」的可执行检查。"""
    out_dir = Path(args.out)
    path = out_dir / "artifacts" / f"{args.spec_id}.json"
    if not path.exists():
        available = sorted(p.stem for p in (out_dir / "artifacts").glob("*.json"))
        print(f"找不到 {path}", file=sys.stderr)
        if available:
            print(f"可用的 spec：{', '.join(available[:10])}", file=sys.stderr)
        else:
            print(
                "（artifacts/ 里没有可载入的检索组件 —— "
                "本项目只导出**入选配置**的组件）",
                file=sys.stderr,
            )
        return 1

    artifact = json.loads(path.read_text(encoding="utf-8"))
    from .embed import EmbeddingModel

    emb = artifact["embedding"]
    model = EmbeddingModel(
        name=str(emb["name"]),
        dim=int(emb["dim"]),
        vocab=list(emb.get("vocab", [])),
        index={t: i for i, t in enumerate(emb.get("vocab", []))},
        idf={k: float(v) for k, v in (emb.get("idf") or {}).items()},
        avg_len=float(emb.get("avg_len", 1.0)),
        params=dict(emb.get("params", {})),
        bucket_idf=[float(v) for v in (emb.get("bucket_idf") or [])],
    )
    corpus = artifact["corpus"]
    vectors = corpus["vectors"]
    doc_ids = corpus["doc_ids"]
    chunk_ids = corpus["chunk_ids"]
    idx = artifact["index"]
    # 种子必须从产物里读，不能自己定一个：ivf 的分簇由它派生，
    # 用错种子会得到一个「能跑但结果与基准不一致」的索引（不报错，只是数字变了）。
    index = build_index(
        str(idx["name"]),
        vectors,
        str(idx["distance"]),
        {k: v for k, v in (idx.get("params") or {}).items()},
        int(artifact.get("seed", 0)),
    )

    text = " ".join(args.text)
    qvec = model.transform(text)
    result = index.search(qvec, int(artifact.get("fetch_k", 20)))

    from .retrieve import merge_to_docs

    ranked = merge_to_docs(result.ids, doc_ids, int(artifact.get("k", 5)))

    print(f"spec: {args.spec_id}")
    print(f"  表示空间：embedding={artifact['spec']['embedding']}　chunking={artifact['spec']['chunking']}")
    print(f"  检索空间：index={idx['name']}　distance={idx['distance']}　search={artifact['spec']['search_tag']}")
    print(f"  语料：{len(vectors)} 个块 / {len(set(doc_ids))} 篇文档")
    print()
    print(f"查询：{text!r}")
    print(
        f"距离计算次数：{result.n_distance_computations}　"
        f"（该配置在基准里的均值是 "
        f"{artifact.get('metrics', {}).get('n_distance_computations', '-')}）"
    )
    print()
    print("top-k 文档：")
    for i, doc in enumerate(ranked, start=1):
        print(f"  {i}. {doc}")
    print()
    print("命中的块（原始 chunk）：")
    for i in result.ids[: min(5, len(result.ids))]:
        print(f"  {chunk_ids[i]}  ({doc_ids[i]})")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embeddingfactory",
        description=(
            "把「表示空间」与「检索空间」解耦：在同一份语料上对比 "
            "embedding × chunking × index × distance × search，"
            "产出消融矩阵与 Latency–Recall 前沿"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="跑完整流水线并落盘产物")
    p_run.add_argument("--config", default="configs/default.json", help="配置文件路径")
    p_run.add_argument("--out", default="out", help="产物目录")
    p_run.add_argument("--quiet", action="store_true", help="不打印逐个配置的进度")
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare", help="打印消融矩阵（只读）")
    p_cmp.add_argument("--out", default="out")
    p_cmp.set_defaults(func=_cmd_compare)

    p_front = sub.add_parser("frontier", help="打印 Pareto 前沿（只读）")
    p_front.add_argument("--out", default="out")
    p_front.add_argument(
        "--pair",
        default="",
        help="只看某个目标对，例如 n_distance_computations+recall_at_k",
    )
    p_front.set_defaults(func=_cmd_frontier)

    p_abl = sub.add_parser("ablation", help="打印单变量消融 / 空间归因 / 达额成本（只读）")
    p_abl.add_argument("--out", default="out")
    p_abl.set_defaults(func=_cmd_ablation)

    p_sp = sub.add_parser("spaces", help="打印两个空间的取值与效果（只读）")
    p_sp.add_argument("--out", default="out")
    p_sp.set_defaults(func=_cmd_spaces)

    p_exp = sub.add_parser("explain", help="解释某个配置的结果（只读）")
    p_exp.add_argument("spec_id")
    p_exp.add_argument("--out", default="out")
    p_exp.set_defaults(func=_cmd_explain)

    p_list = sub.add_parser("list", help="列出全部配置（只读）")
    p_list.add_argument("--out", default="out")
    p_list.set_defaults(func=_cmd_list)

    p_dag = sub.add_parser("dag", help="打印血缘图（只读）")
    p_dag.add_argument("--out", default="out")
    p_dag.add_argument("--dot", action="store_true", help="输出 Graphviz DOT")
    p_dag.add_argument(
        "--selected-only",
        action="store_true",
        help="只画入选配置的祖先链（配合 --dot）",
    )
    p_dag.add_argument("--limit", type=int, default=30, help="ASCII 输出最多显示多少节点")
    p_dag.set_defaults(func=_cmd_dag)

    p_q = sub.add_parser("query", help="用导出的检索组件真的跑一次查询（只读）")
    p_q.add_argument("spec_id")
    p_q.add_argument("text", nargs="+", help="查询文本（多个词会拼成一句）")
    p_q.add_argument("--out", default="out")
    p_q.set_defaults(func=_cmd_query)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"配置或参数错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
