"""端到端：设计空间的展开、收拢、落盘、只读契约与可复现性。

这一层是**验收标准的最终落点**：

1. **消融矩阵**：60 个配置 × 5 个设计轴，矩阵的每一行都能对回设计空间的一个点；
2. **Latency–Recall 前沿**：主前沿用可复现的成本轴，含实测轴的副前沿被标注；
3. **收拢点的输入恰好是前沿候选** —— 这是「扇出-收拢型」这个形状存在的理由，
   多一个（把落选者接进来）或少一个（前沿没接住）都算 bug；
4. **只读子命令不改产物** —— 观察者会改状态是最容易把人绕晕的一类设计。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from embeddingfactory import compare, report
from embeddingfactory.cli import main as cli_main
from embeddingfactory.config import expand_specs
from embeddingfactory.index import build_index
from embeddingfactory.lineage import LineageDAG, bench_node_id
from embeddingfactory.pipeline import MATRIX_HEADER, run_pipeline, write_outputs
from embeddingfactory.retrieve import evaluate, merge_to_docs
from embeddingfactory.embed import EmbeddingModel
from embeddingfactory.schema import DESIGN_AXES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def rerun(cfg):
    """第二次完整跑。用于「确定性」断言 —— 只跑一次的测试证明不了可复现。"""
    return run_pipeline(cfg)[0]


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ----------------------------------------------------------------------
# 设计空间
# ----------------------------------------------------------------------
def test_design_space_expands_to_60(cfg) -> None:
    """60 = embedding(3) × chunking(2) × index(3) × distance(2) × search。

    **不能**把 5 个轴的取值数直接相乘：``search`` 的取值依赖索引
    （flat 只有 ``exact``，ivf 有 np1/np3，hnsw 有 ef16/ef64），
    所以正确口径是按索引分档求和。
    """
    specs = expand_specs(cfg)
    r, q = cfg["representation"], cfg["retrieval"]
    expected = (
        len(r["embeddings"])
        * len(r["chunkings"])
        * len(q["distances"])
        * sum(len(q["search_params"][i]) for i in q["indexes"])
    )
    assert len(specs) == 60
    assert len(specs) == expected
    assert len({s.spec_id for s in specs}) == 60, "spec_id 必须唯一"


def test_expansion_order_is_representation_first(cfg) -> None:
    """展开顺序 = embedding → chunking → index → distance → search。

    顺序变了，「同一批向量换索引」这类行就不再相邻，读者要自己找。
    """
    specs = expand_specs(cfg)
    embeddings = [s.embedding for s in specs]
    assert embeddings[0] == cfg["representation"]["embeddings"][0]
    assert embeddings == sorted(embeddings, key=embeddings.index)  # 连续分块
    # 一个 embedding 段内的顺序是 chunking → index → distance → search
    first_segment = [s for s in specs if s.embedding == embeddings[0]]
    chunkings = [s.chunking for s in first_segment]
    assert chunkings == sorted(chunkings, key=chunkings.index)
    assert set(chunkings) == set(cfg["representation"]["chunkings"])


def test_runs_match_the_design_space(result, cfg) -> None:
    assert result.known_spec_ids() == sorted(s.spec_id for s in expand_specs(cfg))
    assert len(result.runs) == 60
    assert len(result.matrix) == 60


def test_axis_values_are_the_configured_ones(result, cfg) -> None:
    assert result.axis_values("embedding") == sorted(cfg["representation"]["embeddings"])
    assert result.axis_values("chunking") == sorted(cfg["representation"]["chunkings"])
    assert result.axis_values("index") == sorted(cfg["retrieval"]["indexes"])
    assert result.axis_values("distance") == sorted(cfg["retrieval"]["distances"])
    assert result.axes() == list(DESIGN_AXES)


# ----------------------------------------------------------------------
# 收拢点：本形状的存在理由
# ----------------------------------------------------------------------
def test_merge_point_inputs_are_exactly_the_frontier(result, cfg) -> None:
    """``sel:main`` 的输入必须**恰好**是主前沿候选 —— 一个不多、一个不少。

    多一个 = 把落选者偷偷接进来了；少一个 = 前沿没被接住。
    这条不变量保证「谁进了前沿」看一张图就知道，不必翻报告。
    """
    key = compare.frontier_key(cfg["pareto"]["objectives"])
    frontier_ids = sorted(result.frontiers[key])
    assert result.dag.parents("sel:main") == [bench_node_id(sid) for sid in frontier_ids]
    assert len(result.dag.parents("sel:main")) == len(frontier_ids) == 5


def test_merge_points_are_select_and_artifact(result) -> None:
    """多输入节点只有 ``sel:main``（收拢点）。

    ``art:main`` 只有一个输入（``sel:main``）—— 它不是收拢点，而是收拢之后的
    最终产物，所以不该出现在 ``merge_points()`` 里。这条区分很重要：
    把 art 也算成收拢点的话，「收拢点的输入恰好是前沿候选」这条不变量就没法写了。
    """
    assert result.dag.merge_points() == ["sel:main"]
    assert result.dag.by_kind("artifact") == ["art:main"]
    assert len(result.dag.parents("art:main")) == 1


def test_every_run_reaches_the_corpus_root(result, corpus) -> None:
    """每个 spec 的血缘链必须是本图里的一条真实路径，终点是 ``corpus``。"""
    for run in result.runs:
        path = result.dag.path_to_root(bench_node_id(run.spec_id))
        assert path[-1] == "corpus"
        assert f"embed:{run.spec.embedding}+{run.spec.chunking}" in path
        assert f"chunk:{run.spec.chunking}" in path


def test_one_embed_node_is_reused_by_many_indexes(result, cfg) -> None:
    """一个 ``embed`` 节点有多个 ``idx`` 子节点 ——
    这正是「表示空间的产物被检索空间复用」的图形化表达。

    每套 (embedding, chunking) 的 idx 子节点数 = distance × Σ 各索引的搜索档位数。
    """
    q = cfg["retrieval"]
    expected = len(q["distances"]) * sum(len(q["search_params"][i]) for i in q["indexes"])
    assert expected == 10
    for embedding in cfg["representation"]["embeddings"]:
        for chunking in cfg["representation"]["chunkings"]:
            node = f"embed:{embedding}+{chunking}"
            idx_children = [c for c in result.dag.children(node) if c.startswith("idx:")]
            assert len(idx_children) == expected, f"{node} 只被复用了 {len(idx_children)} 次"


def test_dag_summary_counts(result, cfg) -> None:
    summary = result.dag.summary()
    n_chunkings = len(cfg["representation"]["chunkings"])
    n_embeddings = len(cfg["representation"]["embeddings"])
    assert summary["by_kind"]["corpus"] == 1
    assert summary["by_kind"]["chunk"] == n_chunkings
    assert summary["by_kind"]["embed"] == n_embeddings * n_chunkings
    assert summary["by_kind"]["index"] == 60
    assert summary["by_kind"]["bench"] == 60
    assert summary["by_kind"]["select"] == 1
    assert summary["by_kind"]["artifact"] == 1
    assert summary["nodes"] == 1 + n_chunkings + n_embeddings * n_chunkings + 60 + 60 + 2
    # 边：corpus→chunk(2) + chunk→embed(6) + embed→idx(60) + idx→bench(60)
    #     + bench→sel(5，恰好是主前沿成员) + sel→art(1)
    assert summary["edges"] == n_chunkings + n_embeddings * n_chunkings + 60 + 60 + 5 + 1
    assert summary["merge_points"] == 1
    assert summary["fan_out_ratio"] > 1.0


def test_dag_is_acyclic_and_topological_order_is_stable(result) -> None:
    order = result.dag.topological()
    assert len(order) == len(result.dag.nodes)
    position = {n: i for i, n in enumerate(order)}
    for a, b in result.dag.edges:
        assert position[a] < position[b]
    assert order == result.dag.topological()


# ----------------------------------------------------------------------
# 血缘的可复现导出
# ----------------------------------------------------------------------
def test_lineage_round_trip_is_bitwise_identical(result) -> None:
    """``to_dict`` → ``from_dict`` → ``to_dict`` 必须逐位相同。

    两个细节容易漏：``outputs`` 要排序后导出、``edges`` 也要排序 ——
    否则同一个图的两条构建路径会导出两份不同的 ``lineage.json``，
    于是「产物是否可复现」这个检查永远过不了。
    """
    payload = result.dag.to_dict()
    rebuilt = LineageDAG.from_dict(payload)
    assert rebuilt.to_dict() == payload


def test_lineage_json_round_trips_from_disk(result, out_dir) -> None:
    """只读的 ``dag`` 子命令就是走这条路重建图的。"""
    payload = json.loads((out_dir / "lineage.json").read_text(encoding="utf-8"))
    assert payload == result.dag.to_dict()
    rebuilt = LineageDAG.from_dict(payload)
    assert rebuilt.merge_points() == result.dag.merge_points()
    assert rebuilt.topological() == result.dag.topological()


def test_lineage_structure_is_deterministic_across_runs(result, rerun) -> None:
    """图的结构（节点 / 边 / 参数 / 统计）必须跨 run 逐位相同。

    唯一例外是节点的 ``created_at`` —— 那是**墙上时间**（provenance），
    所以两次 run 的 ``lineage.json`` 不会逐字节相同。报告里如实写明了这一点，
    测试也如实只比结构。
    """
    def strip_created_at(payload: dict) -> dict:
        return {
            "nodes": [{k: v for k, v in n.items() if k != "created_at"} for n in payload["nodes"]],
            "edges": payload["edges"],
            "summary": payload["summary"],
        }

    assert strip_created_at(result.dag.to_dict()) == strip_created_at(rerun.dag.to_dict())
    assert result.dag.to_dot() == rerun.dag.to_dot()
    assert result.dag.to_ascii(limit=20) == rerun.dag.to_ascii(limit=20)


def test_lineage_dot_is_independent_of_build_path(result) -> None:
    rebuilt = LineageDAG.from_dict(result.dag.to_dict())
    assert rebuilt.to_dot() == result.dag.to_dot()
    assert rebuilt.to_dot(only_selected_ancestry=True) == result.dag.to_dot(
        only_selected_ancestry=True
    )


def test_selected_dot_is_a_proper_subgraph(result) -> None:
    full = result.dag.to_dot()
    selected = result.dag.to_dot(only_selected_ancestry=True)
    assert len(selected) < len(full), "入选链应该比全图小得多"
    assert "sel:main" in selected
    assert "art:main" in selected


# ----------------------------------------------------------------------
# 落盘产物
# ----------------------------------------------------------------------
def test_written_files(out_dir) -> None:
    expected = {
        "matrix.csv",
        "frontier.csv",
        "runs.jsonl",
        "result.json",
        "ablations.json",
        "lineage.json",
        "lineage.dot",
        "lineage_selected.dot",
        "version.json",
        "report.md",
    }
    assert expected <= set(_snapshot(out_dir))
    assert list((out_dir / "artifacts").glob("*.json"))


def test_matrix_csv_shape(out_dir, result) -> None:
    lines = (out_dir / "matrix.csv").read_text(encoding="utf-8").strip().splitlines()
    assert tuple(lines[0].split(",")) == MATRIX_HEADER
    assert len(lines) - 1 == 60
    # 第一行是参考配置（展开顺序的第一个），且它的 frontier 列不该为空
    first = dict(zip(MATRIX_HEADER, lines[1].split(",")))
    assert first["spec_id"] == result.runs[0].spec_id


def test_frontier_csv_covers_every_objective_pair(out_dir, result) -> None:
    lines = (out_dir / "frontier.csv").read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    assert "objective_pair" in header
    seen: dict[str, int] = {}
    for line in lines[1:]:
        row = dict(zip(header, line.split(",")))
        seen[row["objective_pair"]] = seen.get(row["objective_pair"], 0) + 1
    assert seen == result.counters["frontier_sizes"]


def test_runs_jsonl_has_samples_and_result_json_does_not(out_dir) -> None:
    """样本只进 ``runs.jsonl`` —— ``result.json`` 要能直接给人看，不该被样本撑爆。"""
    runs = [
        json.loads(line)
        for line in (out_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(runs) == 60
    assert all(run["sample"] for run in runs)

    payload = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert "sample" not in json.dumps(payload["matrix"])
    assert payload["selected"] == runs[0]["spec_id"] or payload["selected"] in {
        r["spec_id"] for r in runs
    }


def test_result_json_carries_the_config_but_not_internal_paths(out_dir) -> None:
    payload = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert "corpus" in payload["config"]
    assert "representation" in payload["config"]
    assert not any(k.startswith("_") for k in payload["config"])


# ----------------------------------------------------------------------
# artifact：导出的检索组件必须真的能用
# ----------------------------------------------------------------------
def test_artifact_reproduces_the_selected_run(result, bundle, out_dir, cfg) -> None:
    """载入 ``artifacts/<selected>.json``，重建模型与索引，重跑全部查询 ——
    召回率必须与 run 里记录的一致。

    这是「导出的产物真的能用」这条承诺的可执行检查。向量落盘时按 6 位小数
    截断，所以允许极小的浮点差异。

    **这条测试抓到过一个真 bug**：artifact 一开始没记录 ``seed``，
    于是重建 ivf 时只能自己猜一个（``cli.query`` 传的是 0）——
    分簇不同 → 召回率从 0.6806 变成 0.7917。不报错，只是数字对不上。
    所以下面第一句就是断言「产物里必须带种子」。
    """
    spec_id = result.selected
    artifact = json.loads(
        (out_dir / "artifacts" / f"{spec_id}.json").read_text(encoding="utf-8")
    )
    assert artifact["seed"] == int(cfg["seed"]), "产物必须带种子，否则重建的索引不一样"

    emb = artifact["embedding"]
    model = EmbeddingModel(
        name=emb["name"],
        dim=int(emb["dim"]),
        vocab=list(emb["vocab"]),
        index={t: i for i, t in enumerate(emb["vocab"])},
        idf={k: float(v) for k, v in (emb["idf"] or {}).items()},
        avg_len=float(emb["avg_len"]),
        params=dict(emb["params"]),
        bucket_idf=[float(v) for v in (emb["bucket_idf"] or [])],
    )
    index = build_index(
        artifact["index"]["name"],
        artifact["corpus"]["vectors"],
        artifact["index"]["distance"],
        dict(artifact["index"]["params"]),
        int(artifact["seed"]),
    )
    outcome = evaluate(
        index,
        model,
        bundle.corpus.queries,
        chunk_doc_ids=artifact["corpus"]["doc_ids"],
        k=int(artifact["k"]),
        fetch_k=int(artifact["fetch_k"]),
        latency_repeats=1,
    )
    stored = result.find(spec_id).metrics
    assert outcome.mean_recall == pytest.approx(stored.recall_at_k, abs=1e-6)
    assert outcome.mean_ndcg == pytest.approx(stored.ndcg_at_k, abs=1e-6)
    assert outcome.mrr == pytest.approx(stored.mrr, abs=1e-6)
    # 成本模型必须逐位一致 —— 它不依赖浮点
    assert outcome.mean_distance_computations == pytest.approx(
        stored.n_distance_computations, abs=1e-9
    )


def test_artifact_seed_matters_for_ivf(result, out_dir, cfg) -> None:
    """反证：换一个种子，ivf 的结果**会**变。

    这条断言存在的理由：如果种子其实不影响结果，那上面那条「必须带种子」
    就只是一条形式主义的要求。实测它影响很大 —— 所以必须带。
    """
    artifact = json.loads(
        (out_dir / "artifacts" / f"{result.selected}.json").read_text(encoding="utf-8")
    )
    assert artifact["index"]["name"] == "ivf", "选中的是 ivf，这条反证才成立"
    with_seed = build_index(
        "ivf",
        artifact["corpus"]["vectors"],
        artifact["index"]["distance"],
        dict(artifact["index"]["params"]),
        int(artifact["seed"]),
    )
    with_other_seed = build_index(
        "ivf",
        artifact["corpus"]["vectors"],
        artifact["index"]["distance"],
        dict(artifact["index"]["params"]),
        int(artifact["seed"]) + 1,
    )
    assert with_seed.cells != with_other_seed.cells


def test_artifact_records_the_same_spec(result, out_dir) -> None:
    artifact = json.loads(
        (out_dir / "artifacts" / f"{result.selected}.json").read_text(encoding="utf-8")
    )
    assert artifact["spec"]["spec_id"] == result.selected
    assert artifact["corpus"]["doc_ids"] and artifact["corpus"]["vectors"]
    assert len(artifact["corpus"]["chunk_ids"]) == len(artifact["corpus"]["doc_ids"])
    assert len(artifact["corpus"]["vectors"]) == len(artifact["corpus"]["doc_ids"])


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def test_report_contains_all_ten_sections(result) -> None:
    text = report.render_markdown(result)
    for section in (
        "## 一、语料画像",
        "## 二、两个空间",
        "## 三、消融矩阵",
        "## 四、单变量消融",
        "## 五、空间归因",
        "## 六、表示空间的上限",
        "## 七、Pareto 前沿",
        "## 八、达额成本",
        "## 九、预算敏感性",
        "## 十、血缘与版本",
    ):
        assert section in text, f"报告缺少章节：{section}"


def test_report_marks_timing_frontiers(result, cfg) -> None:
    """含实测轴的副前沿必须被标注为「不可逐位复现」。"""
    text = report.render_markdown(result)
    assert "⚠️ 含实测轴" in text
    assert "不可逐位复现" in text
    for pair in compare.frontier_pairs(cfg):
        key = compare.frontier_key(pair)
        assert key in text
        if compare.has_timing_axis(pair):
            assert f"### {key}　⚠️ 含实测轴" in text
        else:
            assert f"### {key}　（可复现）" in text


def test_report_warns_that_attribution_is_local(result) -> None:
    """极差是**局部量** —— 换参考点排序可能变，报告必须说清这一点。"""
    text = report.render_markdown(result)
    assert "极差是局部量" in text


def test_frontier_header_never_has_duplicate_columns(cfg) -> None:
    """任意两个**不同**指标的组合都不得出现重复列名。

    TokenizerFactory 项目在这里踩过坑：无条件加一列「实际成本」，
    于是当目标轴恰好叫同一个名字时表头出现两个同名列 —— 而且只在真实配置上暴露。
    本项目改为「只在标签里没有它时才补」，并由这条测试守着。

    两个轴相同的组合由配置校验直接拒绝（见下面那条测试），所以这里只遍历不同的对。
    """
    from embeddingfactory.metrics import METRIC_DIRECTIONS

    names = sorted(METRIC_DIRECTIONS)
    for a in names:
        for b in names:
            if a == b:
                continue
            header = report.frontier_header([a, b])
            assert len(set(header)) == len(header), f"{a}+{b} 表头重复：{header}"
    # 主目标对（成本轴恰好叫「距离计算次数」）是最容易撞名的那一个
    main = report.frontier_header(list(cfg["pareto"]["objectives"]))
    assert len(set(main)) == len(main)
    assert main.count("距离计算次数") == 1


def test_frontier_table_width_matches_header(result, cfg) -> None:
    for pair in compare.frontier_pairs(cfg):
        key = compare.frontier_key(pair)
        rows = report.frontier_rows(result, key)
        width = len(report.frontier_header(pair))
        for row in rows:
            assert len(row) == width, f"{key} 的行宽 {len(row)} != 表头 {width}"


def test_matrix_table_width_matches_header(result) -> None:
    rows = report.matrix_rows(result)
    assert len(rows) == 60
    for row in rows:
        assert len(row) == len(report.MATRIX_HEADER)


def test_explain_mentions_spaces_and_metrics(result) -> None:
    text = report.explain(result, result.selected)
    assert "【它是什么】" in text
    assert "表示空间" in text and "检索空间" in text
    assert "【逐查询样本】" in text
    assert "【索引诊断】" in text


def test_explain_handles_unknown_spec(result) -> None:
    text = report.explain(result, "does-not-exist")
    assert "未知 spec_id" in text


def test_explain_for_a_spec_on_no_frontier(result) -> None:
    """不在**任何**前沿上的配置要说明「被谁支配」。

    注意不能随便挑一个「不在主前沿上」的配置：它可能仍然在
    ``index_bytes`` / ``build_seconds`` 这些副前沿上。要挑的是
    所有前沿都不含它的那个。
    """
    on_some = set().union(*(set(members) for members in result.frontiers.values()))
    off = [r for r in result.runs if r.spec_id not in on_some]
    assert off, "本语料实测存在任何前沿都进不去的配置"
    text = report.explain(result, off[0].spec_id)
    assert "不在任何前沿上" in text
    assert "支配" in text


def test_explain_mentions_frontier_rank_for_a_frontier_spec(result, cfg) -> None:
    key = compare.frontier_key(cfg["pareto"]["objectives"])
    member = result.frontiers[key][0]
    text = report.explain(result, member)
    assert "【它在哪条前沿上】" in text
    assert key in text
    assert "第 1/" in text


# ----------------------------------------------------------------------
# 可复现性
# ----------------------------------------------------------------------
def test_run_is_deterministic_except_timings(result, rerun) -> None:
    """连跑两次：所有**可复现**字段必须逐位相同。

    这是本项目「可复现的量化结果」这条验收标准的最终检查。

    **含实测轴的副前沿被刻意排除在外** —— 那正是本项目要证明的事：
    延迟与建索引耗时是实测值，按它们排序的前沿会在两次运行之间换成员
    （``build_seconds`` 前沿这次就从 4 个点变成 5 个点）。主前沿因此
    改用完全由索引结构决定的「距离计算次数」当成本轴。
    把这条对比写在测试里，是为了让「为什么不用延迟」这个决定一直有证据。
    """
    deterministic = set(result.counters["deterministic_frontiers"])
    assert deterministic, "至少得有一条可复现的前沿"

    assert result.selected == rerun.selected
    for key in deterministic:
        assert result.frontiers[key] == rerun.frontiers[key], f"{key} 不可复现"
        assert result.counters["hypervolume"][key] == rerun.counters["hypervolume"][key]
        assert result.counters["reference_point"][key] == rerun.counters["reference_point"][key]
        assert result.counters["frontier_duplicates"].get(key) == (
            rerun.counters["frontier_duplicates"].get(key)
        )
    assert result.version.digest == rerun.version.digest
    assert result.version.version_id == rerun.version.version_id

    timing = {"query_latency_ms", "build_seconds"}
    for a, b in zip(result.matrix, rerun.matrix):
        assert a["spec_id"] == b["spec_id"]
        for field, value in a.items():
            if field in timing:
                continue
            assert value == b[field], f"{a['spec_id']} 的 {field} 两次运行不同"


def test_timing_frontiers_are_the_only_unstable_ones(result, rerun) -> None:
    """把「不可复现」这件事本身也测出来。

    含实测轴的副前沿不保证稳定 —— 这条断言不要求它们一定不同
    （不同才正常，相同也有可能），只要求**它们与可复现前沿不重叠**，
    这样「哪些结论可以引用」才是一个确定的集合。
    """
    deterministic = set(result.counters["deterministic_frontiers"])
    timing = set(result.counters["timing_frontiers"])
    assert deterministic & timing == set()
    assert deterministic | timing == set(result.frontiers)
    assert deterministic == {
        compare.frontier_key(p)
        for p in compare.frontier_pairs(result.cfg)
        if not compare.has_timing_axis(p)
    }
    for key in timing:
        # 副前沿在报告里必须被标注，否则读者会拿它当结论
        assert compare.has_timing_axis(key.split("+"))


def test_ablations_are_deterministic(result, rerun) -> None:
    timing = {"query_latency_ms", "d_latency_ms"}

    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k not in timing}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    assert strip(result.ablations) == strip(rerun.ablations)


def test_version_is_content_addressed(result, rerun) -> None:
    """digest 由**入选配置**决定：只是重跑则不变。"""
    assert result.version.digest == rerun.version.digest
    assert result.version.version_id.startswith(f"{result.cfg['version_tag']}-")
    assert result.version.counts["specs"] == 60
    assert result.version.counts["docs"] == result.corpus.n_docs
    # 换个入选配置就该换版本 —— 但「入选配置」是选型规则的产物，
    # 所以这里只验证 digest 确实是从 spec 内容算出来的
    from embeddingfactory.schema import sha256_text

    run = result.find(result.selected)
    assert result.version.digest == sha256_text(
        json.dumps(
            {
                "spec": run.spec.to_dict(),
                "recall": round(run.metrics.recall_at_k, 6),
                "ndcg": round(run.metrics.ndcg_at_k, 6),
                "n_chunks": run.metrics.n_chunks,
            },
            sort_keys=True,
        )
    )


# ----------------------------------------------------------------------
# 配置校验：把错误挡在运行之前
# ----------------------------------------------------------------------
def test_validate_rejects_identical_metrics_in_a_pair(cfg) -> None:
    """同一个指标写两遍不会报错，只会让前沿表出现两个同名列。"""
    import copy

    from embeddingfactory.config import validate

    bad = copy.deepcopy(cfg)
    bad["pareto"]["objectives"] = ["build_seconds", "build_seconds"]
    with pytest.raises(ValueError, match="不能是同一个指标"):
        validate(bad)


def test_validate_rejects_reference_outside_design_space(cfg) -> None:
    """参考配置必须在设计空间里 —— 否则消融表没有基准行，
    而报告会安静地少一整节。"""
    import copy

    from embeddingfactory.config import validate

    bad = copy.deepcopy(cfg)
    bad["ablation"]["reference"] = "tfidf-sent-flat-cos"  # 缩写不是合法 spec_id
    with pytest.raises(ValueError, match="不在设计空间里"):
        validate(bad)


def test_validate_rejects_mismatched_embedding_dims(cfg) -> None:
    """三个模型的 dim 必须一致，否则「换 embedding」混进了「换维度」。"""
    import copy

    from embeddingfactory.config import validate

    bad = copy.deepcopy(cfg)
    bad["representation"]["embedding_params"]["hashing"]["dim"] = 256
    with pytest.raises(ValueError, match="dim 不一致"):
        validate(bad)


def test_validate_rejects_k_larger_than_corpus(cfg) -> None:
    """k 大于文档数会让 recall@k 恒等于「相关文档全召回」，度量失去意义。"""
    import copy

    from embeddingfactory.config import validate

    bad = copy.deepcopy(cfg)
    bad["retrieval"]["k"] = 999
    with pytest.raises(ValueError, match="大于文档总数"):
        validate(bad)
    # 反过来，好配置不该抛
    validate(copy.deepcopy(cfg))


def test_defaults_alone_are_valid_and_consistent(cfg) -> None:
    """``DEFAULTS`` 必须自洽：默认参考配置得真的在设计空间里。

    踩过的坑：``DEFAULTS`` 里写了缩写 ``tfidf-sent-flat-cos``，而 chunking
    的全名是 ``sentence`` —— 于是任何「不覆盖这一项」的配置都会在 validate
    阶段报「不在设计空间里」，但真正的原因只是少写了三个字母。
    """
    from embeddingfactory.config import DEFAULTS, expand_specs

    ids = {s.spec_id for s in expand_specs(DEFAULTS)}
    assert DEFAULTS["ablation"]["reference"] in ids
    # configs/default.json 覆盖的项必须与 DEFAULTS 保持一致
    assert cfg["ablation"]["reference"] == DEFAULTS["ablation"]["reference"]
    assert cfg["metrics"]["latency_repeats"] == DEFAULTS["metrics"]["latency_repeats"]


# ----------------------------------------------------------------------
# CLI：只读子命令不改产物
# ----------------------------------------------------------------------
_READ_ONLY_COMMANDS = [
    ["compare"],
    ["frontier"],
    ["frontier", "--pair", "query_latency_ms+recall_at_k"],
    ["ablation"],
    ["spaces"],
    ["list"],
    ["dag"],
    ["dag", "--dot"],
    ["dag", "--dot", "--selected-only"],
    ["dag", "--limit", "12"],
]


@pytest.mark.parametrize("argv", _READ_ONLY_COMMANDS, ids=lambda a: " ".join(a))
def test_read_only_commands_do_not_touch_outputs(argv, out_dir, capsys) -> None:
    """观察者会改状态是最容易把人绕晕的一类设计 —— 所以只读子命令一个字节都不许写。"""
    before = _snapshot(out_dir)
    code = cli_main([*argv, "--out", str(out_dir)])
    capsys.readouterr()
    assert code == 0
    assert _snapshot(out_dir) == before, f"{argv} 修改了产物"


def test_read_only_explain_and_query_do_not_touch_outputs(out_dir, result, capsys) -> None:
    before = _snapshot(out_dir)
    assert cli_main(["explain", result.selected, "--out", str(out_dir)]) == 0
    assert cli_main(["query", result.selected, "向量", "检索", "--out", str(out_dir)]) == 0
    captured = capsys.readouterr()
    assert "top-k 文档" in captured.out
    assert _snapshot(out_dir) == before


def test_unknown_spec_id_returns_one_with_suggestions(out_dir, capsys) -> None:
    assert cli_main(["explain", "nope", "--out", str(out_dir)]) == 1
    err = capsys.readouterr().err
    assert "未知 spec_id" in err
    assert "相近的可选项" in err


def test_missing_output_dir_returns_two(tmp_path, capsys) -> None:
    """``run`` 之前跑只读命令要给出可执行的提示，而不是抛栈。"""
    code = cli_main(["compare", "--out", str(tmp_path / "nope")])
    assert code == 2
    assert "先跑一次 run" in capsys.readouterr().err


def test_unknown_frontier_pair_returns_one(out_dir, capsys) -> None:
    assert cli_main(["frontier", "--pair", "nope+recall_at_k", "--out", str(out_dir)]) == 1
    assert "未知目标对" in capsys.readouterr().err


def test_query_command_actually_retrieves(out_dir, result, capsys) -> None:
    """``query`` 是「导出的产物真的能用」的可执行检查 ——
    它会载入 artifact、嵌入新文本、真的跑一次搜索。"""
    assert cli_main(["query", result.selected, "向量", "检索", "--out", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "距离计算次数：" in out
    assert "命中的块（原始 chunk）：" in out


def test_main_py_entry_point_works(out_dir) -> None:
    """零安装入口：不装包也能跑（核心链路零第三方依赖这条约束的最终检查）。"""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "list", "--out", str(out_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "★" in proc.stdout
    assert "共 60 个配置" in proc.stdout


def test_run_command_writes_a_fresh_output_dir(cfg, tmp_path) -> None:
    """``run`` 是唯一带 ``--out`` 的写命令。"""
    out = tmp_path / "fresh"
    result, bundle = run_pipeline(cfg)
    written = write_outputs(result, bundle, out)
    assert "matrix.csv" in written
    assert (out / "lineage.json").exists()
    # 再跑一次 report 渲染（CLI 的 run 会做这件事）
    (out / "report.md").write_text(report.render_markdown(result), encoding="utf-8")
    assert (out / "report.md").exists()
    assert (out / "artifacts" / f"{result.selected}.json").exists()
