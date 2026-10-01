"""基础原语：schema / rng / 分词 / 分块。

这一层的测试都**不依赖 demo 语料的具体数字** —— 它们断言的是「口径」，
而口径一旦被悄悄改掉，上层所有结论都会跟着错，所以必须钉死。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from embeddingfactory.corpus import (
    CHUNKERS,
    FixedChunker,
    SentenceChunker,
    build_chunks,
    chunk_profile,
    is_sentence_end,
    tokenize,
)
from embeddingfactory.rng import derive_seed, make_rng, stable_hash64
from embeddingfactory.schema import (
    DESIGN_AXES,
    REPRESENTATION_AXES,
    RETRIEVAL_AXES,
    DocChunk,
    Document,
    RetrievalSpec,
    stable_id,
)

SRC = Path(__file__).resolve().parents[1] / "src"


# ----------------------------------------------------------------------
# schema：两个空间
# ----------------------------------------------------------------------
def test_design_axes_split_into_two_spaces() -> None:
    """表示空间在前、检索空间在后 —— 展开顺序与报告列顺序都靠它。"""
    assert REPRESENTATION_AXES == ("embedding", "chunking")
    assert RETRIEVAL_AXES == ("index", "distance", "search")
    assert DESIGN_AXES == REPRESENTATION_AXES + RETRIEVAL_AXES
    assert DESIGN_AXES[:2] == REPRESENTATION_AXES


def test_spec_id_has_no_exact_tail_for_flat() -> None:
    """flat 没有搜索参数，spec_id 不该多出一个恒为 ``exact`` 的尾巴。

    否则 ``tfidf-sentence-flat-cos`` 会变成 ``tfidf-sentence-flat-cos-exact``，
    而配置里的 ``ablation.reference`` 是按前者写的 —— 两边对不上，
    validate 会报「不在设计空间里」，但真正的原因只是多了个尾巴。
    """
    flat = RetrievalSpec(
        embedding="tfidf", chunking="sentence", index="flat", distance="cos"
    )
    assert flat.spec_id == "tfidf-sentence-flat-cos"
    assert flat.search_tag == "exact"


def test_search_tag_uses_short_abbreviations() -> None:
    """``nprobe`` / ``ef_search`` 必须缩成 ``np`` / ``ef``。

    它们会进 spec_id，spec_id 又进表格与 DAG 节点 id —— 全称会让每一行
    多出十几个字符。
    """
    ivf = RetrievalSpec(
        embedding="bm25",
        chunking="sentence",
        index="ivf",
        distance="cos",
        search={"nprobe": 3},
    )
    assert ivf.search_tag == "np3"
    assert ivf.spec_id == "bm25-sentence-ivf-cos-np3"

    hnsw = RetrievalSpec(
        embedding="tfidf",
        chunking="fixed",
        index="hnsw",
        distance="l2",
        search={"ef_search": 64},
    )
    assert hnsw.search_tag == "ef64"
    assert hnsw.spec_id == "tfidf-fixed-hnsw-l2-ef64"


def test_search_tag_is_sorted_and_stable() -> None:
    """多个搜索参数时按参数名排序拼接 —— 否则同一组参数会有两个 spec_id。"""
    a = RetrievalSpec("e", "c", "i", "d", {"b": 2, "a": 1})
    b = RetrievalSpec("e", "c", "i", "d", {"a": 1, "b": 2})
    assert a.search_tag == b.search_tag == "a1-b2"
    assert a.spec_id == b.spec_id


def test_axis_returns_search_tag_for_search() -> None:
    spec = RetrievalSpec("e", "c", "ivf", "d", {"nprobe": 2})
    assert spec.axis("search") == "np2"
    assert spec.axis("embedding") == "e"
    assert spec.axis("index") == "ivf"


def test_to_dict_groups_fields_by_space() -> None:
    """消融表与空间归因直接读 ``representation`` / ``retrieval`` 两组，
    不在报告层再猜哪个字段属于哪个空间。"""
    spec = RetrievalSpec("tfidf", "sentence", "ivf", "cos", {"nprobe": 1})
    payload = spec.to_dict()
    assert set(payload["representation"]) == set(REPRESENTATION_AXES)
    assert set(payload["retrieval"]) == set(RETRIEVAL_AXES)
    assert payload["representation"] == {"embedding": "tfidf", "chunking": "sentence"}
    assert payload["retrieval"] == {
        "index": "ivf",
        "distance": "cos",
        "search": "np1",
    }
    # search 字典按 key 排序导出，保证 digest 稳定
    assert payload["search"] == {"nprobe": 1}


def test_spec_digest_is_stable_and_sensitive() -> None:
    base = RetrievalSpec("tfidf", "sentence", "flat", "cos")
    same = RetrievalSpec("tfidf", "sentence", "flat", "cos")
    other = RetrievalSpec("tfidf", "sentence", "flat", "l2")
    assert base.digest == same.digest
    assert base.digest != other.digest
    assert len(base.digest) == 64


def test_stable_id_is_not_python_hash() -> None:
    """``stable_id`` 必须跨进程稳定 —— 不能用内置 ``hash()``（它对 str 加盐）。"""
    assert stable_id("a", "b") == stable_id("a", "b")
    assert stable_id("a", "b") != stable_id("b", "a")
    assert len(stable_id("a", "b")) == 12


def test_doc_chunk_to_dict_keeps_chunk_id_and_cut() -> None:
    chunk = DocChunk(chunk_id="d1#2", doc_id="d1", text="abcd", index_in_doc=2, cut=True)
    payload = chunk.to_dict()
    assert payload["chunk_id"] == "d1#2"
    assert payload["index_in_doc"] == 2
    assert payload["cut"] is True
    assert payload["n_chars"] == 4


# ----------------------------------------------------------------------
# rng：可复现性的地基
# ----------------------------------------------------------------------
def test_derive_seed_is_deterministic_and_discriminating() -> None:
    assert derive_seed("embedding", "sentence", 7) == derive_seed("embedding", "sentence", 7)
    assert derive_seed("a", "b") != derive_seed("a", "c")
    # 片段边界必须可分辨，否则 ("ab", "c") 与 ("a", "bc") 会撞成同一个种子
    assert derive_seed("ab", "c") != derive_seed("a", "bc")


def test_make_rng_gives_same_sequence_for_same_parts() -> None:
    a = make_rng("ivf-init", 6, 92)
    b = make_rng("ivf-init", 6, 92)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]
    assert make_rng("ivf-init", 6, 93).random() != make_rng("ivf-init", 6, 92).random()


def test_stable_hash64_range_and_stability() -> None:
    h = stable_hash64("hashing", "abc")
    assert isinstance(h, int)
    assert 0 <= h < 2**64
    assert h == stable_hash64("hashing", "abc")
    assert h != stable_hash64("hashing", "abd")
    # 不同的用途前缀必须落到不同位置，否则「模型名」这个变量就没生效
    assert stable_hash64("hashing", "abc") != stable_hash64("tfidf", "abc")


_CROSS_PROCESS_SNIPPET = """
import sys
sys.path.insert(0, {src!r})
from embeddingfactory.rng import derive_seed, make_rng, stable_hash64
print("seed", derive_seed("embedding", "sentence", 7))
print("hash64", stable_hash64("hashing", "abc"))
print("rng", make_rng("ivf-init", 6, 92).random())
# 内置 hash 是**故意**打出来当对照的：它必须随 PYTHONHASHSEED 变化，
# 否则说明这个测试的环境变量没生效，整个跨进程断言就成了空转。
print("builtin", hash("abc"))
"""


def _run_with_hash_seed(seed: str) -> list[str]:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    code = _CROSS_PROCESS_SNIPPET.format(src=str(SRC))
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def test_seed_derivation_is_cross_process_stable() -> None:
    """换 ``PYTHONHASHSEED`` 重开进程，派生结果必须逐位相同。

    这是本项目「可复现的量化结果」这条验收标准的最小可执行检查：
    一旦有人把 ``stable_hash64`` 换成内置 ``hash()``，这个测试立刻红。
    """
    a = _run_with_hash_seed("0")
    b = _run_with_hash_seed("12345")

    stable_a = [line for line in a if not line.startswith("builtin")]
    stable_b = [line for line in b if not line.startswith("builtin")]
    assert stable_a == stable_b
    assert len(stable_a) == 3

    # 反证：内置 hash 在两个进程里**必须**不同，说明 PYTHONHASHSEED 真的生效了
    builtin_a = next(line for line in a if line.startswith("builtin"))
    builtin_b = next(line for line in b if line.startswith("builtin"))
    assert builtin_a != builtin_b, "PYTHONHASHSEED 没生效，跨进程断言是空转"


# ----------------------------------------------------------------------
# 分词：CJK 用 bigram，不用成串切
# ----------------------------------------------------------------------
def test_tokenize_expands_cjk_into_bigrams() -> None:
    """CJK 成串切会让查询必须逐字相同才匹配得上 —— 检索场景必须用 bigram。"""
    assert tokenize("从字符表出发") == ["从字", "字符", "符表", "表出", "出发"]


def test_tokenize_keeps_single_cjk_char() -> None:
    """长度为 1 的 CJK 段保留单字，否则会被整段丢掉。"""
    assert tokenize("甲") == ["甲"]
    assert tokenize("甲，乙") == ["甲", "乙"]


def test_tokenize_lowercases_latin_and_keeps_numbers() -> None:
    assert tokenize("HNSW ef_search") == ["hnsw", "ef_search"]
    assert tokenize("128 3.5") == ["128", "3.5"]
    assert tokenize("k-means") == ["k-means"]
    assert tokenize("Retrieval") == ["retrieval"]


def test_tokenize_strips_punctuation() -> None:
    assert tokenize("向量，检索。") == ["向量", "检索"]
    assert tokenize("！！！") == []


def test_tokenize_mixes_scripts() -> None:
    assert tokenize("向量 embedding") == ["向量", "embedding"]


# ----------------------------------------------------------------------
# 分块
# ----------------------------------------------------------------------
def test_is_sentence_end() -> None:
    assert is_sentence_end("第一句。")
    assert is_sentence_end("第一句？")
    assert is_sentence_end("a;")
    assert not is_sentence_end("第一句")
    assert not is_sentence_end("   ")


def test_sentence_chunker_merges_short_sentences() -> None:
    """合并这一步不能省：一句话单独成块会让一条答案被切成 5 块。"""
    pieces = SentenceChunker.split("第一句。第二句。", {"max_chars": 50})
    assert pieces == [("第一句。第二句。", False)]


def test_sentence_chunker_respects_max_chars() -> None:
    text = "。".join(["一二三四五六七八九十"] * 4) + "。"
    pieces = SentenceChunker.split(text, {"max_chars": 12})
    assert pieces, "不该切出空结果"
    # 单句本身 11 字，两块合并就超 12，所以每块最多 11 字
    assert all(len(t) <= 12 for t, _ in pieces)
    assert all(not cut for _, cut in pieces), "整句成块不该被标记为切碎"


def test_sentence_chunker_hard_splits_overlong_sentence() -> None:
    """单句超长时硬切，且**除最后一片外**都标记 cut=True。"""
    pieces = SentenceChunker.split("啊" * 25, {"max_chars": 10})
    assert [len(t) for t, _ in pieces] == [10, 10, 5]
    assert [cut for _, cut in pieces] == [True, True, False]


def test_fixed_chunker_marks_mid_sentence_cuts() -> None:
    pieces = FixedChunker.split("abcdefghij", {"size": 4, "overlap": 0})
    assert [t for t, _ in pieces] == ["abcd", "efgh", "ij"]
    # 前两块边界都落在「句子中间」（后面还有内容且不以终止符结尾）
    assert [cut for _, cut in pieces] == [True, True, False]


def test_fixed_chunker_stops_at_sentence_boundary() -> None:
    """窗口正好以终止符结尾时，不算切碎 —— 这条口径决定了切碎率的含义。"""
    pieces = FixedChunker.split("abcd。efgh", {"size": 5, "overlap": 0})
    assert [t for t, _ in pieces] == ["abcd。", "efgh"]
    assert [cut for _, cut in pieces] == [False, False]


def test_fixed_chunker_overlap_advances_window() -> None:
    pieces = FixedChunker.split("abcdefgh", {"size": 4, "overlap": 2})
    assert [t for t, _ in pieces] == ["abcd", "cdef", "efgh"]


@pytest.mark.parametrize(
    "chunker, params",
    [
        (SentenceChunker, {"max_chars": 4}),
        (FixedChunker, {"size": 4}),
        (FixedChunker, {"size": 8, "overlap": 8}),
        (FixedChunker, {"size": 8, "overlap": -1}),
    ],
)
def test_chunkers_reject_bad_params(chunker, params) -> None:
    with pytest.raises(ValueError):
        chunker.validate(params)


def test_fixed_chunker_overlap_must_be_smaller_than_size() -> None:
    """overlap >= size 会让窗口不前进 —— 那是死循环，不是慢。"""
    with pytest.raises(ValueError, match="死循环"):
        FixedChunker.validate({"size": 10, "overlap": 10})


def test_chunk_registry_names_are_stable() -> None:
    """注册表键名进 spec_id，改了会让所有历史产物的 spec_id 失效。"""
    assert set(CHUNKERS) == {"sentence", "fixed"}


def test_build_chunks_ids_are_stable_and_ordered() -> None:
    docs = [
        Document(doc_id="d1", title="t", text="第一句。第二句。第三句。"),
        Document(doc_id="d2", title="t", text="甲乙丙丁。戊己庚辛。"),
    ]
    a = build_chunks(docs, "sentence", {"max_chars": 8})
    b = build_chunks(docs, "sentence", {"max_chars": 8})
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.chunk_id for c in a] == ["d1#0", "d1#1", "d2#0", "d2#1"]
    assert [c.doc_id for c in a] == ["d1", "d1", "d2", "d2"]
    assert [c.index_in_doc for c in a] == [0, 1, 0, 1]


def test_build_chunks_unknown_strategy_raises() -> None:
    with pytest.raises(KeyError):
        build_chunks([], "paragraph", {})


def test_chunk_profile_counts_cut_chunks() -> None:
    chunks = [
        DocChunk("d#0", "d", "abcd", 0, cut=True),
        DocChunk("d#1", "d", "ef", 1, cut=False),
    ]
    assert chunk_profile(chunks) == {"n_chunks": 2, "n_cut": 1, "n_chars": 6}
    assert chunk_profile([]) == {"n_chunks": 0, "n_cut": 0, "n_chars": 0}


# ----------------------------------------------------------------------
# 真语料上的口径（这些数字来自 data/corpus/*.jsonl，改语料就该一起改）
# ----------------------------------------------------------------------
def test_corpus_chunking_profiles(corpus) -> None:
    """sentence 的切碎率必须远低于 fixed —— 这是「边界切碎率」这个口径的意义。

    sentence 按句切，所以它切出来的块几乎都以句号收尾；
    fixed 定长窗口必然频繁切断句子。若两者的切碎率接近，
    说明 sentence 策略的实现退化了（比如没真的按句切）。
    """
    sentence = chunk_profile(corpus.chunks["sentence"])
    fixed = chunk_profile(corpus.chunks["fixed"])
    assert sentence["n_chunks"] > len(corpus.documents), "块数等于文档数 = 分块退化成文档级"
    assert fixed["n_cut"] / fixed["n_chunks"] > 0.5
    assert sentence["n_cut"] / sentence["n_chunks"] < 0.1


def test_corpus_queries_reference_real_documents(corpus) -> None:
    doc_ids = {d.doc_id for d in corpus.documents}
    assert corpus.queries, "查询集不能为空"
    for q in corpus.queries:
        assert q.relevant, f"{q.query_id} 没有 ground truth"
        assert set(q.relevant) <= doc_ids
