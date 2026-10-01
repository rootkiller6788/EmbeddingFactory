"""三个嵌入模型。

本项目最核心的命题是「**表示空间决定召回率的上限**」，而表示空间就是这三个模型。
所以它们必须：真的在语料上拉开差距（不是三个玩具）、彼此**只差一个变量**、
而且逐位可复现。

三个必须钉死的口径（改一个都会让消融表的结论失去归因能力）：
1. 三个模型 dim 相同 —— 否则「换 embedding」这一行混进了「换向量维度」；
2. 向量**不做 L2 归一化** —— 归一化是距离的事，在这里做会让 cos 与 l2 结果一样；
3. 词表截断按 (df 降序, 词升序) —— 按出现顺序会让词表随语料行序变化。
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from embeddingfactory.corpus import tokenize
from embeddingfactory.embed import (
    EMBEDDINGS,
    EmbeddingModel,
    cosine_similarity,
    fit_embedding,
)

SRC = Path(__file__).resolve().parents[1] / "src"

TEXTS = [
    "向量检索 用 embedding 表示文本",
    "近似最近邻 索引 hnsw ivf 对比",
    "召回率 与 延迟 的 权衡",
    "分块 策略 影响 召回 上限",
]


def _norm(vec) -> float:
    return math.sqrt(sum(v * v for v in vec))


# ----------------------------------------------------------------------
# 三个模型只差一个变量
# ----------------------------------------------------------------------
def test_three_models_share_the_same_dim() -> None:
    """dim 不同的话，「换 embedding」这一行就混进了「换向量维度」这个变量。"""
    dims = {
        name: fit_embedding(name, TEXTS, {"dim": 64}, 0).dim
        for name in ("tfidf", "bm25", "hashing")
    }
    assert set(dims.values()) == {64}


def test_tfidf_and_bm25_share_vocab_but_not_weights() -> None:
    """两者同族：共享词表与维度，唯一差别是加权方式。

    这是刻意的 —— 若它们来自不同分词或不同维度，消融表里就说不清
    Δ 到底来自哪里。
    """
    tfidf = fit_embedding("tfidf", TEXTS, {"dim": 64}, 0)
    bm25 = fit_embedding("bm25", TEXTS, {"dim": 64}, 0)
    assert tfidf.vocab == bm25.vocab
    assert tfidf.index == bm25.index
    assert tfidf.idf != bm25.idf, "idf 口径不同，权重必须不同"


def test_hashing_has_no_vocab_but_has_bucket_idf() -> None:
    """hashing 的「词表」是哈希空间本身；但**必须**有 IDF 加权。

    第一版只做有符号计数，结果它在中文查询上全军覆没 —— 高频 n-gram
    盖过了真正的关键词，而且「换 embedding」这一行里同时混进了
    「有没有 IDF」这个变量，结论无法归因。
    """
    model = fit_embedding("hashing", TEXTS, {"dim": 128, "ngram": 3}, 0)
    assert model.vocab == []
    assert model.index == {}
    assert len(model.bucket_idf) == 128
    assert model.meta["idf_weighted"] is True
    assert model.meta["buckets_used"] > 0
    assert model.meta["n_ngrams"] > 0


def test_hashing_bucket_idf_is_not_trivial() -> None:
    """IDF 必须真的被算出来（而不是全 1 的占位数组）。"""
    model = fit_embedding("hashing", TEXTS, {"dim": 128}, 0)
    used = [v for v in model.bucket_idf if v > 0]
    assert used
    assert len(set(round(v, 9) for v in used)) > 1, "所有桶权重相同 = IDF 没生效"


def test_hashing_meta_reports_bucket_fill() -> None:
    """冲突太严重时向量会退化，报告里要能看见这个数。"""
    model = fit_embedding("hashing", TEXTS, {"dim": 128}, 0)
    assert 0.0 < model.meta["bucket_fill"] <= 1.0
    assert model.meta["bucket_fill"] == pytest.approx(
        model.meta["buckets_used"] / 128, abs=1e-4
    )


# ----------------------------------------------------------------------
# 确定性
# ----------------------------------------------------------------------
def test_fit_is_deterministic() -> None:
    for name in ("tfidf", "bm25", "hashing"):
        a = fit_embedding(name, TEXTS, {"dim": 64}, 7)
        b = fit_embedding(name, TEXTS, {"dim": 64}, 7)
        assert a.matrix(TEXTS) == b.matrix(TEXTS), f"{name} 的向量不可复现"
        # seed 不同也不该变：三个模型都是确定性的，不消耗随机性
        c = fit_embedding(name, TEXTS, {"dim": 64}, 999)
        assert c.matrix(TEXTS) == a.matrix(TEXTS)


_HASHING_CROSS_PROCESS_SNIPPET = """
import sys
sys.path.insert(0, {src!r})
from embeddingfactory.embed import fit_embedding
texts = ["向量检索 用 embedding 表示文本", "近似最近邻 索引 hnsw ivf 对比"]
m = fit_embedding("hashing", texts, {{"dim": 64, "ngram": 3}}, 0)
vec = m.transform(texts[0])
print("vec", [round(v, 12) for v in vec[:6]])
print("sum", round(sum(vec), 12))
"""


def test_hashing_vectors_are_cross_process_stable() -> None:
    """特征哈希的符号位必须跨进程稳定。

    它用 ``stable_hash64``（sha256）而不是内置 ``hash()`` —— 后者对 str 加盐，
    同一份语料在两次运行里会产生不同向量，召回率整个漂移。
    """
    code = _HASHING_CROSS_PROCESS_SNIPPET.format(src=str(SRC))
    outputs = []
    for seed in ("0", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONHASHSEED=seed),
            check=True,
        )
        outputs.append(out.stdout)
    assert outputs[0] == outputs[1]
    assert "vec" in outputs[0]


# ----------------------------------------------------------------------
# 口径：不做归一化、次线性词频、BM25 饱和
# ----------------------------------------------------------------------
def test_vectors_are_not_l2_normalized() -> None:
    """归一化属于**距离**，不属于嵌入（见 embed.py 模块文档第 1 条）。

    空文本给出全零向量就是最好的反证：一个 L2 归一化过的向量范数恒为 1，
    不可能是零向量。若有人在这里加了归一化，cos 与 l2 会变成同一个东西，
    「换距离」这个轴就废了。
    """
    for name in ("tfidf", "bm25", "hashing"):
        model = fit_embedding(name, TEXTS, {"dim": 64}, 0)
        zero = model.transform("")
        assert len(zero) == 64
        assert all(v == 0.0 for v in zero), f"{name} 对空文本返回了非零向量"
        vec = model.transform(TEXTS[0])
        assert not math.isclose(_norm(vec), 1.0, abs_tol=1e-9), f"{name} 偷偷归一化了"


def test_tfidf_uses_sublinear_tf() -> None:
    """线性 tf 会让「一个词出现 20 次」的块彻底压过「出现 2 次」的块。"""
    texts = ["word word word", "other", "third"]
    model = fit_embedding("tfidf", texts, {"dim": 16}, 0)
    idx = model.index["word"]
    vec = model.transform("word word word word")
    assert vec[idx] == pytest.approx((1.0 + math.log(4.0)) * model.idf["word"])


def test_bm25_tf_saturates() -> None:
    """BM25 的词频是饱和的：tf 翻倍，权重增幅不到一倍。"""
    texts = ["alpha", "beta", "gamma delta"]
    model = fit_embedding("bm25", texts, {"dim": 16}, 0)
    idx = model.index["alpha"]
    once = model.transform("alpha")[idx]
    twice = model.transform("alpha alpha")[idx]
    assert twice > once
    assert twice < 2.0 * once, "BM25 的词频没有饱和，退化成线性 tf 了"


def test_bm25_avg_len_is_mean_token_count() -> None:
    texts = ["a b c", "d e", "f"]
    model = fit_embedding("bm25", texts, {"dim": 16}, 0)
    expected = sum(len(tokenize(t)) for t in texts) / len(texts)
    assert model.avg_len == pytest.approx(expected)


def test_empty_text_gives_zero_vector_for_all_models() -> None:
    for name in ("tfidf", "bm25", "hashing"):
        model = fit_embedding(name, TEXTS, {"dim": 32}, 0)
        assert model.transform("") == [0.0] * 32


def test_matrix_shape_matches_texts_and_dim() -> None:
    model = fit_embedding("tfidf", TEXTS, {"dim": 32}, 0)
    matrix = model.matrix(TEXTS)
    assert len(matrix) == len(TEXTS)
    assert all(len(row) == 32 for row in matrix)


# ----------------------------------------------------------------------
# 词表截断：按 (df 降序, 词升序)
# ----------------------------------------------------------------------
def test_vocab_ranks_higher_df_first() -> None:
    """df 高的词优先 —— 不能按出现顺序。"""
    model = fit_embedding("tfidf", ["zz", "zz", "aa"], {"dim": 8}, 0)
    assert model.vocab[:2] == ["zz", "aa"]


def test_vocab_breaks_df_ties_alphabetically() -> None:
    """df 相同时按词升序 —— 否则词表会随语料行序变化，
    「换个语料结论就变」就会变成一个不可复现的假象。"""
    model = fit_embedding("tfidf", ["bb", "aa", "cc"], {"dim": 8}, 0)
    assert model.vocab == ["aa", "bb", "cc"]
    # 打乱语料行序，词表必须不变
    shuffled = fit_embedding("tfidf", ["cc", "bb", "aa"], {"dim": 8}, 0)
    assert shuffled.vocab == model.vocab


def test_vocab_truncates_to_dim() -> None:
    texts = [f"tok{i}" for i in range(30)]
    model = fit_embedding("tfidf", texts, {"dim": 8}, 0)
    assert len(model.vocab) == 8
    assert len(model.index) == 8
    assert set(model.idf) == set(model.vocab)


def test_vocab_fill_diagnostic() -> None:
    """|vocab| < dim 说明向量尾部全是 0，距离计算里有一半是白算的。"""
    model = fit_embedding("tfidf", ["aa", "bb"], {"dim": 16}, 0)
    assert model.meta["vocab_fill"] == pytest.approx(2 / 16, abs=1e-4)
    assert model.meta["min_df"] == 1


def test_min_df_filters_rare_tokens() -> None:
    model = fit_embedding("tfidf", ["aa", "aa", "bb"], {"dim": 8, "min_df": 2}, 0)
    assert model.vocab == ["aa"]


# ----------------------------------------------------------------------
# 诊断与注册表
# ----------------------------------------------------------------------
def test_coverage_of() -> None:
    model = fit_embedding("tfidf", ["aa bb", "cc"], {"dim": 8}, 0)
    assert model.coverage_of("aa bb") == pytest.approx(1.0)
    assert 0.0 < model.coverage_of("aa unknown") < 1.0
    # hashing 没有词表，覆盖率恒为 1
    hashing = fit_embedding("hashing", TEXTS, {"dim": 32}, 0)
    assert hashing.coverage_of("随便什么文本") == 1.0


def test_cosine_similarity_basics() -> None:
    model = fit_embedding("tfidf", TEXTS, {"dim": 64}, 0)
    vec = model.transform(TEXTS[0])
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)
    assert cosine_similarity(vec, [0.0] * 64) == 0.0
    assert cosine_similarity([0.0] * 64, [0.0] * 64) == 0.0


def test_similar_texts_are_more_similar_than_unrelated() -> None:
    """三个模型都得真的能分辨文本 —— 否则它们只是「看起来像模型」。"""
    for name in ("tfidf", "bm25", "hashing"):
        model = fit_embedding(name, TEXTS, {"dim": 512}, 0)
        near = cosine_similarity(model.transform("向量检索 embedding"), model.transform("向量检索 用 embedding 表示文本"))
        far = cosine_similarity(model.transform("向量检索 embedding"), model.transform("分块 策略 影响 召回 上限"))
        assert near > far, f"{name} 分不出相近与不相近的文本"


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        fit_embedding("word2vec", TEXTS, {"dim": 8}, 0)


def test_dim_too_small_raises() -> None:
    with pytest.raises(ValueError):
        fit_embedding("tfidf", TEXTS, {"dim": 4}, 0)


def test_registry_validate_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        EMBEDDINGS["bm25"].validate({"k1": 0})
    with pytest.raises(ValueError):
        EMBEDDINGS["bm25"].validate({"b": 1.5})
    with pytest.raises(ValueError):
        EMBEDDINGS["hashing"].validate({"ngram": 0})
    with pytest.raises(ValueError):
        EMBEDDINGS["tfidf"].validate({"min_df": 0})
    with pytest.raises(ValueError):
        EMBEDDINGS["tfidf"].validate({"dim": 0})
    # 合法参数不抛
    EMBEDDINGS["bm25"].validate({"dim": 8, "k1": 1.2, "b": 0.75})


def test_registry_names_are_stable() -> None:
    """注册表键名进 spec_id，改了会让所有历史产物的 spec_id 失效。"""
    assert set(EMBEDDINGS) == {"tfidf", "bm25", "hashing"}


def test_model_to_dict_is_serializable() -> None:
    model = fit_embedding("bm25", TEXTS, {"dim": 64}, 0)
    payload = model.to_dict()
    assert payload["name"] == "bm25"
    assert payload["dim"] == 64
    assert payload["vocab_size"] == len(model.vocab)
    assert set(payload["params"]) == {"dim"}


def test_manual_model_can_be_reconstructed() -> None:
    """``cli.query`` 就是靠这条路把 artifact 变成可用模型的 ——
    构造 ``EmbeddingModel`` 时必须能只给词表与 idf 就恢复出同样的向量。
    """
    fitted = fit_embedding("tfidf", TEXTS, {"dim": 64}, 0)
    rebuilt = EmbeddingModel(
        name=fitted.name,
        dim=fitted.dim,
        vocab=list(fitted.vocab),
        index=dict(fitted.index),
        idf=dict(fitted.idf),
        avg_len=fitted.avg_len,
        params=dict(fitted.params),
    )
    assert rebuilt.transform(TEXTS[0]) == fitted.transform(TEXTS[0])
