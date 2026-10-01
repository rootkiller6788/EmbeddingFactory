"""表示空间的核心：三个真实、零依赖、确定性的嵌入模型。

**它们不是玩具。** 三个模型代表三种完全不同的「表示」思路，
而本项目的核心命题正是「表示空间决定召回率的上限」——
所以三者必须在**同一份语料上真的拉开差距**，否则这个命题没法验证。

| 模型 | 思路 | 优势 | 真实弱点 |
| --- | --- | --- | --- |
| ``tfidf`` | 词袋 + 逆文档频率加权 | 精确词匹配极强，工业界基线 | 一个词换个说法就完全匹配不上（词汇鸿沟） |
| ``bm25`` | 词袋 + 词频饱和 + 文档长度归一 | 长文档不被词频刷分，排序更稳 | 与 tfidf 同族，弱点一样 |
| ``hashing`` | **字符 n-gram 特征哈希 + IDF 加权** | 对改写、错字、未登录词鲁棒 | 丢词序、维度冲突带来噪声 |

**刻意让 tfidf 与 bm25 同族**：它们共享词表与向量维度，于是
「换 embedding」这一行消融里，变化只来自加权方式本身。
如果三者来自三个不同维度或不同分词，消融表里就混进了别的变量 ——
这是本工作区反复踩的坑（一次只动一个变量）。

**hashing 也做 IDF 加权**（这条是跑起来才补上的）：第一版只做「有符号计数」，
结果它在中文查询上全军覆没 —— 因为不加 IDF 时，「怎么」「的是」这类高频
n-gram 会盖过真正的关键词。而 tfidf / bm25 都带 IDF，于是「换 embedding」
这一行里同时混进了「有没有 IDF」这个变量，结论无法归因。
补上 IDF 之后三个模型的唯一差别才是**特征空间**（词 vs 字符 n-gram）。

三个必须钉死的口径
------------------

1. **向量一律不做 L2 归一化**。归一化是**距离**的事（``cos`` 会归一化，
   ``l2`` / ``ip`` 不会）。在这里归一化会让「cos 与 l2 结果一样」，
   「换距离」这个轴就废了。
2. **词表截断按 (df 降序, 词升序)**，不按出现顺序。按出现顺序会让
   词表随语料行序变化 —— 而那正是「换个语料结论就变」的经典来源。
3. **特征哈希不用内置 ``hash()``**，用 ``rng.stable_hash64``。
   内置 hash 对 str 加随机盐，同一份语料在两次运行里会产生不同向量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .corpus import tokenize
from .rng import stable_hash64

#: 特征哈希里字符 n-gram 的边界标记。加它才能把「词首/词尾」也编码进去，
#: 否则 ``retrieval`` 与 ``retrieve`` 的前缀 n-gram 会完全重合。
_BOW = "\x02"
_EOW = "\x03"


def _l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _sublinear(count: float) -> float:
    """次线性词频：``1 + log(tf)``。

    线性 tf 会让「一个词出现 20 次」的块彻底压过「出现 2 次」的块，
    而检索里真正有信息量的是「出现过」，不是「出现多少次」。
    """
    return 1.0 + math.log(count) if count > 0 else 0.0


@dataclass
class EmbeddingModel:
    """一个**已拟合**的嵌入模型。

    ``vocab`` 只在词法模型里有意义；``hashing`` 用空词表（它的「词表」
    是哈希空间本身，不需要拟合）。
    """

    name: str
    dim: int
    vocab: list[str] = field(default_factory=list)
    #: token → 词表下标
    index: dict[str, int] = field(default_factory=dict)
    #: token → idf / bm25 的 idf 分量
    idf: dict[str, float] = field(default_factory=dict)
    avg_len: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    #: hashing 专用：哈希桶 → idf 权重（长度 = dim）
    bucket_idf: list[float] = field(default_factory=list)
    #: 诊断信息（词表命中率 / 平均非零维数 …）
    meta: dict[str, Any] = field(default_factory=dict)

    # -- 变换 --------------------------------------------------------
    def transform(self, text: str) -> list[float]:
        """把一段文本变成向量。查询与文档走**同一条路径**（重要）。"""
        if self.name == "hashing":
            return self._transform_hashing(text)
        return self._transform_lexical(text)

    def matrix(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.transform(t) for t in texts]

    # -- 词法模型（tfidf / bm25）-------------------------------------
    def _transform_lexical(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        counts: dict[str, int] = {}
        for tok in tokenize(text):
            if tok in self.index:
                counts[tok] = counts.get(tok, 0) + 1
        if not counts:
            return vec
        n_tokens = sum(counts.values())
        if self.name == "tfidf":
            for tok, tf in counts.items():
                vec[self.index[tok]] = _sublinear(tf) * self.idf[tok]
        elif self.name == "bm25":
            k1 = float(self.params.get("k1", 1.2))
            b = float(self.params.get("b", 0.75))
            norm = 1.0 - b + b * (n_tokens / self.avg_len if self.avg_len else 1.0)
            for tok, tf in counts.items():
                vec[self.index[tok]] = self.idf[tok] * (tf * (k1 + 1.0)) / (tf + k1 * norm)
        else:  # pragma: no cover - 注册表保证不会走到
            raise KeyError(f"未知词法模型 {self.name!r}")
        return vec

    # -- 特征哈希 ----------------------------------------------------
    def _ngrams(self, text: str) -> list[str]:
        n = int(self.params.get("ngram", 3))
        cleaned = "".join(ch for ch in text.lower() if not ch.isspace())
        # 空文本直接给空特征集。**不能**只加边界标记就算了：那样
        # ``""`` 会被编码成一串纯 ``\\x02`` / ``\\x03`` 的 n-gram，
        # 于是「一篇什么都没有的文档」反而有了一个确定的非零向量，
        # 而它与任何同样短的文档都会产生虚假的相似度。
        # 词法模型对空文本给零向量，hashing 也必须一致 —— 否则
        # 「换 embedding」这一行里又混进了一个与特征空间无关的差异。
        if not cleaned:
            return []
        padded = _BOW * (n - 1) + cleaned + _EOW * (n - 1)
        if len(padded) < n:
            return [padded]
        return [padded[i : i + n] for i in range(len(padded) - n + 1)]

    def _transform_hashing(self, text: str) -> list[float]:
        """有符号特征哈希 + IDF 加权。

        两步都要：
        - **符号位**取自哈希的高位。没有符号位的话，维度冲突只会让向量单调
          变大（系统性偏置），不同文本的向量会越来越像；
        - **IDF 加权**让高频 n-gram 不再盖过关键词（见模块文档）。
        """
        vec = [0.0] * self.dim
        raw = [0.0] * self.dim
        for gram in self._ngrams(text):
            h = stable_hash64(self.name, gram)
            idx = h % self.dim
            raw[idx] += 1.0 if (h >> 63) & 1 else -1.0
        idf = self.bucket_idf
        for idx, total in enumerate(raw):
            if total == 0.0:
                continue
            weight = idf[idx] if idf else 1.0
            sign = 1.0 if total > 0 else -1.0
            vec[idx] = sign * weight * _sublinear(abs(total))
        return vec

    # -- 诊断 --------------------------------------------------------
    def coverage_of(self, text: str) -> float:
        """文本里能被词表命中的 token 占比（词法模型专用；hashing 恒为 1）。"""
        if self.name == "hashing":
            return 1.0
        toks = tokenize(text)
        if not toks:
            return 1.0
        return sum(1 for t in toks if t in self.index) / len(toks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "vocab_size": len(self.vocab),
            "avg_len": round(self.avg_len, 4),
            "params": {k: self.params[k] for k in sorted(self.params)},
            "meta": {k: self.meta[k] for k in sorted(self.meta)},
        }

# ----------------------------------------------------------------------
# 拟合
# ----------------------------------------------------------------------
def _top_vocab(
    texts: Sequence[str], dim: int, min_df: int
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """按 (df 降序, 词升序) 取前 ``dim`` 个词。**顺序必须是确定的**。"""
    df: dict[str, int] = {}
    for text in texts:
        for tok in set(tokenize(text)):
            df[tok] = df.get(tok, 0) + 1
    cands = [(tok, d) for tok, d in df.items() if d >= min_df]
    # 排序键里带词本身，保证 df 相同时的顺序与语料顺序无关
    cands.sort(key=lambda kv: (-kv[1], kv[0]))
    vocab = [tok for tok, _ in cands[:dim]]
    return vocab, {tok: i for i, tok in enumerate(vocab)}, df


def fit_embedding(
    name: str, texts: Sequence[str], params: dict[str, Any], seed: int
) -> EmbeddingModel:
    """在一批文本上拟合一个嵌入模型。

    ``seed`` 目前只在 hashing 的极端冲突场景下才可能有影响（本项目不需要），
    但签名里保留它 —— 三个模型统一签名，调用方不必分支。
    """
    if name not in EMBEDDINGS:
        raise KeyError(f"未知嵌入模型 {name!r}；可用：{', '.join(sorted(EMBEDDINGS))}")
    del seed  # 三个模型都是确定性的，不消耗随机性
    p = dict(params or {})
    dim = int(p.get("dim", 256))
    if dim < 8:
        raise ValueError(f"{name}.dim = {dim} 太小（至少 8）")

    if name == "hashing":
        model = EmbeddingModel(name=name, dim=dim, params=p)
        # 桶的 df（有多少篇文本用到这个桶）→ idf。与词法模型用同一套平滑公式，
        # 这样「换 embedding」这一行里唯一的差别才是特征空间。
        bucket_df = [0] * dim
        grams_seen: set[str] = set()
        for text in texts:
            used: set[int] = set()
            for gram in model._ngrams(text):
                grams_seen.add(gram)
                used.add(stable_hash64(name, gram) % dim)
            for idx in used:
                bucket_df[idx] += 1
        n_docs_h = len(texts)
        model.bucket_idf = [
            math.log((1.0 + n_docs_h) / (1.0 + d)) + 1.0 if d else 0.0
            for d in bucket_df
        ]
        # 诊断：维度占用率 —— 冲突太严重的话向量会退化，报告里要能看见
        model.meta = {
            "n_ngrams": len(grams_seen),
            "buckets_used": sum(1 for d in bucket_df if d),
            "bucket_fill": round(sum(1 for d in bucket_df if d) / dim, 4),
            "idf_weighted": True,
        }
        return model

    vocab, index, df = _top_vocab(texts, dim, int(p.get("min_df", 1)))
    if not vocab:
        raise ValueError(f"{name} 的词表为空：语料分词后没有任何 token")

    n_docs = len(texts)
    if name == "tfidf":
        idf = {tok: math.log((1.0 + n_docs) / (1.0 + df[tok])) + 1.0 for tok in vocab}
    elif name == "bm25":
        idf = {
            tok: math.log((n_docs - df[tok] + 0.5) / (df[tok] + 0.5) + 1.0)
            for tok in vocab
        }
    else:  # pragma: no cover
        raise KeyError(f"未知嵌入模型 {name!r}")

    lens = [len(tokenize(t)) for t in texts]
    avg_len = sum(lens) / len(lens) if lens else 1.0

    model = EmbeddingModel(
        name=name,
        dim=dim,
        vocab=vocab,
        index=index,
        idf=idf,
        avg_len=avg_len or 1.0,
        params=p,
    )
    # 诊断：词表占用率 —— 若 |vocab| < dim，说明 dim 开得比实际词量还大，
    # 向量尾部全是 0，距离计算里有一半是白算的。报告里要能看见这个数。
    model.meta = {
        "n_tokens_seen": len(df),
        "vocab_fill": round(len(vocab) / dim, 4),
        "min_df": int(p.get("min_df", 1)),
    }
    return model


# ----------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------
class _Registry:
    """把「拟合函数 + 校验」绑在一起，供 ``EMBEDDINGS[name].validate(...)`` 使用。"""

    def __init__(self, name: str, fitter: Any) -> None:
        self.name = name
        self._fit = fitter

    def validate(self, params: dict[str, Any]) -> None:
        p = params or {}
        dim = int(p.get("dim", 0))
        if dim <= 0:
            raise ValueError(f"embedding_params.{self.name}.dim 必须为正")
        if self.name == "hashing" and int(p.get("ngram", 3)) < 1:
            raise ValueError("hashing.ngram 至少为 1")
        if self.name == "tfidf" and int(p.get("min_df", 1)) < 1:
            raise ValueError("tfidf.min_df 至少为 1")
        if self.name == "bm25":
            if float(p.get("k1", 1.2)) <= 0:
                raise ValueError("bm25.k1 必须为正")
            if not 0.0 <= float(p.get("b", 0.75)) <= 1.0:
                raise ValueError("bm25.b 必须在 [0, 1]")

    def fit(self, texts: Sequence[str], params: dict[str, Any], seed: int) -> EmbeddingModel:
        return self._fit(self.name, texts, params, seed)


EMBEDDINGS: dict[str, Any] = {
    "tfidf": _Registry("tfidf", fit_embedding),
    "bm25": _Registry("bm25", fit_embedding),
    "hashing": _Registry("hashing", fit_embedding),
}


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度（只在测试与报告的解释性计算里用；检索走 index.py 的距离）。"""
    na, nb = _l2_norm(a), _l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
