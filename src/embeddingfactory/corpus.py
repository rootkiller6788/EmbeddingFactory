"""语料层：文档加载、分词、分块。

**分块是「表示空间」的第一个轴**，而它常被当成预处理细节 —— 这是错的。
README 第六节把它单列为 `Representation / Chunking 策略`，指标是
「召回完整度、边界切碎率」。本模块把这两件事都落成可计算的量：

- **召回完整度**：一个块能不能装下一条完整的答案。块越小，单块越精确，
  但一条答案被切散后可能没有任何一块单独进得了 top-k。
- **边界切碎率**（``cut`` 标记）：块边界是不是落在句子中间。
  切碎率高说明块与语义单元错位 —— 它不直接进指标，
  但报告里要能解释「为什么这个分块策略的召回率反而低」。

分词（``tokenize``）
--------------------
词法嵌入（TF-IDF / BM25）的输入是词。这里的分词规则与 TokenizerFactory 的
``unicode`` 预分词**刻意不同**，理由很实在：

- **CJK 用二元组（bigram），不用成串切。** TokenizerFactory 那里要成串切，
  因为 BPE 需要看到「检索」「向量」这样的整词才能把它们合并成 token；
  但检索场景恰好相反 —— 成串切会让「从字符表出发」变成一个 token，
  于是查询必须**逐字相同**才能匹配上，词法检索直接失效。
  CJK bigram 是中文信息检索的经典做法（Lucene 的 CJKBigramFilter 就是它）：
  既保留了局部搭配信息，又不会因为少一个字就完全匹配不上。
- **标点一律剥掉**，但**保留数字与拉丁词**（``HNSW`` / ``ef_search`` / ``128``
  都是检索场景里的关键 token）。
- **大小写折叠到小写**，否则 ``Recall`` 与 ``recall`` 会被当成两个词，词表白白翻倍。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .config import load_jsonl, resolve_corpus_path
from .schema import DocChunk, Document, Query

#: 句子终止符（中英文都算）。切句时**保留**这些字符 ——
#: 丢掉它们会让「块以句号结尾」这个判断失效，而边界切碎率正靠它。
SENTENCE_END = "。！？!?；;"

#: 连续 CJK 码点串。切 token 时按**二元组**展开（见模块文档）。
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_TOKEN_RE = re.compile(
    rf"[{_CJK}]+"          # CJK 串（下面按 bigram 展开）
    r"|[A-Za-z][A-Za-z0-9_+\-]*"  # 拉丁词（含 ef_search / k-means 这类）
    r"|\d+(?:\.\d+)?"      # 数字
)
_CJK_CHAR = re.compile(rf"[{_CJK}]")


def tokenize(text: str) -> list[str]:
    """把一段文本切成词序列。

    CJK 段展开成**二元组**：``"从字符表"`` → ``["从字", "字符", "符表"]``。
    长度为 1 的 CJK 段保留单字（否则会被整段丢掉）。
    """
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        piece = m.group(0)
        if _CJK_CHAR.match(piece[0]):
            if len(piece) == 1:
                out.append(piece)
            else:
                out.extend(piece[i : i + 2] for i in range(len(piece) - 1))
        else:
            out.append(piece.lower())
    return out


def is_sentence_end(chunk_text: str) -> bool:
    """这个块是不是以句子终止符收尾（或本来就是最后一块）。"""
    stripped = chunk_text.rstrip()
    return bool(stripped) and stripped[-1] in SENTENCE_END


# ----------------------------------------------------------------------
# 分块策略
# ----------------------------------------------------------------------
class SentenceChunker:
    """按句切，再把短句合并到 ``max_chars`` 以内。

    合并这一步不能省：一句话往往只有十几个字，单独成块会让「一条答案
    被切成 5 块」，top-k 全被同一篇文档的碎片占满，其他文档挤不进来。
    合并到 ``max_chars`` 既保证块不太碎，又保证块不太长（长块会稀释语义）。
    """

    name = "sentence"

    @staticmethod
    def validate(params: dict[str, Any]) -> None:
        max_chars = int(params.get("max_chars", 120))
        if max_chars < 8:
            raise ValueError(
                f"sentence.max_chars = {max_chars} 太小（至少 8）；"
                "再小就等于按标点切，块会碎到召回率不可用"
            )

    @staticmethod
    def split(text: str, params: dict[str, Any]) -> list[tuple[str, bool]]:
        max_chars = int(params.get("max_chars", 120))
        sentences: list[str] = []
        buf: list[str] = []
        for ch in text:
            buf.append(ch)
            if ch in SENTENCE_END:
                piece = "".join(buf).strip()
                if piece:
                    sentences.append(piece)
                buf = []
        tail = "".join(buf).strip()
        if tail:
            sentences.append(tail)

        out: list[tuple[str, bool]] = []
        cur = ""
        for sentence in sentences:
            # 单句本身就超长：先冲掉累积的，再硬切这一句
            if len(sentence) > max_chars:
                if cur:
                    out.append((cur, False))
                    cur = ""
                for i in range(0, len(sentence), max_chars):
                    piece = sentence[i : i + max_chars]
                    out.append((piece, len(piece) == max_chars and i + max_chars < len(sentence)))
                continue
            if len(cur) + len(sentence) > max_chars and cur:
                out.append((cur, False))
                cur = ""
            cur = cur + sentence if cur else sentence
        if cur:
            out.append((cur, False))
        return [(t, cut) for t, cut in out if t.strip()]


class FixedChunker:
    """定长窗口切块。

    ``cut`` 的定义：窗口**没有以句子终止符收尾，且后面还有内容** ——
    也就是说这个块的边界切在句子中间。它不直接进指标，
    但报告里要能解释「为什么定长切块在中文语料上更差」。
    """

    name = "fixed"

    @staticmethod
    def validate(params: dict[str, Any]) -> None:
        size = int(params.get("size", 80))
        overlap = int(params.get("overlap", 0))
        if size < 8:
            raise ValueError(f"fixed.size = {size} 太小（至少 8）")
        if overlap < 0:
            raise ValueError("fixed.overlap 不能为负")
        if overlap >= size:
            raise ValueError(
                f"fixed.overlap = {overlap} 必须小于 size = {size}；"
                "否则窗口不前进，切块会变成死循环"
            )

    @staticmethod
    def split(text: str, params: dict[str, Any]) -> list[tuple[str, bool]]:
        size = int(params.get("size", 80))
        overlap = int(params.get("overlap", 0))
        step = size - overlap
        out: list[tuple[str, bool]] = []
        for start in range(0, len(text), step):
            piece = text[start : start + size]
            if not piece.strip():
                continue
            has_more = start + size < len(text)
            out.append((piece, has_more and not is_sentence_end(piece)))
            if not has_more:
                break
        return out


#: 分块策略注册表。键名进 spec_id，**不要改**。
CHUNKERS: dict[str, Any] = {
    "sentence": SentenceChunker,
    "fixed": FixedChunker,
}


# ----------------------------------------------------------------------
# 加载
# ----------------------------------------------------------------------
@dataclass
class CorpusBundle:
    """加载好的一整份语料 —— 全部 spec 共用同一份（对象同一性有测试守着）。"""

    documents: list[Document]
    queries: list[Query]
    #: 策略 → 该策略切出来的全部块（顺序与 documents 一致）
    chunks: dict[str, list[DocChunk]]
    name: str

    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.documents]

    def chunk_texts(self, strategy: str) -> list[str]:
        return [c.text for c in self.chunks[strategy]]


def load_documents(cfg: dict[str, Any]) -> list[Document]:
    rows = load_jsonl(resolve_corpus_path(cfg, "documents"))
    docs: list[Document] = []
    seen: set[str] = set()
    for row in rows:
        doc_id = str(row["doc_id"])
        if doc_id in seen:
            raise ValueError(f"文档 id 重复：{doc_id}")
        seen.add(doc_id)
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError(f"文档 {doc_id} 的 text 为空")
        docs.append(
            Document(
                doc_id=doc_id,
                title=str(row.get("title", "")),
                text=text,
                topic=str(row.get("topic", "")),
            )
        )
    return docs


def load_queries(cfg: dict[str, Any]) -> list[Query]:
    rows = load_jsonl(resolve_corpus_path(cfg, "queries"))
    out: list[Query] = []
    seen: set[str] = set()
    for row in rows:
        qid = str(row["query_id"])
        if qid in seen:
            raise ValueError(f"查询 id 重复：{qid}")
        seen.add(qid)
        out.append(
            Query(
                query_id=qid,
                text=str(row.get("text", "")).strip(),
                relevant=[str(x) for x in row.get("relevant", [])],
            )
        )
    return out


def build_chunks(
    documents: Sequence[Document], strategy: str, params: dict[str, Any]
) -> list[DocChunk]:
    """按策略切块。**顺序必须稳定** —— chunk_id 直接由文档顺序与块序号生成。"""
    if strategy not in CHUNKERS:
        raise KeyError(f"未知分块策略 {strategy!r}；可用：{', '.join(sorted(CHUNKERS))}")
    chunker = CHUNKERS[strategy]
    out: list[DocChunk] = []
    for doc in documents:
        pieces = chunker.split(doc.text, params)
        for i, (text, cut) in enumerate(pieces):
            out.append(
                DocChunk(
                    chunk_id=f"{doc.doc_id}#{i}",
                    doc_id=doc.doc_id,
                    text=text,
                    index_in_doc=i,
                    cut=bool(cut),
                )
            )
    return out


def load_corpus(cfg: dict[str, Any]) -> CorpusBundle:
    """加载文档、查询，并为每个分块策略切一遍。

    刻意**在这里就把所有策略的块切好并缓存**：所有 spec 共用同一份
    ``CorpusBundle`` 对象。不缓存的话每个 spec 各自重切一遍，
    「所有配置看的是同一份语料」这个前提会在实现层悄悄失效
    （见 skill 的「扇出-收拢型」一节）。
    """
    documents = load_documents(cfg)
    queries = load_queries(cfg)
    chunks: dict[str, list[DocChunk]] = {}
    for strategy in [str(c) for c in cfg["representation"]["chunkings"]]:
        params = cfg["representation"]["chunking_params"].get(strategy, {})
        chunks[strategy] = build_chunks(documents, strategy, params)
        if not chunks[strategy]:
            raise ValueError(f"分块策略 {strategy!r} 切出 0 个块，无法检索")
    return CorpusBundle(
        documents=documents,
        queries=queries,
        chunks=chunks,
        name=str(cfg["corpus"].get("name", "corpus")),
    )


def chunk_profile(chunks: Iterable[DocChunk]) -> dict[str, int]:
    """分块画像：块数 / 被硬切开的块数 / 字符数。"""
    items = list(chunks)
    return {
        "n_chunks": len(items),
        "n_cut": sum(1 for c in items if c.cut),
        "n_chars": sum(c.n_chars for c in items),
    }
