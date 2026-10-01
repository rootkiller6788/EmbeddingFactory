"""确定性随机数 —— 可复现性的地基。

本项目的验收标准之一是「可复现的量化结果」，所以任何不确定性都必须被钉死。
两个必须绕开的坑（与同工作区其它项目共用同一条纪律）：

1. **内置 ``hash()`` 对 str 加随机盐**（``PYTHONHASHSEED``），跨进程不稳定。
   用它派生种子会让「重跑结果一致」直接失效，因此这里一律用 sha256。
2. **全局 ``random`` 的状态会被任何一次调用推进**，导致结果依赖调用顺序。
   这里按 ``(seed, 用途, 对象 id)`` 派生**独立实例**。

本项目真正需要随机性的地方有三处，**每一处都会直接改变索引结构**，
所以必须可复现 —— 不能「差不多就行」：

- **特征哈希的符号位**：字符 n-gram 落到哪个维度、带正号还是负号，
  决定了向量的方向。用 Python 内置 ``hash`` 会让同一份语料在两次运行里
  产生不同的向量 → 召回率整个漂移。
- **IVF 的 k-means 初始化**（k-means++ 的第一批中心）：中心不同 → 分簇不同 →
  `nprobe` 扫到的候选不同 → 召回率与距离计算次数都变。
- **HNSW/NSW 的邻居选取并列打破**：多个候选邻居到新节点的距离完全相同时，
  取哪 M 个决定了图的连通性。按集合遍历顺序取会让结果随 Python 版本变化。
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


def derive_seed(*parts: Any) -> int:
    """由若干片段派生一个 64bit 整数种子（跨平台、跨进程稳定）。"""
    joined = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(joined.encode("utf-8")).digest()[:8], "big")


def make_rng(*parts: Any) -> random.Random:
    """派生一个独立的 ``Random`` 实例。同样的 ``parts`` 必得同样的序列。"""
    return random.Random(derive_seed(*parts))


def stable_hash64(*parts: Any) -> int:
    """把若干片段映射到 64bit 无符号整数。

    与 :func:`derive_seed` 的区别：那个是「取前 8 字节」，这个是**取全部**
    再折叠，用于特征哈希（需要把任意字符串均匀撒到 D 个桶里）。
    同样不用内置 ``hash()``。
    """
    joined = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
