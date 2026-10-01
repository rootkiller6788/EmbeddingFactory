"""pytest 夹具。

**端到端只跑一次**（``scope="session"``）：本项目一个完整 run 要跑 60 个配置、
每个配置建索引 + 跑 24 条查询，几秒钟。如果每个测试都自己跑一遍，
测试套件会慢到没人愿意在改代码后跑它 —— 而一个没人跑的测试套件等于没有。

两个层次的夹具：

- **真语料 + 真流水线**（``cfg`` / ``corpus`` / ``result`` / ``bundle`` / ``out_dir``）：
  给端到端与「跨模块一致性」类断言用。
- **纯内存小样本**：单元测试自己构造（``build_chunks`` 收 ``Document`` 列表、
  ``fit_embedding`` 收字符串列表、``build_index`` 收向量列表），不需要夹具 ——
  它们不该依赖 demo 语料的任何具体数字，否则改一句语料就要改一堆单元测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# ``pyproject.toml`` 里已经配了 ``pythonpath = ["src"]``；这里再插一次是为了
# 让「直接用 python -m pytest / IDE 单独跑某个测试文件」也能工作。
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embeddingfactory import report  # noqa: E402
from embeddingfactory.config import load_config  # noqa: E402
from embeddingfactory.corpus import load_corpus  # noqa: E402
from embeddingfactory.pipeline import run_pipeline, write_outputs  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg() -> dict:
    """默认配置（已过 validate）。"""
    return load_config(ROOT / "configs" / "default.json")


@pytest.fixture(scope="session")
def corpus(cfg: dict):
    """加载好的语料（文档 / 查询 / 每个分块策略的块）。"""
    return load_corpus(cfg)


@pytest.fixture(scope="session")
def run_result(cfg: dict):
    """``(result, bundle)`` —— 整条流水线只跑一次。"""
    return run_pipeline(cfg)


@pytest.fixture(scope="session")
def result(run_result):
    return run_result[0]


@pytest.fixture(scope="session")
def bundle(run_result):
    return run_result[1]


@pytest.fixture(scope="session")
def out_dir(run_result, tmp_path_factory) -> Path:
    """落盘产物目录。

    刻意**复用** session 级的 result / bundle（而不是再跑一次 CLI），
    并把 ``report.md`` 一并写出来 —— 只读子命令与「只读契约」测试都要用它。
    """
    result, bundle = run_result
    out = tmp_path_factory.mktemp("out")
    write_outputs(result, bundle, out)
    (out / "report.md").write_text(report.render_markdown(result), encoding="utf-8")
    return out
