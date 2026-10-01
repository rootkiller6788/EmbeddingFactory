"""零安装入口：不装包也能跑。

    python main.py run --config configs/default.json

把 ``src/`` 加进 ``sys.path`` 后转交 :func:`embeddingfactory.cli.main`，
因此核心链路零第三方依赖这一约束在**开发态**也成立 ——
不需要 ``pip install -e .`` 就能运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embeddingfactory.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
