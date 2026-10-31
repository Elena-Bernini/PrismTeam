"""Make ``import prism`` work when scripts are run directly (``python scripts/NN_x.py``).

If the package was installed with ``pip install -e .`` this is a no-op.

On macOS, torch and faiss each ship their own OpenMP runtime; using both in the
same process (phases 10-12) segfaults. Limiting OpenMP to one thread avoids it.
Must run before torch/faiss are imported; an explicit OMP_NUM_THREADS wins.
"""

import os
import sys
from pathlib import Path

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
