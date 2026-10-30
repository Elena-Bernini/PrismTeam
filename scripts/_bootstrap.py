"""Make ``import prism`` work when scripts are run directly (``python scripts/NN_x.py``).

If the package was installed with ``pip install -e .`` this is a no-op.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
