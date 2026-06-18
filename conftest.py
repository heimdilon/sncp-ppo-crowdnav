"""Pytest import bridge for the tests/ + scripts/ reorg.

Several tests import root-level scripts at function scope (e.g. ``import run_post_eval``,
``import select_v18_candidate``). After moving tests into ``tests/`` and scripts into
``scripts/``, these ``sys.path`` entries keep those imports resolving. Training/eval are
unaffected (the notebook trains via ``python -m sncp_ppo.train``).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "scripts", _ROOT / "scripts" / "archive"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
