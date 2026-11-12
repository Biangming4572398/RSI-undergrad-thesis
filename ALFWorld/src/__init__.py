import sys
from pathlib import Path

# ALFWorld reuses Jericho's JitRL agent and memory and the root jitrl_prior module.
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
