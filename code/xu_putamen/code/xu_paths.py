"""Output workspace for the unchanged putamen annotation and validation stages.

Use scripts/prepare_putamen_workspace.py before annotation or verification. The
workspace contains copies, never links into deposited inputs.
"""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mg_paths import OUTPUT, require_writable_output

WORK = require_writable_output(Path(os.environ.get("MG_XU_WORK_DIR") or OUTPUT / "xu_putamen"))
if not (WORK / "workspace_manifest.json").is_file():
    raise FileNotFoundError(
        "Run scripts/prepare_putamen_workspace.py first and set MG_XU_WORK_DIR."
    )
