import os
import sys
from pathlib import Path

# Ensure project root and backend directory are in sys.path
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

for path_str in [str(ROOT_DIR), str(BACKEND_DIR)]:
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Ensure cloud environment flags are set
os.environ["ALLOW_NONLOCAL_HOSTS"] = "true"
os.environ["ALLOW_NONLOCAL_ORIGINS"] = "true"

from backend.main import app
