import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT_DIR.parent

sys.path.append(str(ROOT_DIR / "src"))
sys.path.append(str(REPO_ROOT / "src"))
