from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.web.config import settings


def resolve_artifact_dir(
    *,
    input_path: Optional[Path] = None,
    work_dir: Optional[str] = None,
    subdir: Optional[str] = None,
) -> Path:
    """Resolve output directory for tool artifacts.

    Priority:
    1) Explicit work_dir if provided.
    2) runs/{run_id}/artifacts if input_path is under RUNS_DIR.
    3) settings.OUTPUT_DIR fallback.
    """
    if work_dir:
        base = Path(work_dir).expanduser().resolve()
    else:
        base = None
        if input_path:
            runs_root = Path(settings.RUNS_DIR).resolve()
            try:
                if input_path.is_relative_to(runs_root):
                    run_id = input_path.relative_to(runs_root).parts[0]
                    base = runs_root / run_id / "artifacts"
            except ValueError:
                base = None
        if base is None:
            base = Path(settings.OUTPUT_DIR).expanduser().resolve()

    if subdir:
        base = base / subdir

    base.mkdir(parents=True, exist_ok=True)
    return base
