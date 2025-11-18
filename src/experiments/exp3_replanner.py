"""CLI for Experiment 3: replanner and failure recovery study."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

# Support both package (`python -m src.experiments.exp3_replanner`) and
# direct script execution (`python src/experiments/exp3_replanner.py`).
try:  # pragma: no cover - import guard
    from .experiments import ExperimentSuite
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments import ExperimentSuite

app = typer.Typer(help="Run Experiment 3: Replanner vs no-replanner under injected failures.")


@app.command()
def main(
    dataset: Optional[Path] = typer.Option(
        None,
        help="Path to the input .h5ad dataset used by dataset-dependent tasks.",
    ),
    output_dir: Path = typer.Option(
        Path("output/experiment3"),
        help="Directory where Experiment 3 metrics, tables, and figures will be written.",
    ),
    runs_per_task: int = typer.Option(1, min=1, help="Number of repeated runs per task."),
    seed: int = typer.Option(42, help="Random seed for reproducible ablation settings."),
) -> None:
    suite = ExperimentSuite(
        dataset_path=dataset,
        output_dir=output_dir,
        runs_per_task=runs_per_task,
        seed=seed,
    )
    runs = suite.run_experiment3()
    typer.echo(
        f"Completed {len(runs)} runs for Experiment 3. Results saved to {output_dir / 'experiment3_replanner'}."
    )


if __name__ == "__main__":
    app()

