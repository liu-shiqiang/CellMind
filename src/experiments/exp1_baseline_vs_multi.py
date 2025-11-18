"""CLI for Experiment 1: baseline single-agent vs multi-agent workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

# Support both package (`python -m src.experiments.exp1_baseline_vs_multi`) and
# direct script execution (`python src/experiments/exp1_baseline_vs_multi.py`).
try:  # pragma: no cover - import guard
    from .experiments import ExperimentSuite
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments import ExperimentSuite

app = typer.Typer(help="Run Experiment 1: Baseline vs multi-agent workflow.")


@app.command()
def main(
    dataset: Optional[Path] = typer.Option(
        None,
        help="Path to the input .h5ad dataset used by dataset-dependent tasks.",
    ),
    output_dir: Path = typer.Option(
        Path("output/experiment1"),
        help="Directory where Experiment 1 metrics, tables, and figures will be written.",
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
    runs = suite.run_experiment1()
    typer.echo(
        f"Completed {len(runs)} runs for Experiment 1. Results saved to {output_dir / 'experiment1_baseline_vs_multi'}."
    )


if __name__ == "__main__":
    app()

