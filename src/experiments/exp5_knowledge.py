"""CLI for Experiment 5: RAG and dataset QA contribution study."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .experiments import ExperimentSuite

app = typer.Typer(help="Run Experiment 5: Knowledge retrieval vs disabled baseline.")


@app.command()
def main(
    dataset: Optional[Path] = typer.Option(
        None,
        help="Path to the input .h5ad dataset used by dataset-dependent tasks.",
    ),
    output_dir: Path = typer.Option(
        Path("output/experiment5"),
        help="Directory where Experiment 5 metrics, tables, and figures will be written.",
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
    runs = suite.run_experiment5()
    typer.echo(
        f"Completed {len(runs)} runs for Experiment 5. Results saved to {output_dir / 'experiment5_knowledge'}."
    )


if __name__ == "__main__":
    app()

