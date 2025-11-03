"""CLI entrypoint for executing the Genomix evaluation suite."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .experiments import ExperimentSuite

app = typer.Typer(help="Run the end-to-end Genomix multi-agent evaluation experiments.")


@app.command()
def main(
    dataset: Optional[Path] = typer.Option(
        None,
        help="Path to the input .h5ad dataset used by dataset-dependent tasks.",
    ),
    output_dir: Path = typer.Option(
        Path("output/experiment_suite"),
        help="Directory where metrics, tables, and figures will be written.",
    ),
    runs_per_task: int = typer.Option(1, min=1, help="Number of repeated runs per task."),
    seed: int = typer.Option(42, help="Random seed for reproducible ablation settings."),
) -> None:
    """Execute all experiments and persist results to the specified output directory."""

    suite = ExperimentSuite(
        dataset_path=dataset,
        output_dir=output_dir,
        runs_per_task=runs_per_task,
        seed=seed,
    )
    result = suite.run()
    typer.echo(
        f"Completed {len(result.all_runs)} runs across all experiments. Results saved to {output_dir}."
    )


if __name__ == "__main__":
    app()
