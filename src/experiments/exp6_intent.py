"""CLI for Experiment 6: intent recognition accuracy."""

from __future__ import annotations

from pathlib import Path

import typer

# Support both package (`python -m src.experiments.exp6_intent`) and direct
# script execution (`python src/experiments/exp6_intent.py`).
try:  # pragma: no cover - import guard
    from .experiments import ExperimentSuite
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments import ExperimentSuite

app = typer.Typer(help="Run Experiment 6: Intent classification benchmark.")


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("output/experiment6"),
        help="Directory where Experiment 6 metrics, tables, and figures will be written.",
    ),
    seed: int = typer.Option(42, help="Random seed for reproducible ablation settings."),
) -> None:
    suite = ExperimentSuite(
        dataset_path=None,
        output_dir=output_dir,
        runs_per_task=1,
        seed=seed,
    )
    predictions = suite.run_experiment6()
    typer.echo(
        f"Completed {len(predictions)} intent predictions for Experiment 6. Results saved to {output_dir / 'experiment6_intent'}."
    )


if __name__ == "__main__":
    app()

