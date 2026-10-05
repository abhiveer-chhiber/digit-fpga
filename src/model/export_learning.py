import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_dataset
from src.model.train_baseline import softmax


def export_run(resolution):
    directory = (
        ROOT / "experiments/results" / f"baseline-{resolution}x{resolution}"
    )
    config = json.loads((directory / "config.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    inputs, labels, paths = load_dataset(resolution, "validation")

    if paths != config["validation_paths"]:
        raise ValueError("Current validation samples differ from the saved run")

    snapshots = []
    for path in sorted(directory.glob("checkpoint-*.npz")):
        epoch = int(path.stem.split("-")[1])
        with np.load(path, allow_pickle=False) as checkpoint:
            weights = checkpoint["weights"]
            bias = checkpoint["bias"]

        if weights.shape != (resolution * resolution, 10):
            raise ValueError(f"Unexpected weight shape: {path}")
        if bias.shape != (10,):
            raise ValueError(f"Unexpected bias shape: {path}")

        probabilities = softmax(inputs.astype(np.float64) @ weights + bias)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Invalid predictions: {path}")

        snapshots.append({
            "epoch": epoch,
            "probabilities": probabilities.tolist(),
            "predictions": probabilities.argmax(axis=1).tolist(),
        })

    expected_epochs = list(range(0, config["epochs"] + 1, 50))
    if [item["epoch"] for item in snapshots] != expected_epochs:
        raise ValueError("Missing or unexpected checkpoints")

    samples = [
        {
            "path": path,
            "label": int(label),
            "pixels": pixels.reshape(resolution, resolution).tolist(),
        }
        for path, label, pixels in zip(paths, labels, inputs)
    ]

    return {
        "resolution": resolution,
        "config": config,
        "history": history,
        "samples": samples,
        "snapshots": snapshots,
    }


def main():
    runs = [export_run(size) for size in (8, 16)]
    output = ROOT / "experiments/results/learning-view.json"
    output.write_text(json.dumps({"runs": runs}, allow_nan=False) + "\n")
    print(f"Exported {len(runs)} runs")
    for run in runs:
        print(
            f'{run["resolution"]}x{run["resolution"]}: '
            f'{len(run["snapshots"])} checkpoints, '
            f'{len(run["samples"])} validation samples'
        )
    print(f"Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
