import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_dataset
from src.model.neural_network import NeuralNetwork
from src.model.train_neural import evaluate


def main():
    directory = ROOT / "experiments/results/neural-16x16-v1"
    config = json.loads((directory / "config.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    inputs, labels, paths = load_dataset(config["resolution"], "validation")

    if paths != config["validation_paths"]:
        raise ValueError("Validation split differs from the saved run")

    model = NeuralNetwork(config["dimensions"])
    metrics = {row["epoch"]: row for row in history}
    snapshots = []

    for path in sorted(directory.glob("checkpoint-*.npz")):
        epoch = int(path.stem.split("-")[1])

        with np.load(path, allow_pickle=False) as checkpoint:
            for index in range(len(model.weights)):
                weights = checkpoint[f"w{index}"]
                bias = checkpoint[f"b{index}"]

                if weights.shape != model.weights[index].shape:
                    raise ValueError(f"Unexpected weight shape: {path}")
                if bias.shape != model.biases[index].shape:
                    raise ValueError(f"Unexpected bias shape: {path}")
                if not np.isfinite(weights).all() or not np.isfinite(bias).all():
                    raise ValueError(f"Non-finite parameters: {path}")

                model.weights[index][...] = weights
                model.biases[index][...] = bias

        loss, accuracy = evaluate(model, inputs, labels)
        recorded = metrics[epoch]

        if not np.isclose(loss, recorded["validation_loss"]):
            raise ValueError(f"Loss mismatch at epoch {epoch}")
        if not np.isclose(accuracy, recorded["validation_accuracy"]):
            raise ValueError(f"Accuracy mismatch at epoch {epoch}")

        snapshots.append({
            "epoch": epoch,
            "weights": [weights.tolist() for weights in model.weights],
            "biases": [bias.tolist() for bias in model.biases],
            "probabilities": model.forward(inputs)["probabilities"].tolist(),
        })

    expected = list(range(0, config["epochs"] + 1, 100))
    if expected[-1] != config["epochs"]:
        expected.append(config["epochs"])
    if [snapshot["epoch"] for snapshot in snapshots] != expected:
        raise ValueError("Missing or unexpected checkpoints")

    samples = [
        {
            "path": path,
            "label": int(label),
            "pixels": pixels.tolist(),
        }
        for path, label, pixels in zip(paths, labels, inputs)
    ]

    payload = {
        "schema": 1,
        "data_kind": "recorded custom handwriting",
        "config": config,
        "history": history,
        "samples": samples,
        "snapshots": snapshots,
    }

    output = ROOT / "experiments/results/network-view.json"
    output.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
    )

    print(f"Exported {len(snapshots)} verified checkpoints")
    print(f"Included {len(samples)} validation drawings")
    print(f"Architecture: {config['dimensions']}")
    print(f"Saved {output.relative_to(ROOT)}")
    print(f"File size: {output.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
