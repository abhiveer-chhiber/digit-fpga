import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_dataset


def softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def evaluate(inputs, labels, weights, bias):
    scores = inputs @ weights + bias
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(
        np.exp(shifted).sum(axis=1, keepdims=True)
    )
    loss = -log_probabilities[np.arange(len(labels)), labels].mean()
    accuracy = np.mean(scores.argmax(axis=1) == labels)
    return float(loss), float(accuracy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, choices=(8, 16), required=True)
    args = parser.parse_args()

    epochs = 1000
    learning_rate = 0.1
    regularization = 0.001
    seed = 42

    output = ROOT / "experiments/results" / f"baseline-{args.resolution}x{args.resolution}"
    if output.exists():
        raise SystemExit(f"Run directory already exists: {output}")

    train_x, train_y, train_paths = load_dataset(args.resolution, "train")
    val_x, val_y, val_paths = load_dataset(args.resolution, "validation")
    train_x = train_x.astype(np.float64)
    val_x = val_x.astype(np.float64)

    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.01, (train_x.shape[1], 10))
    bias = np.zeros(10)
    targets = np.eye(10)[train_y]

    output.mkdir(parents=True)
    config = {
        "model": "linear softmax classifier",
        "resolution": args.resolution,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "regularization": regularization,
        "seed": seed,
        "split": "writer01-v1",
        "train_paths": train_paths,
        "validation_paths": val_paths,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    history = []
    for epoch in range(epochs + 1):
        if epoch > 0:
            probabilities = softmax(train_x @ weights + bias)
            error = (probabilities - targets) / len(train_y)
            weight_gradient = train_x.T @ error + regularization * weights
            bias_gradient = error.sum(axis=0)
            weights -= learning_rate * weight_gradient
            bias -= learning_rate * bias_gradient

        train_loss, train_accuracy = evaluate(train_x, train_y, weights, bias)
        val_loss, val_accuracy = evaluate(val_x, val_y, weights, bias)
        objective = train_loss + 0.5 * regularization * float(
            np.sum(weights * weights)
        )

        if not np.isfinite([objective, val_loss]).all():
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_objective": objective,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
        })

        if epoch % 50 == 0:
            np.savez(
                output / f"checkpoint-{epoch:04d}.npz",
                weights=weights,
                bias=bias,
            )
            print(
                f"Epoch {epoch:4d} | "
                f"train loss {train_loss:.4f}, accuracy {train_accuracy:.1%} | "
                f"validation loss {val_loss:.4f}, accuracy {val_accuracy:.1%}"
            )

    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    predictions = softmax(val_x @ weights + bias)
    records = [
        {
            "path": path,
            "label": int(label),
            "prediction": int(scores.argmax()),
            "scores": scores.tolist(),
        }
        for path, label, scores in zip(val_paths, val_y, predictions)
    ]
    (output / "validation_predictions.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(f"Saved run: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
