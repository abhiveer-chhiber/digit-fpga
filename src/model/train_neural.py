import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_dataset
from src.model.neural_network import NeuralNetwork


def evaluate(model, inputs, labels):
    result = model.forward(inputs)
    loss = -result["log_probabilities"][np.arange(len(labels)), labels].mean()
    accuracy = np.mean(result["probabilities"].argmax(axis=1) == labels)
    return float(loss), float(accuracy)


def save_checkpoint(model, directory, epoch):
    arrays = {f"w{i}": weights for i, weights in enumerate(model.weights)}
    arrays.update({f"b{i}": bias for i, bias in enumerate(model.biases)})
    np.savez_compressed(directory / f"checkpoint-{epoch:04d}.npz", **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, choices=(8, 16), default=16)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--name", default="neural-16x16-v1")
    args = parser.parse_args()

    if args.epochs < 1 or any(size < 1 for size in args.hidden):
        parser.error("Epochs and layer sizes must be positive")
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("Use a single directory name for --name")

    output = ROOT / "experiments/results" / args.name
    if output.exists():
        raise SystemExit("Run already exists. Choose a new --name.")

    train_x, train_y, train_paths = load_dataset(args.resolution, "train")
    val_x, val_y, val_paths = load_dataset(args.resolution, "validation")
    dimensions = [args.resolution ** 2, *args.hidden, 10]
    model = NeuralNetwork(dimensions, seed=42)
    rng = np.random.default_rng(42)

    learning_rate = 0.001
    regularization = 0.001
    batch_size = 16
    parameters = model.weights + model.biases
    first_moments = [np.zeros_like(p) for p in parameters]
    second_moments = [np.zeros_like(p) for p in parameters]
    step = 0

    config = {
        "model": "NumPy MLP",
        "dimensions": dimensions,
        "parameters": model.parameter_count,
        "resolution": args.resolution,
        "epochs": args.epochs,
        "seed": 42,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "regularization": regularization,
        "batch_size": batch_size,
        "split": "writer01-v1",
        "train_paths": train_paths,
        "validation_paths": val_paths,
    }
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history = []

    for epoch in range(args.epochs + 1):
        if epoch:
            order = rng.permutation(len(train_y))
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                _, dw, db, _ = model.loss_and_gradients(
                    train_x[indices], train_y[indices], regularization
                )
                step += 1

                for index, gradient in enumerate(dw + db):
                    first_moments[index] *= 0.9
                    first_moments[index] += 0.1 * gradient
                    second_moments[index] *= 0.999
                    second_moments[index] += 0.001 * gradient ** 2

                    first = first_moments[index] / (1 - 0.9 ** step)
                    second = second_moments[index] / (1 - 0.999 ** step)
                    parameters[index] -= (
                        learning_rate * first / (np.sqrt(second) + 1e-8)
                    )

        train_loss, train_accuracy = evaluate(model, train_x, train_y)
        val_loss, val_accuracy = evaluate(model, val_x, val_y)
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
        })

        if epoch % 100 == 0 or epoch == args.epochs:
            save_checkpoint(model, output, epoch)
            print(
                f"Epoch {epoch:4d} | "
                f"train {train_accuracy:.1%}, validation {val_accuracy:.1%} | "
                f"loss {train_loss:.4f} / {val_loss:.4f}"
            )

    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print("Architecture:", " -> ".join(map(str, dimensions)))
    print(f"Parameters: {model.parameter_count:,}")
    print(f"Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
