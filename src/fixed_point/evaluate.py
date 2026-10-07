"""Compare fixed-point inference with a saved floating-point checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_dataset
from src.model.neural_network import NeuralNetwork
from src.fixed_point.inference import fixed_forward

FORMATS = [(16, 10), (12, 6), (10, 4), (8, 2)]


def compare(model, inputs, labels, paths, bits, fractional_bits):
    reference = model.forward(inputs)
    float_predictions = reference["probabilities"].argmax(axis=1)
    result = fixed_forward(
        inputs, model.weights, model.biases,
        bits=bits, fractional_bits=fractional_bits, accumulator_bits=32,
    )
    predictions = result["predictions"]
    scale = 1 << fractional_bits
    logits = result["logits"].astype(np.float64) / scale

    errors = []
    for layer, (integer, floating) in enumerate(zip(
        result["activations"], reference["activations"]
    )):
        difference = np.abs(integer.astype(np.float64) / scale - floating)
        errors.append({
            "layer": layer,
            "mean_absolute_error": float(difference.mean()),
            "maximum_absolute_error": float(difference.max()),
        })

    top_two = np.sort(result["logits"], axis=1)[:, -2:]
    changed = np.flatnonzero(predictions != float_predictions)
    saturation_count = result["input_saturations"] + sum(
        sum(value for key, value in row.items() if key.endswith("_saturations"))
        for row in result["layers"]
    )

    return {
        "samples": len(labels),
        "float_correct": int(np.sum(float_predictions == labels)),
        "fixed_correct": int(np.sum(predictions == labels)),
        "agreement_with_float": int(np.sum(predictions == float_predictions)),
        "output_ties": int(np.sum(top_two[:, 0] == top_two[:, 1])),
        "maximum_logit_error": float(np.abs(
            logits - reference["activations"][-1]
        ).max()),
        "input_saturations": result["input_saturations"],
        "layer_statistics": result["layers"],
        "total_saturation_events": saturation_count,
        "layer_errors": errors,
        "changed_predictions": [
            {
                "path": str(paths[i]),
                "label": int(labels[i]),
                "float_prediction": int(float_predictions[i]),
                "fixed_prediction": int(predictions[i]),
            }
            for i in changed
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="neural-16x16-v1")
    parser.add_argument("--name", default="fixed-point-v1")
    args = parser.parse_args()

    for name in (args.run, args.name):
        if Path(name).name != name or name in (".", ".."):
            parser.error("Use single directory/file names")

    run = ROOT / "experiments/results" / args.run
    output = ROOT / "experiments/results" / f"{args.name}.json"
    if output.exists():
        raise SystemExit("Report already exists. Choose a new --name.")

    config = json.loads((run / "config.json").read_text())
    epoch = config["epochs"]
    checkpoint_path = run / f"checkpoint-{epoch:04d}.npz"
    model = NeuralNetwork(config["dimensions"])

    with np.load(checkpoint_path) as checkpoint:
        for i in range(len(model.weights)):
            weight, bias = checkpoint[f"w{i}"], checkpoint[f"b{i}"]
            if weight.shape != model.weights[i].shape:
                raise ValueError(f"Weight shape mismatch: layer {i}")
            if bias.shape != model.biases[i].shape:
                raise ValueError(f"Bias shape mismatch: layer {i}")
            model.weights[i] = weight.copy()
            model.biases[i] = bias.copy()

    datasets = {
        split: load_dataset(config["resolution"], split)
        for split in ("train", "validation")
    }
    report = {
        "schema": 1,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "dimensions": config["dimensions"],
        "split": config["split"],
        "accumulator_bits": 32,
        "rounding": "nearest, halfway away from zero",
        "overflow": "saturate after each MAC addition and bias addition",
        "mac_order": "ascending input index, bias added last",
        "classification": "argmax of integer logits; first index wins ties",
        "formats": [],
    }

    print("Bits / frac | Train correct | Val correct | Val agreement | Saturations")
    print("------------|---------------|-------------|---------------|------------")
    for bits, fractional_bits in FORMATS:
        results = {
            split: compare(model, *dataset, bits, fractional_bits)
            for split, dataset in datasets.items()
        }
        report["formats"].append({
            "bits": bits,
            "fractional_bits": fractional_bits,
            **results,
        })
        train, val = results["train"], results["validation"]
        saturations = (
            train["total_saturation_events"] + val["total_saturation_events"]
        )
        print(
            f"{bits:2d} / {fractional_bits:2d}     | "
            f"{train['fixed_correct']:2d}/{train['samples']:<8d}   | "
            f"{val['fixed_correct']:2d}/{val['samples']:<6d}   | "
            f"{val['agreement_with_float']:2d}/{val['samples']:<8d}   | "
            f"{saturations}"
        )
        print(
            f"  Validation: max logit error {val['maximum_logit_error']:.6f}; "
            f"output ties {val['output_ties']}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output.relative_to(ROOT)}")
    print("Software simulation only; no FPGA timing or resource measurements.")


if __name__ == "__main__":
    main()
