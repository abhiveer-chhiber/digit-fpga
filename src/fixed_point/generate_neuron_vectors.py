"""Generate integer test cases for the 12-bit hardware neuron."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.fixed_point.inference import quantize, fixed_forward


def calculate(inputs, weights, bias, relu):
    low, high = -(1 << 31), (1 << 31) - 1
    accumulator = 0
    accumulator_saturated = False

    for value, weight in zip(inputs, weights):
        total = accumulator + int(value) * int(weight)
        accumulator_saturated |= total < low or total > high
        accumulator = min(high, max(low, total))

    total = accumulator + int(bias)
    accumulator_saturated |= total < low or total > high
    accumulator = min(high, max(low, total))

    if relu:
        accumulator = max(0, accumulator)

    magnitude, remainder = divmod(abs(accumulator), 64)
    if remainder >= 32:
        magnitude += 1
    rounded = -magnitude if accumulator < 0 else magnitude
    output_saturated = rounded < -2048 or rounded > 2047
    output = min(2047, max(-2048, rounded))
    return output, int(accumulator_saturated), int(output_saturated)


def main():
    cases = []

    def add(inputs, weights, bias=0, relu=False, category="boundary"):
        inputs = [int(value) for value in inputs]
        weights = [int(value) for value in weights]
        assert len(inputs) == len(weights) and len(inputs) > 0
        assert all(-2048 <= value <= 2047 for value in inputs + weights)
        assert -(1 << 31) <= int(bias) < (1 << 31)
        expected = calculate(inputs, weights, bias, relu)
        cases.append((category, inputs, weights, int(bias), int(relu), expected))

    for bias in (
        0, 31, 32, 33, -31, -32, -33,
        131008, 131039, 131040,
        -131072, -131103, -131104,
        (1 << 31) - 1, -(1 << 31),
    ):
        add([0], [0], bias)
        add([0], [0], bias, relu=True)

    add([1], [1], (1 << 31) - 1)
    add([-1], [1], -(1 << 31))
    add([-2048] * 512, [-2048] * 512)
    add([-2048] * 513, [2047] * 513)
    add([-2048] * 512 + [2047] * 512, [-2048] * 1024)
    add([-2048] * 513, [2047] * 513, relu=True)

    rng = np.random.default_rng(73)
    for index in range(200):
        count = int(rng.integers(1, 65))
        inputs = rng.integers(-2048, 2048, count)
        weights = rng.integers(-2048, 2048, count)
        bias = int(rng.integers(-(1 << 31), 1 << 31)) if index % 4 == 0 else 0
        add(inputs, weights, bias, bool(index % 2), "generated")

    data = json.loads((ROOT / "site_assets/network-view.json").read_text())
    assert data["config"]["dimensions"] == [256, 128, 64, 10]
    snapshot = next(row for row in data["snapshots"] if row["epoch"] == 1000)
    weights = [np.asarray(value) for value in snapshot["weights"]]
    biases = [np.asarray(value) for value in snapshot["biases"]]
    integer_weights = [quantize(value, 12, 6)[0] for value in weights]
    integer_biases = [quantize(value, 32, 12)[0] for value in biases]
    inputs = np.asarray([sample["pixels"] for sample in data["samples"]])
    reference = fixed_forward(
        inputs, weights, biases, bits=12,
        fractional_bits=6, accumulator_bits=32,
    )

    for sample_index in range(len(inputs)):
        current = quantize(inputs[sample_index], 12, 6)[0].tolist()
        for layer, (weight, bias) in enumerate(zip(integer_weights, integer_biases)):
            relu = layer < len(integer_weights) - 1
            outputs = []
            selected = {0, weight.shape[1] // 2, weight.shape[1] - 1}
            for neuron in range(weight.shape[1]):
                column = weight[:, neuron].tolist()
                expected = calculate(current, column, int(bias[neuron]), relu)
                outputs.append(expected[0])
                if neuron in selected:
                    add(current, column, int(bias[neuron]), relu, "model")
            np.testing.assert_array_equal(
                outputs, reference["activations"][layer + 1][sample_index]
            )
            current = outputs

    output = ROOT / "build/sim/neuron-vectors.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        handle.write(f"{len(cases)}\n")
        for category, values, weights, bias, relu, expected in cases:
            result, acc_clip, output_clip = expected
            handle.write(
                f"{len(values)} {bias} {relu} {result} {acc_clip} {output_clip}\n"
            )
            for value, weight in zip(values, weights):
                handle.write(f"{value} {weight}\n")

    for category in ("boundary", "generated", "model"):
        print(f"{category}: {sum(case[0] == category for case in cases)} cases")
    print(f"Saved {len(cases)} cases to {output.relative_to(ROOT)}")
    print("Real model calculations matched the existing integer implementation")


if __name__ == "__main__":
    main()
