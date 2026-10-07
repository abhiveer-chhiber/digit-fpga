"""Check integer inference against an independent scalar reference."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.fixed_point.inference import fixed_forward, quantize, rounded_shift


def clamp(value, bits):
    low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return min(high, max(low, value))


def encode(value, bits, frac):
    scaled = float(value) * 2 ** frac
    magnitude = math.floor(abs(scaled) + 0.5)
    integer = -magnitude if scaled < 0 else magnitude
    return clamp(integer, bits)


def scalar_forward(inputs, weights, biases, bits, frac, acc_bits):
    scale = 2 ** frac
    current = [
        [encode(value, bits, frac) for value in row]
        for row in inputs
    ]
    layers = [np.array(current, dtype=np.int64)]

    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        next_layer = []
        for row in current:
            outputs = []
            for column in range(weight.shape[1]):
                total = 0
                for index, value in enumerate(row):
                    product = value * encode(weight[index, column], bits, frac)
                    total = clamp(total + product, acc_bits)
                total = clamp(
                    total + encode(bias[column], acc_bits, 2 * frac),
                    acc_bits,
                )
                if layer < len(weights) - 1:
                    total = max(0, total)

                magnitude, remainder = divmod(abs(total), scale)
                if 2 * remainder >= scale:
                    magnitude += 1
                rounded = -magnitude if total < 0 else magnitude
                outputs.append(clamp(rounded, bits))
            next_layer.append(outputs)
        current = next_layer
        layers.append(np.array(current, dtype=np.int64))

    return layers


def main():
    values, clips = quantize(
        [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25], 8, 1
    )
    np.testing.assert_array_equal(values, [-3, -2, -1, 1, 2, 3])
    assert clips == 0
    np.testing.assert_array_equal(
        rounded_shift(np.array([-5, -3, -1, 1, 3, 5]), 1),
        [-3, -2, -1, 1, 2, 3],
    )

    rng = np.random.default_rng(73)
    cases = 0
    saturation_cases = 0

    for bits, frac in [(4, 0), (8, 2), (10, 4), (12, 6), (16, 10)]:
        for acc_bits in (8, 16, 32):
            for _ in range(10):
                inputs = rng.normal(0, 12, (3, 5))
                weights = [
                    rng.normal(0, 2, (5, 4)),
                    rng.normal(0, 2, (4, 3)),
                ]
                biases = [rng.normal(0, 3, 4), rng.normal(0, 3, 3)]
                actual = fixed_forward(
                    inputs, weights, biases, bits=bits,
                    fractional_bits=frac, accumulator_bits=acc_bits,
                )
                expected = scalar_forward(
                    inputs, weights, biases, bits, frac, acc_bits
                )
                for layer, (a, b) in enumerate(zip(
                    actual["activations"], expected
                )):
                    np.testing.assert_array_equal(
                        a, b,
                        err_msg=f"bits={bits}, frac={frac}, "
                                f"accumulator={acc_bits}, layer={layer}",
                    )
                np.testing.assert_array_equal(
                    actual["predictions"], expected[-1].argmax(axis=1)
                )
                saturation_cases += int(any(
                    row["accumulator_saturations"] > 0
                    for row in actual["layers"]
                ))
                cases += 1

    assert saturation_cases > 0
    print("Halfway rounding checks passed")
    print(f"Scalar reference matched all layers in {cases} generated cases")
    print(f"Cases exercising accumulator saturation: {saturation_cases}")
    print("All integer inference checks passed")


if __name__ == "__main__":
    main()
