"""Integer inference with explicit scaling, rounding, and saturation."""

import numpy as np


def limits(bits):
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def saturate(values, bits):
    low, high = limits(bits)
    count = int(np.count_nonzero((values < low) | (values > high)))
    return np.clip(values, low, high).astype(np.int64), count


def quantize(values, bits, fractional_bits):
    """Round to nearest; halfway cases round away from zero."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Values must be finite")
    scaled = values * (1 << fractional_bits)
    if not np.isfinite(scaled).all():
        raise ValueError("Scaling produced non-finite values")
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    return saturate(rounded, bits)


def rounded_shift(values, fractional_bits):
    """Divide signed integers by a power of two with the same rounding rule."""
    if fractional_bits == 0:
        return values.copy()
    magnitude = np.abs(values)
    rounded = (magnitude + (1 << (fractional_bits - 1))) >> fractional_bits
    return np.where(values < 0, -rounded, rounded)


def fixed_forward(inputs, weights, biases, *, bits=16,
                  fractional_bits=10, accumulator_bits=32):
    if not isinstance(bits, int) or not 2 <= bits <= 16:
        raise ValueError("Storage width must be between 2 and 16 bits")
    if not isinstance(fractional_bits, int) or not 0 <= fractional_bits < bits:
        raise ValueError("Fractional bits must be between 0 and bits - 1")
    if not isinstance(accumulator_bits, int) or not 2 <= accumulator_bits <= 32:
        raise ValueError("Accumulator width must be between 2 and 32 bits")
    if len(weights) != len(biases) or not len(weights):
        raise ValueError("Provide matching, nonempty weights and biases")

    inputs = np.asarray(inputs, dtype=np.float64)
    if inputs.ndim != 2 or not len(inputs):
        raise ValueError("Provide a nonempty batch of input rows")

    current, input_clips = quantize(inputs, bits, fractional_bits)
    activations = [current.copy()]
    statistics = []

    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        weight = np.asarray(weight, dtype=np.float64)
        bias = np.asarray(bias, dtype=np.float64)
        if weight.ndim != 2 or weight.shape[0] != current.shape[1]:
            raise ValueError(f"Weight shape mismatch in layer {layer}")
        if weight.shape[1] < 1 or bias.shape != (weight.shape[1],):
            raise ValueError(f"Bias shape mismatch in layer {layer}")

        integer_weights, weight_clips = quantize(weight, bits, fractional_bits)
        # Products carry twice the fractional bits. Biases use that scale.
        integer_biases, bias_clips = quantize(
            bias, accumulator_bits, 2 * fractional_bits
        )
        accumulator = np.zeros(
            (len(current), weight.shape[1]), dtype=np.int64
        )
        accumulator_clips = 0

        # Defined MAC order: input index ascending, then add the bias.
        # Saturate after every addition, matching a finite-width accumulator.
        for index in range(current.shape[1]):
            products = (
                current[:, index:index + 1]
                * integer_weights[index:index + 1, :]
            )
            accumulator, clips = saturate(
                accumulator + products, accumulator_bits
            )
            accumulator_clips += clips

        accumulator, clips = saturate(
            accumulator + integer_biases, accumulator_bits
        )
        accumulator_clips += clips

        if layer < len(weights) - 1:
            accumulator = np.maximum(accumulator, 0)

        current, activation_clips = saturate(
            rounded_shift(accumulator, fractional_bits), bits
        )
        activations.append(current.copy())
        statistics.append({
            "layer": layer + 1,
            "weight_saturations": weight_clips,
            "bias_saturations": bias_clips,
            "accumulator_saturations": accumulator_clips,
            "activation_saturations": activation_clips,
        })

    return {
        "logits": current,
        "predictions": current.argmax(axis=1),
        "activations": activations,
        "input_saturations": input_clips,
        "layers": statistics,
        "bits": bits,
        "fractional_bits": fractional_bits,
        "accumulator_bits": accumulator_bits,
    }
