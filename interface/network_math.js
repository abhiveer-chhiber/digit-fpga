export function forward(pixels, snapshot) {
    const activations = [Array.from(pixels)];
    const preactivations = [];

    snapshot.weights.forEach((weights, layer) => {
        const input = activations[layer];
        const bias = snapshot.biases[layer];

        if (weights.length !== input.length ||
            weights.some(row => row.length !== bias.length)) {
            throw new Error("Network dimensions do not match");
        }

        const values = bias.map((value, neuron) => {
            let sum = value;
            for (let i = 0; i < input.length; i++) {
                sum += input[i] * weights[i][neuron];
            }
            return sum;
        });

        preactivations.push(values);
        activations.push(
            layer < snapshot.weights.length - 1
                ? values.map(value => Math.max(0, value))
                : values
        );
    });

    const logits = activations.at(-1);
    const maximum = Math.max(...logits);
    const exponentials = logits.map(value => Math.exp(value - maximum));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    const probabilities = exponentials.map(value => value / total);

    if (!probabilities.every(Number.isFinite)) {
        throw new Error("Non-finite output probabilities");
    }

    const prediction = probabilities.indexOf(Math.max(...probabilities));
    return { activations, preactivations, probabilities, prediction };
}

export function backward(result, label, snapshot) {
    if (!Number.isInteger(label) ||
        label < 0 || label >= result.probabilities.length) {
        throw new Error("Invalid label");
    }

    let delta = result.probabilities.map(
        (value, index) => value - (index === label ? 1 : 0)
    );
    const nodeGradients = Array(snapshot.weights.length);

    for (let layer = snapshot.weights.length - 1; layer >= 0; layer--) {
        nodeGradients[layer] = [...delta];

        if (layer > 0) {
            const weights = snapshot.weights[layer];
            const values = result.preactivations[layer - 1];

            delta = weights.map((row, neuron) => {
                if (values[neuron] <= 0) return 0;
                return row.reduce(
                    (sum, weight, output) => sum + weight * delta[output],
                    0
                );
            });
        }
    }

    return nodeGradients;
}

export function verifyExport(data) {
    if (data.schema !== 1 || !data.samples.length || !data.snapshots.length) {
        throw new Error("Unsupported or empty network export");
    }

    let maximumError = 0;

    for (const snapshot of data.snapshots) {
        data.samples.forEach((sample, sampleIndex) => {
            const result = forward(sample.pixels, snapshot);
            const expected = snapshot.probabilities[sampleIndex];

            if (!expected || expected.length !== result.probabilities.length) {
                throw new Error("Missing Python reference probabilities");
            }

            result.probabilities.forEach((value, digit) => {
                const error = Math.abs(value - expected[digit]);
                if (!Number.isFinite(error)) {
                    throw new Error("Invalid reference probability");
                }
                maximumError = Math.max(maximumError, error);
            });
        });
    }

    if (maximumError > 1e-8) {
        throw new Error(`Browser/Python mismatch: ${maximumError}`);
    }
    return maximumError;
}
