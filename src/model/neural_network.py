import numpy as np


class NeuralNetwork:
    def __init__(self, dimensions, seed=42):
        if len(dimensions) < 2 or any(size < 1 for size in dimensions):
            raise ValueError("Provide at least two positive layer sizes")

        self.dimensions = list(dimensions)
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.normal(0, np.sqrt(2 / inputs), (inputs, outputs))
            for inputs, outputs in zip(dimensions[:-1], dimensions[1:])
        ]
        self.biases = [np.zeros(size) for size in dimensions[1:]]

    def forward(self, inputs):
        activations = [np.asarray(inputs, dtype=np.float64)]
        if activations[0].ndim != 2:
            raise ValueError("Inputs must be a two-dimensional array")
        if activations[0].shape[1] != self.dimensions[0]:
            raise ValueError("Input width differs from the network")

        preactivations = []
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            values = activations[-1] @ weights + bias
            preactivations.append(values)
            if index < len(self.weights) - 1:
                values = np.maximum(values, 0)
            activations.append(values)

        logits = activations[-1]
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(
            np.exp(shifted).sum(axis=1, keepdims=True)
        )

        return {
            "activations": activations,
            "preactivations": preactivations,
            "log_probabilities": log_probabilities,
            "probabilities": np.exp(log_probabilities),
        }

    def loss_and_gradients(self, inputs, labels, regularization=0.001):
        labels = np.asarray(labels)
        cache = self.forward(inputs)
        count = len(labels)

        if count == 0 or labels.shape != (len(inputs),):
            raise ValueError("Provide one label per input")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Labels must be integers")
        if np.any(labels < 0) or np.any(labels >= self.dimensions[-1]):
            raise ValueError("Label outside output range")
        if regularization < 0:
            raise ValueError("Regularization cannot be negative")

        loss = -cache["log_probabilities"][np.arange(count), labels].mean()
        loss += regularization * sum(
            np.sum(weights * weights) for weights in self.weights
        ) / 2

        delta = cache["probabilities"].copy()
        delta[np.arange(count), labels] -= 1
        delta /= count

        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)
        node_gradients = [None] * len(self.weights)

        for index in range(len(self.weights) - 1, -1, -1):
            node_gradients[index] = delta.copy()
            weight_gradients[index] = (
                cache["activations"][index].T @ delta
                + regularization * self.weights[index]
            )
            bias_gradients[index] = delta.sum(axis=0)

            if index:
                delta = delta @ self.weights[index].T
                delta *= cache["preactivations"][index - 1] > 0

        return float(loss), weight_gradients, bias_gradients, node_gradients

    def predict(self, inputs):
        return self.forward(inputs)["probabilities"].argmax(axis=1)

    @property
    def parameter_count(self):
        return sum(
            weights.size + bias.size
            for weights, bias in zip(self.weights, self.biases)
        )
