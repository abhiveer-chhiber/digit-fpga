# FPGA Handwritten Digit Recognition

I am building a handwritten digit recognition system to learn how machine learning, numerical precision, and digital hardware work together. I have implemented the software model and published an interactive learning website. My next hardware goal is FPGA inference.

**[Explore my model and its training](https://abhiveer-chhiber.github.io/digit-fpga/)**

## Learning Website

I collected 100 handwritten digit drawings and trained a real neural network on 80 of them, holding out 20 for validation.

I built the website to make the calculations and training progress easier to explore:

- Replay 11 recorded training checkpoints
- Compare predictions on my validation drawings
- Inspect neuron activations, biases, and sample gradients
- Rotate the network visualization
- Follow training and validation loss and accuracy
- Read my code decisions and why I made them

I use the actual saved weights to calculate predictions in the browser. I checked all 220 exported sample-checkpoint predictions against Python.

Playback uses recorded checkpoints. I animate signals to illustrate forward and backward calculations, rather than to measure training or hardware timing. I display a subset of neurons for readability and use the complete network for prediction calculations.

## My Dataset

I drew and collected 100 original 128×128 PNG images, with 10 drawings per digit from one writer.

I preprocess the originals into centered grayscale images at 8×8 and 16×16. I preserve the original files. I kept nine drawings that touch an image edge to retain variation in my handwriting.

I use a fixed split of 80 training drawings and 20 validation drawings. Both resolutions use identical sample assignments.

```text
data/raw/
└── writer01/
    └── digit/
        └── sample.png
```

## Models and Results

I implemented a linear baseline and a neural network in NumPy.

For the neural network, I use ReLU hidden layers, softmax output, cross-entropy loss, L2 regularization, and Adam optimization. My architecture is **256 → 128 → 64 → 10**, with **41,802 trainable parameters**.

My final results after 1,000 epochs:

| Model | Input resolution | Training correct | Validation correct | Validation accuracy |
| --- | --- | --- | --- | --- |
| Linear baseline | 8×8 | 79 / 80 | 19 / 20 | 95% |
| Linear baseline | 16×16 | 80 / 80 | 18 / 20 | 90% |
| Neural network | 16×16 | 80 / 80 | 18 / 20 | 90% |

I reached 95% validation accuracy at several earlier neural network checkpoints, then finished at 90%. My larger model did not beat my 8×8 baseline at the final checkpoint.

I evaluated only 20 validation drawings from the same writer as the training data. Each prediction changes validation accuracy by five percentage points. I have not yet established performance on other people's handwriting.

I verified both model implementations with numerical gradient checks. I also reloaded neural network checkpoints and confirmed that they reproduced the saved metrics.

## Questions I Am Investigating

- How does image resolution affect accuracy?
- When does a larger model provide a useful improvement?
- How well does my model handle handwriting from new writers?
- How many bits do I need for pixels, weights, and intermediate calculations?
- How much accuracy do I lose when I reduce numerical precision?
- How does fixed-point inference compare with floating-point software?
- What FPGA resources, latency, and throughput can I achieve?

## Running My Website Locally

I keep the published model snapshot in `site_assets/network-view.json`, so I can run the viewer without retraining.

From the repository root:

```bash
python3 src/build_site.py
python3 -m http.server 8001 --bind 127.0.0.1 --directory build/site
```

Open http://localhost:8001/.

I configured GitHub Actions to rebuild and publish the website when I push changes to `main`.

## Next Steps

- Collect handwriting from additional writers
- Evaluate drawings from writers excluded from training
- Compare model choices through recorded experiments
- Implement and verify fixed-point inference
- Build and simulate my FPGA inference design
- Measure hardware accuracy, latency, and resource use

I have not implemented FPGA execution yet. My current website calculates predictions in the browser.
