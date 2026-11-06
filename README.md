# FPGA Handwritten Digit Recognition

I am building a handwritten digit recognition system to study how machine learning, numerical precision, and digital hardware fit together. The project starts with a model trained on my own handwriting, then compares floating-point and fixed-point inference before moving individual calculations into SystemVerilog.

**[Open the interactive project](https://abhiveer-chhiber.github.io/digit-fpga/)**

[Project overview](https://abhiveer-chhiber.github.io/digit-fpga/interface/overview.html) · [Methods](https://abhiveer-chhiber.github.io/digit-fpga/interface/methods.html) · [Results](https://abhiveer-chhiber.github.io/digit-fpga/interface/results.html) · [Recorded hardware-neuron trace](https://abhiveer-chhiber.github.io/digit-fpga/interface/neuron-trace.html)

## What is implemented

- A custom dataset of 100 original 128×128 handwritten digit drawings, with 10 samples per digit from one writer.
- A preprocessing pipeline that creates matching 8×8 and 16×16 inputs from the same original drawings.
- Linear classification baselines and a NumPy neural network with a 256 → 128 → 64 → 10 architecture and 41,802 trainable parameters.
- Fixed-point software inference with explicit quantization, rounding, and saturation.
- A signed 12-bit fixed-point neuron written in SystemVerilog, with a 32-bit accumulator, optional ReLU, rounding, and saturation flags.
- An interactive browser viewer that uses exported model weights to display predictions, activations, and saved training checkpoints.

## Model results

The current dataset is small and contains handwriting from one writer. The validation split contains 20 drawings from that same writer, so these results are an early experiment, not evidence of generalization to new handwriting.

| Model | Training correct | Validation correct | Validation loss |
| --- | ---: | ---: | ---: |
| 8×8 linear baseline | 79 / 80 | 19 / 20 | 0.5076 |
| 16×16 linear baseline | 80 / 80 | 18 / 20 | 0.3600 |
| 16×16 neural network | 80 / 80 | 18 / 20 | 0.1408 |

The neural network reached 19 / 20 validation accuracy at earlier checkpoints and finished at 18 / 20. Accuracy and loss measure different things, so I track both.

## Fixed-point experiments

I compared integer inference with the floating-point neural network using the same saved checkpoint and dataset split.

- 16-bit storage with 10 fractional bits and 12-bit storage with 6 fractional bits produced the same predictions as the floating-point model on all 100 drawings.
- The 10-bit format changed one validation prediction and classified 19 / 20 validation drawings correctly. This is one changed result on a small, same-writer dataset, not proof that lower precision is better.
- The 8-bit format lost substantial accuracy at the tested scale because many weights rounded to zero.

The experiment report records the format, error, prediction agreement, and saturation counts: [fixed-point report](docs/fixed-point-v1.json).

## SystemVerilog neuron

The hardware module implements one neuron, not the complete neural network. It accepts input-weight pairs over successive clock cycles, accumulates their products, adds a saved bias, optionally applies ReLU, rounds the result, and clips values that exceed the supported range.

The neuron uses signed 12-bit inputs, weights, and outputs with 6 fractional bits. Products and the bias use 12 fractional bits, and the accumulator is signed 32-bit.

I tested the RTL with Icarus Verilog against Python-generated reference vectors:

| Test group | Cases |
| --- | ---: |
| Rounding and range boundaries | 36 |
| Generated integer inputs | 200 |
| Values from the trained model | 180 |
| **Total** | **416** |

All 416 outputs and saturation flags matched the Python reference. Additional checks covered reset during a calculation, gaps between valid inputs, bias and ReLU capture, ignored start signals while busy, and the one-cycle completion signal.

- [Hardware-neuron design and test details](docs/hardware-neuron.md)
- [Recorded simulation report](docs/neuron-sim-v1.json)
- [Interactive recorded neuron trace](https://abhiveer-chhiber.github.io/digit-fpga/interface/neuron-trace.html)

**Current boundary:** the single-neuron RTL has passed simulation. The complete neural network has not been implemented in RTL, synthesized, or run on a physical FPGA. FPGA timing, resource use, and end-to-end inference performance have not been measured. The public browser viewer performs software inference.

## Run the project

Create a virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Check the fixed-point software implementation against its independent scalar reference:

```bash
python src/fixed_point/verify.py
```

Run the SystemVerilog neuron simulation. This requires Icarus Verilog, including both `iverilog` and `vvp`, on your PATH:

```bash
python src/fixed_point/run_neuron_sim.py --name neuron-sim-local
```

Build the static website locally:

```bash
python src/build_site.py
python -m http.server 8001 --directory build/site
```

Then open http://localhost:8001/.

## Repository structure

- `data/raw/`: original handwritten PNG files.
- `src/data/`: collection, preprocessing, and dataset loading.
- `src/model/`: NumPy baselines, neural network training, and exports.
- `src/fixed_point/`: quantized inference, verification, test-vector generation, and simulation tools.
- `hardware/rtl/`: SystemVerilog design.
- `hardware/tb/`: SystemVerilog testbenches.
- `interface/`: interactive browser viewer and project pages.
- `docs/`: experiment reports and hardware design notes.

## Next steps

- Collect handwriting from additional writers and evaluate on writers excluded from training.
- Expand the single-neuron RTL into a layer and then the full network.
- Compare each hardware stage against the Python reference.
- Synthesize the design and measure resource use and timing on a target FPGA.
- Connect the hardware inference path to the interactive interface.
