# Fixed-Point Hardware Neuron

The SystemVerilog neuron follows the same calculation order as the Python integer model.

## Number format

- Inputs, weights, and outputs: signed 12-bit values with 6 fractional bits
- Products and biases: 12 fractional bits
- Accumulator: signed 32-bit
- Rounding: nearest, with halfway values rounded away from zero
- Overflow: clip to the signed range and set a saturation flag
- ReLU: enabled for hidden neurons, disabled for output neurons

## Interface

A calculation starts when `start` is high and `busy` is low at a clock edge. The neuron saves `bias` and `relu_enable`, then clears the accumulator.

Starting at the next clock edge, each cycle with `input_valid` high adds one input-weight product. `input_last` marks the final valid pair. Invalid input cycles leave the calculation unchanged. Starts while busy are ignored.

One clock after the final pair, the neuron adds the saved bias, applies optional ReLU, rounds, and clips the output. `output_valid` is high for one cycle, and `busy` returns low. The output stays available while idle.

Without input gaps, completion takes N + 1 clock cycles after the accepted start, where N is the number of input-weight pairs. This is a cycle count from simulation, not a measured FPGA speed.

`accumulator_saturated` records clipping during accumulation or bias addition. `output_saturated` records clipping of the final output. Both flags clear at the next accepted start.

Reset is synchronous and active high. It cancels the current calculation and clears the output and flags.

## Simulation results

| Case group | Count |
| --- | --- |
| Rounding and range boundaries | 36 |
| Generated integer inputs | 200 |
| Values from the trained model | 180 |
| Total | 416 |

All outputs and saturation flags matched Python.

Additional checks passed for reset during a calculation, input gaps, saved bias and ReLU settings, ignored starts while busy, the completion pulse, and output held while idle.

## Running the checks

From the repository root:

`.venv/bin/python src/fixed_point/run_neuron_sim.py --name neuron-sim-v2`

Requires NumPy, Icarus Verilog, and vvp. The command generates vectors, compiles the neuron, runs the tests, and saves a report. Existing reports are not overwritten.

The first report is saved in `docs/neuron-sim-v1.json`, including hashes of the tested source files and vectors.

Current status: simulation passed. Synthesis and physical FPGA testing are still pending.
