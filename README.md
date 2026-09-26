# FPGA Handwritten Digit Recognition

A handwritten digit recognition system that I am building to learn how machine learning, numerical representation, and digital hardware can work together in an FPGA-based system.

## Project Idea

The goal is to recognize handwritten digits from 0-9 and eventually perform the prediction calculations on an FPGA.

Instead of only using an existing handwritten digit dataset, I am building a custom dataset and experimenting with different design choices.

Some of the main questions I want to investigate are:

- How does image resolution affect accuracy?
- Is 8x8 enough, or does 16x16 provide a useful improvement?
- How much FPGA hardware does a larger image require?
- What type of classifier works well when it needs to run on an FPGA?
- How many bits are actually needed for pixels, weights, and calculations?
- How much accuracy is lost when numerical precision is reduced?
- How does fixed-point inference compare with normal software calculations?
- How fast can the FPGA classify a digit?
- How many FPGA resources does the design use?

## Current Progress

The first version of the custom dataset collector is working.

The collector uses a simple local webpage for drawing digits and a Python server for saving the collected data.

Current features:

- 128x128 drawing canvas
- Writer ID and digit labeling
- PNG image saving
- Blank-drawing protection
- Unique sample numbering
- Protection against overwriting samples when numbering has gaps
- Per-digit collection progress
- Progress that persists after refreshing the page

The raw dataset is organized by writer and digit:

```text
data/raw/
└── writerID/
    └── digit/
        └── sample.png
