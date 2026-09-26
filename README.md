# FPGA Handwritten Digit Recognition

A handwritten digit recognition system I am building to learn how to design, optimize, and eventually implement a machine learning classifier on an FPGA.

## Project Idea

The goal is to recognize handwritten digits from 0-9 and eventually perform the actual prediction calculations on an FPGA.

Instead of only using an existing handwritten digit dataset, I plan to collect my own data and experiment with different design choices.

Some of the main things I want to test are:

- different image resolutions
- different classifier designs
- floating-point vs. fixed-point calculations
- different numerical precisions
- classification accuracy
- FPGA resource usage
- inference latency

I also plan to build a visual interface where a digit can be drawn, and the different stages of the recognition process can be viewed.

## Current Plan

The project will start with data collection and a software version of the classifier. Once that is working, I will experiment with the model and convert its calculations to fixed-point arithmetic.

The fixed-point version will then be used as a reference for a SystemVerilog implementation. After simulation and testing, the design will be moved onto a physical FPGA.

A rough version of the final system is:

Drawing -> Image Processing -> Pixel Data -> FPGA -> Prediction -> Visual Interface

## Current Status

The repository and project structure are being set up.

The next step is to design the custom handwritten digit dataset and decide how the samples will be collected, labeled, and stored.
