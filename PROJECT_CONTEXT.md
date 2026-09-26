# FPGA Digit Recognition Project Notes

## Project Idea

I want to build a handwritten digit recognition system that eventually runs on an FPGA.

The basic idea is that someone can draw a number from 0-9, the image gets converted into pixel data, and a classifier predicts which number was drawn. I eventually want the actual prediction calculations to happen on an FPGA instead of just running a Python model on my computer.

I also want a website or popup where I can draw a digit and actually see what is happening. It would be cool to show the original drawing, what it looks like after processing, the prediction, and eventually information about how the FPGA handled it.

I don't want this to just be a copy of a normal digit recognition project. I want to make my own dataset and use the project to test different ideas.

## Things I Want to Figure Out

Some questions I want to answer while building this:

- What is the best image resolution to use?
- Is 8x8 enough, or would something like 16x16 work noticeably better?
- How much extra FPGA hardware does a larger image require?
- What kind of classifier makes sense for an FPGA?
- How accurate can I make it without making the hardware unnecessarily complicated?
- How many bits are actually needed to represent pixels, weights, and calculations?
- What happens to accuracy when I reduce the number of bits?
- How different is fixed-point math from the normal calculations used while training?
- How fast can the FPGA classify one digit?
- How many FPGA resources will the design use?
- How does FPGA inference compare with running the same classifier normally on my computer?
- Will it recognize handwriting from people whose writing was not in the training data?

These can change as I learn more.

## Current System Idea

Right now I imagine the system working something like this:

Drawing
-> image
-> crop/resize/process image
-> pixel values
-> classifier
-> predicted number

Later:

Drawing in browser
-> image processing
-> pixel data sent to FPGA
-> FPGA performs classifier calculations
-> FPGA sends result back
-> browser displays prediction

The browser should not secretly do the final prediction once I reach the FPGA version. I want the FPGA to actually be responsible for inference.

## Dataset Idea

I originally considered starting with an existing handwritten digit dataset, but I would rather create my own main dataset.

I want samples of all ten digits:

0 1 2 3 4 5 6 7 8 9

I still need to decide:

- how I will collect the handwriting
- how many examples I need
- how many different people should write digits
- what image format to save
- how the files should be named
- how labels should be stored
- whether digits should be drawn digitally, written on paper, or both
- how to make sure the test data is actually separate from the training data

I want to keep every original sample unchanged. Anything produced by resizing, cropping, converting to grayscale, or other processing should be saved separately.

I might later test an existing public dataset too, but mainly as a benchmark against my own dataset.

## Image Resolution

I do not want to choose 8x8 just because an existing dataset uses it.

This seems like a good first experiment.

For example:

8x8 = 64 pixels

16x16 = 256 pixels

A 16x16 image has four times as many pixels, so it could preserve more information about the handwriting, but it also means the classifier has more input data to process.

I want to measure whether the extra accuracy is worth the extra hardware and computation.

Possible comparison:

8x8 vs 16x16

Things to measure:

- classification accuracy
- number of calculations
- memory needed
- FPGA resource usage
- inference time

I can change or add resolutions later if there is a reason to.

## Classifier

Not chosen yet.

I want to learn enough about the possible classifiers before deciding which one makes the most sense.

The important part is that the classifier eventually needs to be practical to implement with digital hardware.

I should not choose something only because it gives good accuracy in Python if it becomes extremely complicated or inefficient on an FPGA.

## Number Representation

Python and machine learning software often use floating-point numbers.

An FPGA implementation may be much more efficient using fixed-point integers with a limited number of bits.

I want this to become another experiment instead of choosing a bit width randomly.

Possible tests could include different precisions such as:

8-bit
6-bit
4-bit

Exact formats have not been decided yet.

I want to compare how reducing precision changes:

- accuracy
- memory
- hardware size
- speed

## FPGA

I do not have an FPGA board yet.

I don't think I need to buy one immediately because I can first work on:

- collecting data
- preprocessing
- Python models
- experiments
- fixed-point versions
- SystemVerilog
- simulation

Once I understand what the hardware needs, I can choose a board that actually fits the project.

Things to consider when choosing one:

- LUTs / logic resources
- flip-flops
- block RAM
- DSP blocks
- USB or another way to communicate with my computer
- price
- development software
- whether the tools will work with my Chromebook/Linux setup

I already own an Arduino. I can use it if I find an actual reason for it later, but I don't want to add it just to make the project look more complicated.

## Visual Interface Idea

Eventually I want a local website or similar interface.

A user could draw a digit with the mouse and press something like "Predict."

I would like the interface to eventually show things such as:

- original drawing
- processed image
- pixel grid
- predicted digit
- output scores if they are useful
- current image resolution
- current number precision
- FPGA clock cycles
- inference time

It could also become useful for comparing experiments.

For example, I might be able to run the same drawing through an 8x8 design and a 16x16 design and compare the results.

This is for later. I want the actual recognition system working before spending too much time making the interface look good.

## Folder Notes

### data/raw/

Original dataset.

Do not let preprocessing code overwrite these files.

### data/processed/

Generated versions of the dataset after processing.

### src/data/

Python code dealing with the dataset and image preprocessing.

### src/model/

Classifier training, prediction, and evaluation code.

### src/fixed_point/

Python versions of the classifier calculations that imitate the limited-precision math I eventually want to implement on the FPGA.

### experiments/

Measurements and experiment results.

### hardware/rtl/

SystemVerilog modules that can eventually be synthesized onto the FPGA.

### hardware/tb/

Testbenches for checking the hardware modules.

### interface/

Code for the visual interface.

### docs/

Other documentation if I need it.

## Rough Build Order

This will probably change, but my current plan is:

1. Set up the project and GitHub repository.
2. Decide exactly how to create the dataset.
3. Collect the first version of the dataset.
4. Learn how to load and inspect the images with Python.
5. Build the preprocessing system.
6. Build a simple software classifier as a baseline.
7. Test different image resolutions.
8. Test different classifier ideas if needed.
9. Study how the classifier's calculations could work in hardware.
10. Make a fixed-point Python version.
11. Experiment with different bit widths.
12. Design the FPGA architecture.
13. Write the SystemVerilog modules.
14. Write testbenches and simulate them.
15. Compare the hardware simulation against the Python reference.
16. Choose an FPGA board.
17. Synthesize and run the design on the actual FPGA.
18. Set up communication between the Chromebook and FPGA.
19. Connect everything to the visual interface.
20. Measure the final accuracy, latency, and hardware usage.
21. Document what worked, what did not, and what I learned.

## Current Progress

### Done

- Created the GitHub repository.
- Connected the local repository to GitHub.
- Created a Python virtual environment.
- Decided to make a custom dataset instead of making an existing dataset the center of the project.
- Decided that image resolution should be tested instead of automatically using 8x8.
- Decided that I eventually want a visual interface.
- Created the new project folder structure.

### Currently Working On

Finishing the initial repository setup.

### Next

Figure out exactly how I want to collect the handwritten digit dataset.

Before writing the classifier, I want to decide how samples will be collected, labeled, stored, and separated into training/testing data.

## Project Structure Right Now

digit-fpga/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── experiments/
├── hardware/
│   ├── rtl/
│   └── tb/
├── interface/
├── src/
│   ├── data/
│   ├── fixed_point/
│   └── model/
├── .gitignore
├── README.md
└── PROJECT_CONTEXT.md

## Running Notes

Keep adding important decisions, discoveries, experiment results, problems, and changes here as the project develops.

Do not treat early ideas in this file as permanent decisions. If testing shows that an idea is bad, change it and record why.

The point of these notes is to keep track of how the project developed from the beginning, including the reasoning behind important choices.
