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

## Dataset Planning

### Original Sample Format

I want to collect the original handwritten digits at a higher resolution instead of drawing them directly as 8x8 or 16x16 images.

My current idea is to make a simple browser drawing canvas where a person can draw one digit at a time. The original drawing will be saved before it is resized for any experiments.

This way I can create different versions, such as 8x8 and 16x16, from the same original handwriting. This should make it easier to compare resolutions fairly because the actual handwriting stays the same.

At this point, I had not decided on the exact original canvas size yet. I decided on 128x128 later while building the first collector.

### Sample Information

Each saved sample should represent one handwritten digit.

For each sample, I want to keep track of:

- the correct digit from 0-9
- a writer ID
- the original 128x128 drawing
- a unique sample ID

I want to use writer IDs because I eventually want to test whether the classifier works on handwriting from people it did not see during training. This should give me a better test of how well the system generalizes instead of only randomly mixing samples from the same people between training and testing.

The writer ID does not need to contain the person's actual name. It can be something simple such as `writer01`, `writer02`, etc.

### Original Resolution

The current original drawing canvas is 128x128 pixels.

I chose a larger original image so I can create smaller versions from the same handwriting later instead of collecting separate drawings for each resolution.

The first planned resolution comparison is:

- 8x8
- 16x16

These are not final choices. I can change or add resolutions if the early experiments give me a reason to.

### First Collection Target

For the first version of the dataset, I plan to collect 10 samples of each digit from each writer.

That means:

10 digits x 10 samples = 100 samples per writer

I do not have a final number of writers yet. I will start by collecting my own samples and make sure the full data pipeline works before asking other people to contribute handwriting.

If I eventually collect from 10 writers, that would give me 1,000 original samples.

I can increase the number of samples later if the experiments show that the dataset is too small.

I also want the collector to keep track of how many examples of each digit have already been collected so that the dataset stays balanced.

## 2026-09-26 - Dataset Collector v1

I built the first working version of the custom dataset collector.

The collector runs as a local webpage and currently has:

- a 128x128 drawing canvas
- a writer ID input
- a digit label from 0-9
- a clear button
- a record sample button
- a session sample counter
- basic input validation

The canvas is displayed larger in the browser so it is easier to draw on, but the actual saved image stays 128x128.

I decided to use a small Python HTTP server for saving the dataset instead of trying to handle file storage entirely in the browser. The browser handles drawing and sends the sample to Python. Python validates the information and writes the PNG into the raw dataset.

Current data path:

Browser canvas
-> PNG data
-> HTTP POST
-> Python collector server
-> data/raw/

Raw samples are organized like this:

data/raw/writerID/digit/sample.png

Example:

data/raw/writer01/7/0001.png

This keeps the writer and correct digit easy to identify without needing a separate database for basic labels.

### First End-to-End Test

I tested the collector using a temporary writer ID and a handwritten 7.

The browser reported that the sample was saved to:

data/raw/test/7/0001.png

I checked the saved file and confirmed:

- the file existed
- it contained PNG data
- the PNG signature was valid
- the image dimensions were 128x128

The temporary test data was deleted after verification so it will not become part of the real dataset.

### Current Collector Architecture

The interface uses basic HTML, CSS, and JavaScript for drawing and sending samples.

Python handles the actual dataset storage.

I want to keep the browser side simple because the interface is only a tool for collecting and viewing data. Most of the project logic will stay in Python, and the final inference calculations will eventually move to the FPGA.

### Collector Improvements Completed

After the first end-to-end test, I made several changes before using the collector for real data.

I added blank-drawing protection so the browser will not record a sample unless something has actually been drawn on the canvas.

I also changed the sample numbering system. Originally, the next number was based only on how many PNG files were already in the folder. This could cause an existing file to be overwritten if there was a gap in the numbering. The collector now finds the largest existing sample number and adds one.

I tested this by creating 0001.png, 0002.png, and 0003.png, deleting 0002.png, and recording another sample. The new sample correctly became 0004.png instead of overwriting 0003.png.

The collector now also shows progress for each digit from 0-9 using the target of 10 samples per digit per writer.

These counts come from the actual files stored in the dataset instead of only being stored in the browser. I verified this by recording a sample, refreshing the page, and confirming that the session counter reset while the saved writer progress remained.

### Collector v1 Verification

The current collector has been tested for:

- drawing and clearing digits
- writer ID validation
- digit labeling
- blank-sample prevention
- PNG saving
- 128x128 image dimensions
- unique sequential sample numbering
- protection against overwriting when numbering has gaps
- per-digit progress tracking
- progress persistence across browser refreshes

Temporary test samples were deleted after testing.

At this point, the first version of the dataset collector is complete enough to begin collecting the initial custom dataset.

The next major step after data collection will be building the preprocessing pipeline.

## 2026-10-05 - First Real Handwriting Collection

I collected my first 100 samples under writer01, with 10 of each digit from 0-9.

I checked the counts and used Pillow to open all the images. All 100 were readable 128x128 PNGs. I also looked at them together in a contact sheet and did not notice any blank images or wrong labels.

Nine drawings touched the canvas edge:

- 0/0007.png
- 4/0001.png
- 5/0002.png
- 5/0004.png
- 5/0009.png
- 6/0001.png
- 7/0002.png
- 8/0008.png
- 9/0009.png

Some strokes were cut off, but the digits were still recognizable. I kept them because I want to include imperfect handwriting too. Later I can check whether edge-touching drawings are harder for the classifier.

I decided to keep the original PNGs in Git while the dataset is small so someone else can use the same samples to repeat my experiments. Processed images will stay separate from the originals.

I installed Pillow in the project's virtual environment.

These samples are all my handwriting. I can start building preprocessing with them, but I still need other writers to test whether the classifier recognizes handwriting it has not trained on.

Next I want to see how the same drawings look at 8x8 and 16x16. I have not chosen the classifier yet.

## 2026-10-05 - First Preprocessing Version

I wrote src/data/preprocess.py to make 8x8 and 16x16 versions of the same original drawings.

The script finds the digit using an ink threshold of 32, crops the surrounding whitespace, and centers the crop in a square. The longest side of the digit takes up about 75% of that square, leaving padding around it. It keeps the original proportions instead of stretching narrow digits.

I used BOX resizing to keep average ink intensity when shrinking the images. Background pixels are 0 and ink pixels go up to 255. The threshold is only used to find the crop, so the processed images still keep grayscale values.

The script also creates a CSV with the original and processed paths, writer, label, crop bounds, and whether the original touches an edge.

It produced 100 images at each resolution. I checked their dimensions, grayscale format, labels, and counts against the CSV. All checks passed, and the original files were unchanged.

I looked at one sample of each digit at both resolutions. The 16x16 versions kept more detail, especially the loops in 6 and 8. I still need to compare recognition accuracy before deciding which resolution to use.

Generated data is already ignored by Git. It can be recreated by running the preprocessing script.

## 2026-10-05 - Training and Validation Split

I saved a fixed split with 80 training samples and 20 validation samples. Each digit has 8 training drawings and 2 validation drawings. The split uses seed 42 and is saved in data/splits/writer01-v1.json.

Both resolutions use the same original samples in each group, so the resolution comparison will not depend on different drawings being selected.

I also wrote a dataset loader that converts processed pixels to values from 0 to 1 and flattens each image into a row. An 8x8 image has 64 inputs and a 16x16 image has 256.

I checked the input shapes, pixel ranges, label counts, and sample assignments. All checks passed.

This is still a same-writer validation split. It does not tell me how well the model handles other people's handwriting. There are only 20 validation samples, so one mistake changes accuracy by 5 percentage points.
