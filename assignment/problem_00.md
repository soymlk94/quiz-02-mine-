# Problem 00: Basics

Edit this file to solve problem 00. Problem 00 will not be auto-graded.

This question is most aligned to our fourth class, where we discussed basic ML best practices applied at different stages of the ML lifecycle.

## Problem 00 - Part A

When designing test cases for ML algorithms, care must be taken to avoid training cases leaking into test cases in unexpected ways. Select the choice which provides the safest way to split training and test data for a problem where objects must be detected in a video stream.

Indicate your choice below the options.

A. Always split the data randomly. Be sure to treat each frame as its own test case. Splitting the data randomly ensures there is no bias in the selection of test cases.

B. Be sure to train and test on completely separate videos. Do not make any modifications or apply any transformations / augmentations to the video frames used as test cases to ensure the test is as accurate as possible.

C. Be sure to train and test on completely separate videos, taking care to ensure that the videos do not contain overlapping scenes or recordings. Apply appropriate transformations to some test videos to evaluate robustness, updating bounding boxes or other annotations appropriately. Review the videos selected for testing and the modifications applied to ensure the test cases chosen are representative.

D. Split the data randomly and apply random transformations and augmentations. The specific types of transformations and the order they are applied in are not important. It is most important to ensure that the widest possible variety of transforms are applied.

Your choice (2.5 points): [your choice here]

Explain why you chose the answer in 1-2 sentences (2.5 points): [your text here]

## Problem 00 - Part B

It is important to use the correct tools for each stage of our ML pipeline when we deploy models to production systems. Select the toolchain below that is best suited for deployment to a production system for the correct reasons.

A. Train models in PyTorch, use pytest for unit tests with the highest practical coverage, and save the models to pickle files so they are easy to load later. This is important because pickle files make it easy to load data into Python programs and Jupyter notebooks.

B. Train the models in PyTorch, use pytest for unit tests with the highest practical coverage, and Captum for explainability. Save the models to ONNX files. Saving to ONNX files ensures that Captum will be able to load our models with all relevant gradient information needed to run explainability algorithms on them.

C. Train the models in PyTorch, use pytest for unit tests with the highest practical coverage, and Captum for explainability. Save the models to ONNX files. Captum should be used during model development and after the final models are ready to be cut for deployment. Use multiple algorithms in Captum. This enables using the outputs of Captum to guide training and tuning.

D. Train the models in a custom C++ framework. Ensure the code has the highest unit test practical coverage. Save the models to ONNX files. After training runs for production are complete, load the ONNX files directly into Captum and run Integrated Gradients, since ONNX preserves the PyTorch automatic differentiation graph.

Your choice (2.5 points): [your choice here]

Explain why you chose the answer in 1-2 sentences (2.5 points): [your text here]
