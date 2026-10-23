# Quiz 01

This is a quiz on ML basics and Python tools for deep learning.

See the instructions in the `assignment` folder for each problem.

For open-ended questions, fill in the words or code to answer the question where indicated. For multiple choice questions, uncomment your selection

Use Python 3.13 or newer and uv. From the repository root, install the project dependencies and run the tests with

```bash
uv sync --locked
uv run pytest
```

Problem 01 has four independent tests worth 3, 4, 4, and 4 points for image
dimensions, vertical flip, horizontal flip, and composition/order. The tests
fail until the marked student code is filled in. To run just Problem 01, use
`uv run pytest tests/test_problem_01.py`.

## Testing Individual Problems

You can test each individual problem without running the unit tests by doing the following.

```bash
uv run python -m assignment.problem_01
uv run python -m assignment.problem_02
```

These will output images, called `images.png` and `attribution.png` respectively, that you can use to check your work.

To check code style and imports, run `uv run ruff check .` and
`uv run ruff format --check .`.

## Do Not Change Grading Notebooks

The notebooks in the `assignment/grading_resources` folder are not part of the assignment. They help generate test data for grading. Do not edit or run them. Editing and running these notebooks may prevent auto-grading from working.

## License

This repository is provided with an MIT license. See the `LICENSE` file.

## Contributing

Please email Mauro Sanchirico at ms3978@camden.rutgers.edu (academic) or sanchirico.mauro@gmail.com (personal) with questions, comments, bug reports, or suggestions for improvement.
