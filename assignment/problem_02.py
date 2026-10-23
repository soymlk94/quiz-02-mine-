"""problem_02.py - attribution with Captum (15 points)

Write the lines of code to fill in the blanks and make the unit tests pass.
Pay attention to the instructions in the function docstring carefully. The
correct attribution technique **with the correct parameters** will have to be
called for the unit tests to pass and full credit to be awarded. Partial credit
is available for independently satisfied requirements.

Problems and parts of problems do not depend on each other. Be sure to move on
to the next problem if you get stuck on one problem.

**TIP**: To uncomment multiple lines in VS Code, highlight them and use
`Ctrl` + `/`.

Functions:
    problem_02: Tests basic knowledge of generating attribution maps with
       Captum. **This is the part of the code that must be edited to receive
       credit.**
"""

import numpy as np
import torch
from captum.attr import Saliency
from matplotlib.figure import Figure
from torch import Tensor


def problem_02() -> np.ndarray:
    """Problem 02 (15 points).

    In the space provided, write the code required to generate a Saliency map.
    Note that Saliency maps are a different type of attribution than the types
    we have looked at in class so far. Be sure to consult the online
    documentation for Captum to see how to generate a Saliency map.

    **Important**: Be sure to set "target = 2" (and be sure to check how to do
    that in the Captum documentation online). Saliency maps will be different
    depending on the target because the model's output for each class may have
    different sensitivities to changes in individual input pixels.

    Only edit the code **inside** the marked area. Partial credit will be
    awarded for code that does not pass the unit test.

    Returns:
        A numpy array with the attribution map.
    """

    # Load the pretrained model on the CPU and disable training-time dropout.
    model: torch.jit.ScriptModule = torch.jit.load(
        "assignment/models/model.pth", map_location=torch.device("cpu")
    )
    model.eval()

    # Load a test image to run through the model
    test_image_array: np.ndarray = np.load(
        "tests/test_data/problem_02_test_image.npy"
    )
    # The saved float32 image is (channels, height, width). Add a batch axis.
    test_image_tensor: Tensor = torch.tensor(
        test_image_array[np.newaxis, ...], device="cpu", requires_grad=True
    )

    # Be sure to set target = 2 to get the right answer!
    # Remember that attribution maps will look different for different targets.
    # For example, setting target = 1 will check how sensitive model output for
    # class 1 is to changes in the input image. Setting target = 2 will check
    # how sensitive model output for class 2 is to changes in the input image.

    # **** Start of your code *************************************************

    # **** End of your code ***************************************************

    # Convert the attribution output to an array
    actual_attribution_array = (
        actual_attribution.detach().cpu().numpy().squeeze()
    )

    return actual_attribution_array


def show_attributions(attribution_array: np.ndarray) -> None:
    """Save the attribution map to attribution.png.

    Args:
        attribution_array: the array of attribution outputs.

    Returns:
        None
    """
    figure = Figure(figsize=(4, 3))
    ax = figure.subplots()
    ax.imshow(attribution_array)
    ax.set_title("Attributions")
    ax.axis("off")
    figure.tight_layout()
    figure.savefig("attribution.png")


if __name__ == "__main__":
    # From the repository root, `uv run python -m assignment.problem_02`
    # produces attribution.png to help check the attribution map visually.
    attribution_array: np.ndarray = problem_02()
    show_attributions(attribution_array)
