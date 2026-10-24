"""problem_01.py - transforms in torchvision (15/40 points)

Write the lines of code to fill in the blanks and make the unit tests pass.
Pay attention to the instructions in the function docstring carefully. The
    transforms will need to be implemented exactly as described for the unit
    tests to pass and full credit to be awarded. Each passing test earns credit.

Problems and parts of problems do not depend on each other. Be sure to move on
to the next problem if you get stuck on one problem.

**TIP**: To uncomment multiple lines in VS Code, highlight them and use
`Ctrl` + `/`.

Classes:
    CustomImageDataset: Loads a custom image dataset with three images in it
        for the purposes of testing. **Do not edit this class. It is not part
        of the assessment.**

Functions:
    problem_01: Tests basic knowledge of building composite transforms in
        torch / torchvision. **This is the part of the code that must be
        edited to receive credit.**
"""

import os
from collections.abc import Callable

import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

Transform = Callable[[Tensor], Tensor]


class CustomImageDataset(Dataset):
    """Load the three labeled images used by this problem.

    **Do not edit this class. This is not part of the assignment.**
    """

    def __init__(
        self,
        annotations_file: str,
        image_directory: str,
        transform: Transform | None = None,
        target_transform: Transform | None = None,
    ) -> None:
        """Initialize the example dataset.

        Args:
            annotations_file: The file that holds the annotations.
            image_directory: The directory the images are stored in.
            transform: Torch transform to apply to the data when each item is
                loaded.
            target_transform: Torch transform to apply to targets when each
                item is loaded.

        Returns:
            None
        """
        self.image_labels: pd.DataFrame = pd.read_csv(
            annotations_file, header=None
        )
        self.image_directory: str = image_directory
        self.transform: Transform | None = transform
        self.target_transform: Transform | None = target_transform

    def __len__(self) -> int:
        """Return the length of the loaded dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.image_labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get an item from the loaded dataset.

        Args:
            idx: The index to access.

        Returns:
            The image and label corresponding to the provided index.
        """
        img_path = os.path.join(
            self.image_directory, str(self.image_labels.iloc[idx, 0])
        )
        image: Tensor = read_image(img_path)
        label: Tensor = torch.tensor(self.image_labels.iloc[idx, 1])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


def problem_01() -> CustomImageDataset:
    """Problem 01 (15 points).

    In the space provided, write the code required to perform the following
    transforms in the order specified.

    1. Resize the image to 16x16 pixels.
    2. Flip the image vertically. Ensure the flip always happens (probability
        of occurrence should be equal to 100%).
    3. Flip the image horizontally. Ensure the flip always happens (probability
        of occurrence should be equal to 100%).

    Only edit the code **inside** the marked area. The resize, each flip, and
    their composition are tested separately for partial credit.

    Be sure to apply the transforms **exactly** as specified, in the order
    specified. Be sure to consult online documentation to set the parameters
    of each transform correctly.

    Returns:
        A CustomImageDataset object.
    """

    resize = v2.Resize((16, 16))
    vertical_flip = v2.RandomVerticalFlip(p=1.0)
    horizontal_flip = v2.RandomHorizontalFlip(p=1.0)

    transform = v2.Compose([
    resize,
    vertical_flip,
    horizontal_flip,
    ])

    dataset = CustomImageDataset(
        annotations_file="./assignment/example_data/custom_dataset/labels.txt",
        image_directory="./assignment/example_data/custom_dataset",
        transform=transform,
    )

    return dataset


def show_images(dataset: CustomImageDataset) -> None:
    """Save a visualization of the transformed dataset to images.png.

    Args:
        dataset: The custom image dataset to show.

    Returns:
        None
    """
    figure = Figure()
    for index in range(len(dataset)):
        image: Tensor
        label: Tensor
        image, label = dataset[index]
        axes = figure.add_subplot(len(dataset), 1, index + 1)
        axes.axis("off")
        axes.imshow(image.permute(1, 2, 0).squeeze(), cmap="gray")
        axes.set_title(f"Label = {label.item()}")
    figure.tight_layout()
    figure.savefig("images.png")


if __name__ == "__main__":
    # From the repository root, `uv run python -m assignment.problem_01`
    # produces images.png to help check the transforms visually.
    dataset: CustomImageDataset = problem_01()
    show_images(dataset)
