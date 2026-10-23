"""Independent checks for the 15-point image transform problem."""

import pytest
import torch
from torch import Tensor
from torchvision.transforms import v2

from assignment.problem_01 import CustomImageDataset, problem_01


def _student_dataset() -> CustomImageDataset:
    """Turn the blank template's undefined transform into a test failure."""
    try:
        return problem_01()
    except NameError as error:
        if error.name != "transform":
            raise
        pytest.fail(
            "Define transform inside the marked area of problem_01().",
            pytrace=False,
        )


def _student_transform() -> v2.Transform:
    transform = _student_dataset().transform
    assert transform is not None, "Pass the transform to CustomImageDataset."
    return transform


def test_image_dimensions() -> None:
    """3 points: every dataset image has exactly 16 by 16 pixels."""
    dataset = _student_dataset()
    for index in range(len(dataset)):
        image, _ = dataset[index]
        assert image.shape[-2:] == (16, 16)


def test_vertical_flip() -> None:
    """4 points: the vertical flip always reverses an asymmetric image."""
    transform = _student_transform()
    steps = getattr(transform, "transforms", ())
    vertical = next(
        (step for step in steps if isinstance(step, v2.RandomVerticalFlip)),
        None,
    )
    assert vertical is not None, "Add a vertical flip to the composition."
    assert vertical.p == 1.0, "The vertical flip must always occur."

    image = torch.arange(16, dtype=torch.uint8).reshape(1, 16, 1)
    image = image.expand(1, 16, 16).clone()
    actual: Tensor = transform(image)
    assert actual.shape[-2] >= 2
    assert actual[:, 0, :].float().mean() > actual[:, -1, :].float().mean()


def test_horizontal_flip() -> None:
    """4 points: the horizontal flip always reverses an asymmetric image."""
    transform = _student_transform()
    steps = getattr(transform, "transforms", ())
    horizontal = next(
        (step for step in steps if isinstance(step, v2.RandomHorizontalFlip)),
        None,
    )
    assert horizontal is not None, "Add a horizontal flip to the composition."
    assert horizontal.p == 1.0, "The horizontal flip must always occur."

    image = torch.arange(16, dtype=torch.uint8).reshape(1, 1, 16)
    image = image.expand(1, 16, 16).clone()
    actual: Tensor = transform(image)
    assert actual.shape[-1] >= 2
    assert actual[:, :, 0].float().mean() > actual[:, :, -1].float().mean()


def test_composition_and_order() -> None:
    """4 points: compose exactly resize, vertical flip, horizontal flip."""
    transform = _student_transform()
    assert isinstance(transform, v2.Compose)
    steps = transform.transforms
    assert len(steps) == 3
    assert isinstance(steps[0], v2.Resize)
    assert isinstance(steps[1], v2.RandomVerticalFlip)
    assert isinstance(steps[2], v2.RandomHorizontalFlip)
