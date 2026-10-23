"""Problem 02: algorithm (4), target (4), output (3), and map (4) points."""

from dataclasses import dataclass, field
from functools import wraps
from inspect import signature

import numpy as np
import pytest
from captum.attr import Saliency
from torch import Tensor

from assignment.problem_02 import problem_02


@dataclass
class Submission:
    """Keep partial observations even when student code is incomplete."""

    targets: list[object] = field(default_factory=list)
    output: np.ndarray | None = None
    error: Exception | None = None


@pytest.fixture
def submission(monkeypatch: pytest.MonkeyPatch) -> Submission:
    """Observe the real Captum call without replacing its computation."""
    observed = Submission()
    attribute = Saliency.attribute
    attribute_signature = signature(attribute)

    @wraps(attribute)
    def record_attribute(self, *args, **kwargs):
        arguments = attribute_signature.bind(self, *args, **kwargs)
        observed.targets.append(arguments.arguments.get("target"))
        return attribute(self, *args, **kwargs)

    # Patch the class method so aliases and imports inside student code work.
    monkeypatch.setattr(Saliency, "attribute", record_attribute)
    try:
        observed.output = problem_02()
    except Exception as error:
        # Store the error instead of aborting fixture setup for every test.
        observed.error = error
    return observed


def _output(submission: Submission) -> np.ndarray:
    if submission.error is not None:
        error = submission.error
        if isinstance(error, NameError) and error.name == "actual_attribution":
            pytest.fail(
                "Define actual_attribution inside the marked area of "
                "problem_02().",
                pytrace=False,
            )
        pytest.fail(
            f"problem_02() raised {type(error).__name__}: {error}",
            pytrace=False,
        )
    assert isinstance(submission.output, np.ndarray)
    return submission.output


def test_saliency_algorithm(submission: Submission) -> None:
    """4 points: call the attribution method of Captum's Saliency class."""
    assert submission.targets, "Use Captum's Saliency attribution method."


def test_target_class(submission: Submission) -> None:
    """4 points: request class 2 when calling the Saliency method."""
    for target in submission.targets:
        if isinstance(target, Tensor):
            target = target.detach().cpu().numpy()
        target = np.asarray(target)
        # Captum also accepts a single-element list or tensor for this batch.
        if target.size == 1 and target.item() == 2:
            return
    pytest.fail("Request target=2 in the Saliency call.", pytrace=False)


def test_output_shape_and_values(submission: Submission) -> None:
    """3 points: return a finite floating-point map with one value per pixel."""
    actual = _output(submission)
    assert actual.shape == (28, 28)
    assert np.issubdtype(actual.dtype, np.floating)
    assert np.isfinite(actual).all()


def test_expected_attribution(submission: Submission) -> None:
    """4 points: match the stored absolute Saliency map for class 2."""
    actual = _output(submission)
    expected = np.load("tests/test_data/problem_02_expected_attribution.npy")
    assert actual.shape == expected.shape, "Do not broadcast attribution maps."
    # The map peaks near 2e-4; a 1e-4 absolute tolerance is too permissive.
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-7)
