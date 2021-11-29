from pathlib import Path

from PIL import Image
import numpy as np

SUBMISSION_DIR = Path("predictions")
TEST_DIR = Path("data/test_features")
MAX_FILE_SIZE = 512 * 512 * 2  # 2x fudge factor over one byte (uint8) per pixel
EXPECTED_SHAPE = 512, 512
EXPECTED_VALUES = set((0, 1))

image_names = set(
    path.stem for path in TEST_DIR.glob("*") if not path.name.startswith(".")
)
submission_names = set(path.stem for path in SUBMISSION_DIR.glob("*.tif"))


def test_no_missing_files():
    missing_paths = image_names - submission_names

    assert (
        len(missing_paths) == 0
    ), f"""Submission is missing predictions for the following file(s)\n{", ".join(missing_paths)}"""


def test_no_extra_files():
    extra_paths = submission_names - image_names

    assert (
        len(extra_paths) == 0
    ), f"""Submission includes extra predictions for the following file(s)\n{", ".join(extra_paths)}"""


def test_valid_values():
    for submission_name in submission_names:
        submission = np.array(Image.open(SUBMISSION_DIR / f"{submission_name}.tif"))
        assert (
            submission.shape == EXPECTED_SHAPE
        ), f"{submission_name} shape={submission.shape}, expected {EXPECTED_SHAPE}"

        extra_values = set(np.unique(submission)) - EXPECTED_VALUES
        assert (
            len(extra_values) == 0
        ), f"""Invalid value(s) {", ".join(str(value) for value in extra_values)} present in {submission_name}.tif. """
        f"""Valid values are {", ".join(EXPECTED_VALUES)}"""


def test_file_sizes_are_within_limit():
    for name in submission_names:
        size_bytes = (SUBMISSION_DIR / f"{name}.tif").stat().st_size
        err_msg = (
            f"{name} is {size_bytes:,} bytes; over limit of {MAX_FILE_SIZE:,} bytes"
        )
        assert size_bytes <= MAX_FILE_SIZE, err_msg
