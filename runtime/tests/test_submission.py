from pathlib import Path

SUBMISSION_DIR = Path("submission")
INPUT_IMAGES_DIR = Path("data/test_features")
MAX_FILE_SIZE = 512 * 512 * 2  # 2x fudge factor over one byte (uint8) per pixel


def test_all_files_in_format_have_corresponding_submission_file():
    pass


def test_no_unexpected_tif_files_in_submission():
    pass


def test_file_sizes_are_within_limit():
    for submission_path in SUBMISSION_DIR.glob("*.tif"):
        size_bytes = submission_path.stat().st_size
        err_msg = f"{submission_path} is {size_bytes:,} bytes; over limit of {MAX_FILE_SIZE:,} bytes"
        assert size_bytes <= MAX_FILE_SIZE, err_msg
