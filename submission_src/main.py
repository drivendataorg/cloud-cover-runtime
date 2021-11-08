from pathlib import Path

from loguru import logger
import numpy as np
from PIL import Image
import typer

ROOT_DIRECTORY = Path("/codeexecution")
SUBMISSION_DIRECTORY = ROOT_DIRECTORY / "submission"

feature_directory = ROOT_DIRECTORY / "data" / "test_features"

chips = sorted(chip for chip in feature_directory.glob("*") if chip.is_dir())
logger.info(f"Processing {len(chips)} chips in {feature_directory}")


def main():
    for chip in chips:
        images = np.array(
            [np.array(Image.open(image)) for image in chip.glob("*.tif")]
        ).mean(0)
        Image.fromarray((images > images.mean()).astype(np.uint8)).save(
            SUBMISSION_DIRECTORY / f"{chip.name}.tif"
        )


if __name__ == "__main__":
    typer.run(main)
