from pathlib import Path

import numpy as np
from PIL import Image
import typer


def main(output_dir: Path, seed: int = 42):
    for chip in range(10):
        chip_dir = output_dir / f"{chip:04d}"
        chip_dir.mkdir(exist_ok=True, parents=True)
        for band in range(4):
            Image.fromarray(
                np.random.randint(low=0, high=256, size=(10, 10)), mode="L"
            ).save(chip_dir / f"B{band:02d}.tif")


if __name__ == "__main__":
    typer.run(main)
