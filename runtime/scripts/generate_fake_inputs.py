from pathlib import Path

import numpy as np
from PIL import Image
import typer


def main(output_dir: Path, seed: int = 42):
    for chip in range(10):
        chip_dir = output_dir / f"{chip:04d}"
        chip_dir.mkdir(exist_ok=True, parents=True)
        for band in ["B02", "B03", "B04", "B08"]:
            im = Image.new(mode="L", size=(512, 512))
            im.putdata(np.random.randint(0, 256, size=(512, 512), dtype=np.uint8))
            im.save(chip_dir / f"{band}.tif")


if __name__ == "__main__":
    typer.run(main)
