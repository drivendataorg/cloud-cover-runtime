from functools import reduce
from operator import mul
from pathlib import Path
from random import randint

from PIL import Image
import typer


def main(output_dir: Path, seed: int = 42):
    for chip in range(10):
        chip_dir = output_dir / f"{chip:04d}"
        chip_dir.mkdir(exist_ok=True, parents=True)
        for band in range(4):
            im = Image.new(mode="L", size=(10, 10))
            im.putdata([randint(0, 255) for _ in range(reduce(mul, im.size))])
            im.save(chip_dir / f"B{band:02d}.tif")


if __name__ == "__main__":
    typer.run(main)
