from pathlib import Path

from PIL import Image
import typer


def main(output_dir: Path, seed: int = 42):
    for image in range(10):
        Image.new("RGB", (10, 10)).save(output_dir / f"{image}.tif")


if __name__ == "__main__":
    typer.run(main)
