from pathlib import Path
from string import ascii_lowercase
import numpy as np
import pandas as pd
from PIL import Image
import typer


def main(output_dir: Path, seed: int = 42):
    metadata = []
    for chip in range(10):
        chip_dir = output_dir / "test_features" / f"{chip:04d}"
        chip_dir.mkdir(exist_ok=True, parents=True)
        for band in ["B02", "B03", "B04", "B08"]:
            im = Image.new(mode="L", size=(512, 512))
            im.putdata(np.random.randint(0, 256, size=(512, 512), dtype=np.uint8))
            im.save(chip_dir / f"{band}.tif")
        metadata.append(
            {
                "chip_id": ascii_lowercase[chip] * 4,
                "location": "Earth",
                "datetime": f"2021-{chip + 1:02}-01T00:00:00Z",
            }
        )
    pd.DataFrame(metadata).to_csv(output_dir / "test_metadata.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
