from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import typer

rng = np.random.RandomState(seed=45)


def main(output_dir: Path, n: int = 4, seed: int = 45):
    metadata = []
    for chip in range(n):
        chip_dir = output_dir / "test_features" / f"{chip:04d}"
        chip_dir.mkdir(exist_ok=True, parents=True)
        for band in ["B02", "B03", "B04", "B08"]:
            im = Image.new(mode="L", size=(512, 512))
            im.putdata(rng.randint(0, 256, size=(512, 512), dtype=np.uint8))
            im.save(chip_dir / f"{band}.tif")
        metadata.append(
            {
                "chip_id": chip,
                "location": "Earth",
                "datetime": f"2021-{(chip % 12) + 1:02}-01T00:00:00Z",
            }
        )
    pd.DataFrame(metadata).to_csv(output_dir / "test_metadata.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
