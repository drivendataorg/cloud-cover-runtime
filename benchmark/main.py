from pathlib import Path
from urllib.request import urlopen

import planetary_computer as pc
from PIL import Image
from pystac_client import Client
import typer

ROOT_DIRECTORY = Path("/codeexecution")
SUBMISSION_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"


def main():
    """Just check if we can call the Planetary Computer STAC API."""
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    area_of_interest = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.27508544921875, 47.54687159892238],
                [-121.96128845214844, 47.54687159892238],
                [-121.96128845214844, 47.745787772920934],
                [-122.27508544921875, 47.745787772920934],
                [-122.27508544921875, 47.54687159892238],
            ]
        ],
    }

    search = catalog.search(
        collections=["landsat-8-c2-l2"],
        intersects=area_of_interest,
        datetime="2020-12-01/2020-12-31",
    )

    items = list(search.get_items())
    selected_item = sorted(items, key=lambda item: item.properties["eo:cloud_cover"])[0]
    thumbnail_asset = selected_item.assets["thumbnail"]
    signed_href = pc.sign(thumbnail_asset.href)

    Image.open(urlopen(signed_href)).save(SUBMISSION_DIRECTORY / "thumb.tif")


if __name__ == "__main__":
    typer.run(main)
