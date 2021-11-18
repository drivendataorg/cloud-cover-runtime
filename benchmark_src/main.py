import os
from pathlib import Path
import shutil

from loguru import logger
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import typer

from cloud_dataset import CloudDataset
from cloud_model import CloudModel

ROOT_DIRECTORY = Path("/codeexecution")
SUBMISSION_DIRECTORY = ROOT_DIRECTORY / "submission"
ASSETS_DIRECTORY = ROOT_DIRECTORY / "assets"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"


def get_metadata(features_dir: os.PathLike, bands: list[str]):
    """
    Given a folder of feature data, return a dataframe where the index is the chip id
    and there is a column for the path to each band's TIF image.

    Args:
        features_dir (os.PathLike): path to the directory of feature data, which should have
            a folder for each chip
        bands (list[str]): list of bands provided for each chip
    """
    chip_metadata = pd.DataFrame(index=[f"{band}_path" for band in bands])
    chip_ids = [pth.name for pth in features_dir.iterdir()]

    for chip_id in chip_ids:
        chip_bands = [features_dir / chip_id / f"{band}.tif" for band in bands]
        chip_metadata[chip_id] = chip_bands

    return chip_metadata.transpose().reset_index().rename(columns={"index": "chip_id"})


def make_predictions(model: CloudModel, x_paths: pd.DataFrame, bands: list[str]):
    """
    Returns a dictionary where each key is the chip id and each value is the
    predict cloud mask as a numpy array.

    Args:
        model (CloudModel): an instantiated CloudModel based on pl.LightningModule
        x_paths (pd.DataFrame): a dataframe with a row for each chip. There must be a column for chip_id,
                and a column with the path to the TIF for each of bands provided
        bands (list[str]): list of bands provided for each chip
    """
    test_dataset = CloudDataset(x_paths=x_paths, bands=bands)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=model.batch_size,
        num_workers=model.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    chip_preds = {}
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        x = batch["chip"]
        preds = model.forward(x)
        preds = torch.softmax(preds, dim=1)[:, 1]
        preds = (preds > 0.5).detach().numpy().astype("uint8")
        chip_preds.update(dict(zip(batch["chip_id"], preds)))

    return chip_preds


def save_predictions(chip_preds: dict, predictions_dir: os.PathLike, overwrite: bool):
    """
    Save the predictions provided in chip_preds to predictions_dir.

    Args:
        chip_preds (dict): Dictionary of predictions where the keys are chip_ids and the values
            are predicted TIF masks as numpy arrays
        predictions_dir (os.PathLike): Destination directory to save the predicted TIF masks
        overwrite (bool): whether to delete and overwrite predictions_dir if it already exists
    """
    if predictions_dir.exists():
        if len(list(predictions_dir.iterdir())) > 0 and not overwrite:
            raise ValueError(
                f"{predictions_dir} already exists. To overwrite it, set overwrite to True"
            )
        else:
            print(
                f"{predictions_dir} already exists. Deleting it to generate new predictions."
            )
            shutil.rmtree(predictions_dir)
    predictions_dir.mkdir(parents=True)

    for chip_id, chip_pred in tqdm(chip_preds.items()):
        chip_pred_path = predictions_dir / f"{chip_id}.tif"
        chip_pred_im = Image.fromarray(chip_pred)
        chip_pred_im.save(chip_pred_path)


def main(
    model_weights_path: Path = ASSETS_DIRECTORY / "cloud_model.pt",
    test_features_dir: Path = DATA_DIRECTORY / "test_features",
    predictions_dir: Path = SUBMISSION_DIRECTORY,
    overwrite: bool = False,
    bands: list[str] = ["B02", "B03", "B04", "B08"],
    fast_dev_run: bool = False,
):
    """
    Generate predictions for the chips in test_features_dir using the model saved at model_weights_path.
    Predictions are saved in predictions_dir.

    Args:
        model_weights_path (Path): Path to the weights of a trained CloudModel
        test_features_dir (Path): Path to the features for the test data
        predictions_dir (Path): Destination directory to save the predicted TIF masks
        overwrite (bool, optional): Whether to delete and overwrite predictions_dir if it already exists.
            Defaults to False.
        bands (list[str], optional): List of bands provided for each chip
    """
    if not test_features_dir.exists():
        raise ValueError(
            f"The directory for test feature images must exists and {test_features_dir} does not exist"
        )

    logger.info("Loading model")
    model = CloudModel(bands=bands)
    model.load_state_dict(torch.load(model_weights_path))

    logger.info("Loading test metadata")
    test_metadata = get_metadata(test_features_dir, bands=bands)
    if fast_dev_run:
        test_metadata = test_metadata.head()
    logger.info(f"Found {len(test_metadata)} chips")

    logger.info("Generating predictions in batches")
    chip_preds = make_predictions(model, test_metadata, bands)

    logger.info(f"Saving predictions to {predictions_dir}")
    save_predictions(chip_preds, predictions_dir, overwrite=overwrite)
    logger.info(f"Saved {len(list(predictions_dir.iterdir()))} predictions")


if __name__ == "__main__":
    typer.run(main)
