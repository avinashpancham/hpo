import logging
from logging import config as log_config
from pathlib import Path

from helpers.preprocessing import (
    unzip_and_parse_reviews,
    add_metadata,
    remove_html,
    remove_pipe_from_body,
    store_datasets,
)

log_config.fileConfig(r"./log.conf")
logger = logging.getLogger(__name__)

# Folder with tar.gz file from http://ai.stanford.edu/~amaas/data/sentiment/
data_folder = Path(r"../data")


if __name__ == "__main__":
    logger.info("Preprocess IMDB review databset")
    unzip_and_parse_reviews(folder=data_folder / "raw").pipe(add_metadata).pipe(
        remove_html
    ).pipe(remove_pipe_from_body).pipe(store_datasets, folder=data_folder / "processed")
