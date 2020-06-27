import logging
import re
import tarfile
from logging import config as log_config
from pathlib import PosixPath
from typing import Tuple

import pandas as pd
import html2text
from tqdm import tqdm

log_config.fileConfig(r"./log.conf")
logger = logging.getLogger(__name__)

# Initialize html2text object with correct ignore settings
h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_emphasis = True


file_name_regex = re.compile(r"aclImdb/(train|test)/(pos|neg)/\d+_([1-9]|10).txt")


def unzip_and_parse_reviews(folder: PosixPath) -> pd.DataFrame:
    logger.info("Extract and parse dataset")
    with tarfile.open(folder / "aclImdb_v1.tar.gz") as tar:
        reviews = [
            {
                "file_path": file.name,
                "review": tar.extractfile(file).read().decode("utf-8"),
            }
            for file in tqdm(tar.getmembers())
            if file_name_regex.match(file.name)
        ]

    return pd.DataFrame(reviews)


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Add metadata")
    df[["dataset", "sentiment", "file_name"]] = df.file_path.str.split(
        "/", expand=True
    )[[1, 2, 3]]
    df[["movie_id", "grade"]] = df.file_name.str.split("_|.txt", expand=True)[[0, 1]]

    # Binarize sentiment prediction
    df.sentiment.replace({"pos": 1, "neg": 0}, inplace=True)

    # Drop redundant columns
    return df.drop(columns=["file_path", "file_name"])


def remove_html(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Remove HTML content from reviews")
    df["review"] = df.review.apply(h.handle)
    return df


def remove_pipe_from_body(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Remove pipe (|) from reviews")
    df["review"] = df.review.replace("|", " ")
    return df


def store_datasets(df: pd.DataFrame, folder: PosixPath) -> pd.DataFrame:
    for dataset in ("train", "test"):
        logger.info("%s dataset stored in %s", dataset.title(), folder)
        df.query(f'dataset=="{dataset}"').drop(columns="dataset").to_csv(
            folder / f"{dataset}.csv", sep="|", index=False
        )

    return df


def load_data(folder: PosixPath, sample_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(folder / "train.csv", sep="|").sample(sample_size, random_state=0),
        pd.read_csv(folder / "test.csv", sep="|").sample(sample_size, random_state=0),
    )
