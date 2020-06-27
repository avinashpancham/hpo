import pandas as pd
import logging
from logging import config as log_config
from itertools import chain
from pathlib import Path, PosixPath

import html2text
from tqdm import tqdm

log_config.fileConfig(r"./log.conf")
logger = logging.getLogger(__name__)


# Initialize html2text object with correct ignore settings
h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_emphasis = True

# Base data folders with data from http://ai.stanford.edu/~amaas/data/sentiment/
raw_data_folder = Path(r"../data/raw/aclImdb")
processed_data_folder = Path(r"../data/processed/aclImdb")


def load_and_parse_reviews(folder: PosixPath) -> pd.DataFrame:
    logger.info("Parsing %s dataset", folder.name)
    files = chain(folder.glob("pos/*.txt"), folder.glob("neg/*.txt"))

    reviews = [
        {
            "movie_id": file.name.split("_")[0],
            "grade": int(file.name.split(".txt")[0].split("_")[1]),
            "review": file.read_text().lower(),
            "sentiment": 1 if file.parent.name == "pos" else 0,
        }
        for file in tqdm(files, desc="Parsing reviews")
    ]

    return pd.DataFrame(reviews)


def remove_html(df: pd.DataFrame) -> pd.DataFrame:
    df["review"] = df.review.apply(h.handle)
    return df


def remove_pipe_from_body(df: pd.DataFrame) -> pd.DataFrame:
    df["review"] = df.review.replace("|", " ")
    return df


def store_df(df: pd.DataFrame, folder: PosixPath) -> pd.DataFrame:
    df.to_csv(folder.parent / f"{folder.name}.csv", sep="|", index=False)
    logger.info("%s dataset stored in %s", folder.name.title(), folder.parent)
    return df


if __name__ == "__main__":
    for dataset in ("train", "test"):
        load_and_parse_reviews(folder=raw_data_folder / dataset).pipe(remove_html).pipe(
            remove_pipe_from_body
        ).pipe(store_df, folder=processed_data_folder / dataset)
