from pathlib import Path
import logging
from logging import config as log_config

import mlflow

from helpers.cli_options import get_cli_options_hpo
from helpers.gridsearch_helpers import optimize
from helpers.mlflow_helpers import create_experiment, mlflow_sklearn_logging
from helpers.transformers import Anonymizer
from helpers.preprocessing import load_data


log_config.fileConfig(r"./log.conf")
logger = logging.getLogger(__name__)


METRIC_SCORE = "accuracy"
base_folder = Path("../data/processed")
mlflow.tracking.set_tracking_uri("http://localhost:5000")


def train_model(
    sample_size: int,
    workers: int,
    random_optimizer: bool,
    experiment: mlflow.entities.experiment.Experiment,
) -> None:
    logger.info("Load IMDB reviews")
    df_train, _ = load_data(folder=base_folder, sample_size=sample_size)

    # Anonymize data before pipeline, since this step is slow and constant
    logger.info("Preprocess reviews with spaCy. This may take a while..")
    anonymized_reviews = Anonymizer().transform(df_train.review)

    # Perform Hyperparameter optimization
    optimizer = optimize(
        X=anonymized_reviews,
        y=df_train.sentiment,
        workers=workers,
        random_optimizer=random_optimizer,
    )

    # MLflow logging of results
    logger.info("Write results to MLflow experiment: %s", experiment.name)
    mlflow_sklearn_logging(
        optimizer=optimizer,
        experiment_id=experiment.experiment_id,
        sample_size=sample_size,
        data=base_folder / "train.csv",
    )


if __name__ == "__main__":
    sample_size, workers, random_optimizer = get_cli_options_hpo(bayesian=False)

    experiment = create_experiment(base_name="Sentiment")
    train_model(
        sample_size=sample_size,
        workers=workers,
        random_optimizer=random_optimizer,
        experiment=experiment,
    )

    logger.info(
        "Train best model by running 'python train_best_model.py -n %s'",
        experiment.name,
    )
