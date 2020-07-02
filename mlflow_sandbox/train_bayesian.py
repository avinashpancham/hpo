import logging
from logging import config as log_config
from pathlib import Path

import mlflow
import mlflow.sklearn
import optuna

from helpers.cli_options import get_cli_options_hpo
from helpers.mlflow_helpers import create_experiment, mlflow_optuna_logging
from helpers.bayesian_helpers import objective, terminal_logging
from helpers.transformers import Anonymizer
from helpers.preprocessing import load_data

log_config.fileConfig(r"./log.conf")
logger = logging.getLogger(__name__)


base_folder = Path("../data/processed")
mlflow.tracking.set_tracking_uri("http://localhost:5000")


def train_model(
    sample_size: int,
    workers: int,
    trials: int,
    experiment: mlflow.entities.experiment.Experiment,
) -> None:
    logger.info("Load IMDB reviews")
    df_train, _ = load_data(folder=base_folder, sample_size=sample_size)

    # Anonymize data before pipeline, since this step is slow and constant
    logger.info("Preprocess reviews with spaCy. This may take a while..")
    anonymized_reviews = Anonymizer().transform(df_train.review)

    logger.info("Explore search space")
    study = optuna.create_study(direction="maximize")
    study.set_user_attr(key="sample_size", value=sample_size)
    study.set_user_attr(key="experiment", value=experiment)
    study.set_user_attr(key="data", value=base_folder / "train.csv")

    # Perform Hyperparameter optimization and log results
    study.optimize(
        lambda trial: objective(
            trial, X=anonymized_reviews, y=df_train.sentiment, workers=workers,
        ),
        n_trials=trials,
        callbacks=[terminal_logging, mlflow_optuna_logging],
    )


if __name__ == "__main__":
    sample_size, workers, trials = get_cli_options_hpo(bayesian=True)

    experiment = create_experiment(base_name="Sentiment")
    train_model(
        sample_size=sample_size, workers=workers, trials=trials, experiment=experiment,
    )

    logger.info(
        "Train best model by running 'python train_best_model.py -n %s'",
        experiment.name,
    )
