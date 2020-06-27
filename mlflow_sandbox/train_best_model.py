import logging
from logging import config as log_config
from pathlib import Path
from typing import Optional

import mlflow
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

from helpers.cli_options import get_cli_options_single
from helpers.mlflow_helpers import get_experiment_id, get_best_model
from helpers.pipeline import Anonymizer
from helpers.preprocessing import load_data


log_config.fileConfig(r"./log.conf")
logger = logging.getLogger(__name__)

base_folder = Path("../data/processed")


def train_model(experiment_name: Optional[str]) -> None:
    logger.info("Load IMDB reviews")
    df_train, df_test = load_data(folder=base_folder, sample_size=10)

    logger.info("Load best model from MLflow")
    experiment_id = get_experiment_id(experiment_name=experiment_name)
    run_id, pipeline = get_best_model(experiment_id=experiment_id)

    logger.info("Construct new pipeline")
    anonymizer = Anonymizer()
    pipeline.steps.insert(0, ("anonymizer", anonymizer))
    pipeline.verbose = True

    logger.info("Train model on complete dataset")
    pipeline.fit(df_train.review, df_train.sentiment)

    logger.info("Evaluate model performance")
    prediction = pipeline.predict(df_test.review)
    logger.info(
        "Accuracy: %s", (accuracy := accuracy_score(df_test.sentiment, prediction))
    )
    logger.info(
        "Precision: %s",
        (precision := average_precision_score(df_test.sentiment, prediction)),
    )
    logger.info("F1: %s", (f1 := f1_score(df_test.sentiment, prediction)))

    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id):
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_f1_score": f1,
            }
        )
        mlflow.sklearn.log_model(pipeline, "final_model")
        mlflow.log_artifacts(base_folder)


if __name__ == "__main__":
    experiment_name = get_cli_options_single()
    if experiment_name:
        logger.info("Train best model from experiment %s", experiment_name)
    train_model(experiment_name=experiment_name)
