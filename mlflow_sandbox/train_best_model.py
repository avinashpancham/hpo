import logging
import pandas as pd
from logging import config as log_config
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import spacy
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

from helpers.cli_options import get_cli_options_single
from helpers.mlflow_helpers import get_experiment, get_best_model
from helpers.preprocessing import load_data


log_config.fileConfig(r"./log.conf")
logger = logging.getLogger(__name__)

base_folder = Path("../data/processed")
mlflow.tracking.set_tracking_uri("http://localhost:5000")


# Defining class here (again) because pickling model will give issues
# if the custom transformer is saved in another file
class Anonymizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._nlp_model = spacy.load("en_core_web_sm")

    # Return self nothing else to do here
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        def _anonymize(doc: spacy.tokens.doc.Doc) -> str:
            text = doc.text
            for ent in doc.ents:
                text = text.replace(ent.text, ent.label_)
            return text

        return pd.Series(
            [
                _anonymize(doc)
                for doc in self._nlp_model.pipe(X, disable=["tagger", "parser"])
            ]
        )


def train_model(experiment_name: Optional[str]) -> None:
    logger.info("Load IMDB reviews")
    df_train, df_test = load_data(folder=base_folder, sample_size=None)

    logger.info("Load best model from MLflow")
    experiment = get_experiment(experiment_name=experiment_name)
    run_id, pipeline = get_best_model(experiment_id=experiment.experiment_id)

    logger.info("Construct new pipeline")
    anonymizer = Anonymizer()
    pipeline.steps.insert(0, ("anonymizer", anonymizer))
    pipeline.verbose = True

    logger.info("Train model on complete dataset")
    pipeline.fit(df_train.review, df_train.sentiment)

    logger.info("Evaluate model performance")
    prediction = pipeline.predict(df_test.review)
    accuracy = accuracy_score(df_test.sentiment, prediction)
    precision = average_precision_score(df_test.sentiment, prediction)
    f1 = f1_score(df_test.sentiment, prediction)

    logger.info("Accuracy: %s", accuracy)
    logger.info("Precision: %s", precision)
    logger.info("F1: %s", f1)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_id):
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_f1_score": f1,
            }
        )
        mlflow.sklearn.save_model(
            pipeline, f"../trained_model/{experiment.name.lower()}"
        )
        mlflow.log_artifact(base_folder / "test.csv", "data")


if __name__ == "__main__":
    experiment_name = get_cli_options_single()
    if experiment_name:
        logger.info("Train best model from experiment %s", experiment_name)
    train_model(experiment_name=experiment_name)
