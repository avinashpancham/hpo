from pathlib import Path
import logging
from logging import config as log_config

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


from helpers.cli_options import get_cli_options_multiple
from helpers.mlflow_helpers import create_experiment
from helpers.pipeline import Anonymizer, combine_spaces, explore_search_space
from helpers.preprocessing import load_data


log_config.fileConfig(r"./log.conf")
logger = logging.getLogger(__name__)


METRIC_SCORE = "f1"
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

    # Define pipeline structure
    logger.info("Set up pipeline")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=2,
        max_features=None,
        analyzer="word",
    )
    pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("clf", DummyClassifier()),  # DummyClassifier as Placeholder
        ]
    )

    # Define hyperparameter space
    vectorizer_space = [{"vectorizer__ngram_range": [(1, 1), (1, 2)]}]

    clf_space = [
        {"clf": [MultinomialNB()], "clf__alpha": [0.01, 0.1]},
        {"clf": [RandomForestClassifier()], "clf__max_depth": np.arange(2, 4)},
        {"clf": [SVC()], "clf__C": [0.1, 0.2]},
    ]
    space = combine_spaces(spaces=[vectorizer_space, clf_space])

    # Explore search space
    logger.info("Explore search space")
    optimizer = explore_search_space(
        X=anonymized_reviews,
        y=df_train.sentiment,
        random_optimizer=random_optimizer,
        pipeline=pipeline,
        space=space,
        scoring=["accuracy", "average_precision"] + [METRIC_SCORE],
        refit=METRIC_SCORE,
        workers=workers,
    )
    models = optimizer.cv_results_

    # MLflow logging of results
    logger.info("Write results to MLflow experiment: %s", experiment.name)
    for ind, (model, acc, precision, f1, rank) in enumerate(
        zip(
            models["params"],
            models["mean_test_accuracy"],
            models["mean_test_average_precision"],
            models["mean_test_f1"],
            models["rank_test_f1"],
        )
    ):
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            mlflow.log_params(
                {
                    "N": sample_size,
                    **{
                        parameter: (
                            str(value)
                            if parameter != "clf"
                            else str(value).split("(")[
                                0
                            ]  # Clf value in SK learn outputs wrong argument for model
                        )
                        for parameter, value in model.items()
                    },
                }
            )
            mlflow.log_metrics({"accuracy": acc, "precision": precision, "f1": f1})

            if rank == 1:
                mlflow.set_tag("best_model", True)
                mlflow.sklearn.log_model(optimizer.best_estimator_, "model")
                mlflow.log_artifact(
                    base_folder / "train.csv", "train.csv",
                )


if __name__ == "__main__":
    sample_size, workers, random_optimizer = get_cli_options_multiple()

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
