import logging
from logging import config as log_config
from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

log_config.fileConfig(r"./log.conf")
logger = logging.getLogger(__name__)


def define_hyperparameters(trial):
    ngram_range = trial.suggest_categorical("vectorizer__ngram_range", ["11", "12"])
    vectorizer = TfidfVectorizer(
        stop_words="english", min_df=2, ngram_range=string_to_tuple(ngram_range),
    )

    clf = trial.suggest_categorical(
        "clf", ["SVC", "RandomForestClassifier", "MultinomialNB"]
    )
    if clf == "SVC":
        C = trial.suggest_uniform("clf__C", 0.1, 0.2)
        classifier = SVC(C=C)
    elif clf == "RandomForestClassifier":
        max_depth = trial.suggest_int("clf__max_depth", 2, 4)
        classifier = RandomForestClassifier(max_depth=max_depth)
    else:
        alpha = trial.suggest_loguniform("clf__alpha", 1e-2, 1e-1)
        classifier = MultinomialNB(alpha=alpha)

    return Pipeline([("vectorizer", vectorizer), ("clf", classifier)])


def string_to_tuple(s: str) -> Tuple[int, int]:
    return int(s[0]), int(s[-1])


def terminal_logging(study, trial):
    trial_value = trial.value if trial.value is not None else float("nan")
    logger.info(
        "Trial number: %s, %s",
        trial.number,
        ", ".join(f"{key}: {value}" for key, value in trial.params.items()),
    )
    logger.info("Accuracy: %s", trial_value)


def objective(trial, X: pd.Series, y: pd.Series, workers: int) -> int:
    # Get architecture and its parameters
    pipeline = define_hyperparameters(trial=trial)

    # Evaluate architectures and store settings and metrics
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=3,
        scoring=["accuracy", "average_precision", "f1"],
        n_jobs=workers,
    )

    trial.set_user_attr(key="model", value=pipeline)
    for metric in ("accuracy", "average_precision", "f1"):
        trial.set_user_attr(key=metric, value=scores[f"test_{metric}"].mean())

    return scores["test_accuracy"].mean()
