import logging
from collections import ChainMap
from itertools import product
from logging import config as log_config
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


log_config.fileConfig(r"./log.conf")
logger = logging.getLogger(__name__)


def define_hyperparameters():
    # Define pipeline structure
    logger.info("Set up pipeline")
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2,)
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
    return pipeline, space


def combine_spaces(spaces: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [dict(ChainMap(*combination)) for combination in product(*spaces)]


def explore_search_space(
    X: pd.Series,
    y: pd.Series,
    random_optimizer: bool,
    pipeline: Pipeline,
    space: List[Dict[str, Any]],
    scoring: List[str],
    refit: str,
    workers: int,
    cv: int = 3,
    verbose: int = 2,
) -> Dict[str, Any]:
    if not random_optimizer:
        optimizer = GridSearchCV(
            estimator=pipeline,
            param_grid=space,
            scoring=scoring,
            refit=refit,
            cv=cv,
            n_jobs=workers,
            verbose=verbose,
        )
    else:
        optimizer = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=space,
            scoring=scoring,
            refit=refit,
            cv=cv,
            n_jobs=workers,
            n_iter=3,
            random_state=0,
            verbose=verbose,
        )

    # Train models
    return optimizer.fit(X, y)


def optimize(X: pd.Series, y: pd.Series, workers: int, random_optimizer: bool):
    # Get architecture and its hyperparameters
    pipeline, space = define_hyperparameters()

    # Explore search space and evaluate architectures
    logger.info("Explore search space")
    return explore_search_space(
        X=X,
        y=y,
        random_optimizer=random_optimizer,
        pipeline=pipeline,
        space=space,
        scoring=["accuracy", "average_precision", "f1"],
        refit="accuracy",
        workers=workers,
    )
