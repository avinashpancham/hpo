from typing import Any, Dict, List

import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline


class Anonymizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._nlp_model = spacy.load("en_core_web_sm")

    # Return self nothing else to do here
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        def _anonymize(doc: str):
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
    optimizer.fit(X, y)

    return optimizer.cv_results_
