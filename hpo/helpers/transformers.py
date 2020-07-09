import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin


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
