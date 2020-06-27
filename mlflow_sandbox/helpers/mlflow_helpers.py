from collections import ChainMap
from datetime import datetime
from itertools import product
from typing import Any, Dict, List

import mlflow


def get_experiment(base_name: str) -> mlflow.entities.experiment.Experiment:
    uid = datetime.utcnow().strftime("%Y%m%d_%H%M")
    experiment_id = mlflow.create_experiment(name=f"{base_name}_{uid}")
    return mlflow.get_experiment(experiment_id=experiment_id)


def combine_spaces(spaces: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [dict(ChainMap(*combination)) for combination in product(*spaces)]
