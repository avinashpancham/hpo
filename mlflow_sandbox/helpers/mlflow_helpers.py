from collections import ChainMap
from datetime import datetime
from hashlib import sha256
from itertools import product
from typing import Any, Dict, List

import mlflow


def get_experiment_id(name: str) -> str:
    if mlflow.get_experiment_by_name(name):
        return mlflow.get_experiment_by_name(name).experiment_id
    else:
        return mlflow.create_experiment(name=name)


def combine_spaces(spaces: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [dict(ChainMap(*combination)) for combination in product(*spaces)]


def get_group_id() -> str:
    return sha256(str(datetime.utcnow().timestamp()).encode()).hexdigest()[:5]
