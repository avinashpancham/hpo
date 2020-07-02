from datetime import datetime
from typing import Optional, Tuple, Any, Dict

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline


def create_experiment(base_name: str) -> mlflow.entities.experiment.Experiment:
    uid = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    experiment_id = mlflow.create_experiment(name=f"{base_name}_{uid}")
    return mlflow.get_experiment(experiment_id=experiment_id)


def get_experiment(
    experiment_name: Optional[str],
) -> mlflow.entities.experiment.Experiment:

    client = MlflowClient()
    experiments = client.list_experiments()

    # Return most recent experiment if the user did not provide an experiment
    if not experiment_name:
        return max(
            [experiment for experiment in experiments],
            key=lambda exp: int(exp.experiment_id),
        )

    # Return the experiment provided by the user if it exists
    experiment = [
        experiment for experiment in experiments if experiment.name == experiment_name
    ]

    if not experiment:
        raise ValueError(f"Experiment {experiment_name} does not exist")

    return experiment[0]


def get_best_model(experiment_id: str) -> Tuple[str, Pipeline]:
    df_runs = mlflow.search_runs(
        experiment_ids=experiment_id, filter_string="tags.model='True'"
    )
    max_acc_idx = df_runs.sort_values(
        by=["metrics.accuracy", "metrics.average_precision", "metrics.f1"],
        ascending=False,
    ).index[0]
    run_id = df_runs.loc[max_acc_idx].run_id
    return run_id, mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def mlflow_sklearn_logging(
    optimizer: Dict[str, Any], experiment_id: str, sample_size: int, data: str
) -> None:
    # TODO: change optimier type hinting
    models = optimizer.cv_results_

    for ind, (model, acc, precision, f1, rank) in enumerate(
        zip(
            models["params"],
            models["mean_test_accuracy"],
            models["mean_test_average_precision"],
            models["mean_test_f1"],
            models["rank_test_f1"],
        )
    ):
        with mlflow.start_run(experiment_id=experiment_id):
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
            mlflow.log_metrics(
                {"accuracy": acc, "average_precision": precision, "f1": f1}
            )

            if rank == 1:
                mlflow.set_tag("model", True)
                mlflow.sklearn.log_model(optimizer.best_estimator_, "model")
                mlflow.log_artifact(
                    data, "data",
                )


def mlflow_optuna_logging(study, trial):
    experiment = study.user_attrs["experiment"]
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_params({"N": study.user_attrs["sample_size"], **trial.params})
        mlflow.sklearn.log_model(trial.user_attrs.pop("model"), "model")
        mlflow.log_metrics(trial.user_attrs)
        mlflow.log_artifact(study.user_attrs["data"], "data")
        mlflow.set_tag(key="model", value=True)
