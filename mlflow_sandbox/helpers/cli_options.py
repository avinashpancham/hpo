import argparse
from typing import Tuple, Optional, Union


def get_cli_options_hpo(
    bayesian: bool,
) -> Union[Tuple[int, int, int], Tuple[int, int, bool]]:
    parser = argparse.ArgumentParser(description="Provide input for training models")
    parser.add_argument(
        "--size", "-n", type=int, default=10000, help="Provide sample size"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=2, help="Provide number of workers"
    )

    if bayesian:
        parser.add_argument(
            "--trials",
            "--t",
            type=int,
            default=10,
            help="Number of trails for Bayesian optimization",
        )

    else:
        parser.add_argument(
            "--random",
            "-r",
            action="store_true",
            default=False,
            help="Use RandomSearch Optimizer, else use default GridSearch Optimizer",
        )

    args = parser.parse_args()
    return args.size, args.workers, args.trials if bayesian else args.random


def get_cli_options_single() -> Optional[str]:
    parser = argparse.ArgumentParser(description="Provide input for training models")
    parser.add_argument(
        "--name", "-n", type=str, default=None, help="Provide MLflow experiment name"
    )
    args = parser.parse_args()
    return args.name
