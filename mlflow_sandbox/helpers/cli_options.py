import argparse
from typing import Tuple, Optional


def get_cli_options_multiple() -> Tuple[int, int, bool]:
    parser = argparse.ArgumentParser(description="Provide input for training models")
    parser.add_argument(
        "--size", "-n", type=int, default=10000, help="Provide sample size"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=2, help="Provide number of workers"
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        default=False,
        help="Use RandomSearch Optimizer, else use default GridSearch Optimizer",
    )

    args = parser.parse_args()
    return args.size, args.workers, args.random


def get_cli_options_single() -> Optional[str]:
    parser = argparse.ArgumentParser(description="Provide input for training models")
    parser.add_argument(
        "--name", "-n", type=str, default=None, help="Provide MLflow experiment name"
    )
    args = parser.parse_args()
    return args.name
