import argparse
import sys
import inspect
import dotenv
import logging

from .install import install_prerequisits
from .datasets import (coauthor_create_dataset, boundary_detection_create_dataset, detectrl_create_dataset,
                       mixset_create_dataset, sendetex_create_dataset)
from .datasets.aitdna_dataset.postprocessing import run_dataset_anonymization, run_synthetic_data_generation
from .datasets.aitdna_dataset.processing import run_data_formatting
from .datasets.aitdna_dataset.preprocessing import run_process_csv_data
from .analysis import run_stats_computation
from .experiments.mgtd import run_evaluate_predictors

def _get_main_parser():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="CLI for benchmarking notions of MGTD for hybrid text"
    )

    current_module = sys.modules[__name__]
    functions = {
        name: func
        for name, func in inspect.getmembers(current_module, inspect.isfunction)
        if func.__module__ == __name__ and name not in {"main", "_parser", "_logging", "_get_main_parser"}
    }

    parser.add_argument(
        "function",
        choices=functions.keys(),
        help="Function to run"
    )

    return parser, functions


def _logging(level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    root = logging.getLogger()

    root.setLevel(level)
    root.addHandler(handler)


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m aitdna` and `$ aitdna `.

    Entry point to most important functions.
    """
    _logging()
    logger = logging.getLogger(__name__)

    loaded = dotenv.load_dotenv() # loads the .env
    if loaded:
        logger.info("Loaded .env file")
    else:
        logger.error("Failed to load .env file")

    parser, functions = _get_main_parser()
    args, remaining_args = parser.parse_known_args()

    # Call the selected function
    logger.info(f"Running {args.function}...")
    functions[args.function](remaining_args)

# ####
#
# Below follow the CLI functions that can be called
#
# ####

def install(argv=None):
    install_prerequisits()


def create_dataset(argv=None):
    parser = argparse.ArgumentParser(
        description="CLI for creating datasets of the benchmark"
    )
    parser.add_argument("-dt", "--dataset_type", type=str, required=True, help="Type of dataset to create")
    args, remaining_args = parser.parse_known_args(argv)
    dt = args.dataset_type

    handlers = {
        "boundary_detection": boundary_detection_create_dataset,
        "coauthor": coauthor_create_dataset,
        "detectRL": detectrl_create_dataset,
        "mixset": mixset_create_dataset,
        "senDetEx": sendetex_create_dataset,
    }

    if dt not in handlers:
        raise ValueError(f"Unknown dataset type: {dt}")
    else:
        handlers[dt](remaining_args)


def format_dataset(argv=None):
    logging.info(f"formatting data: {argv}")
    run_data_formatting(argv=argv)

def process_csv(argv=None):
    run_process_csv_data(argv=argv)

def anonymize_dataset(argv=None):
    run_dataset_anonymization(argv=argv)

def generate_synthetic_texts(argv=None):
    run_synthetic_data_generation(argv=argv)

def compute_dataset_stats(argv=None):
    run_stats_computation(argv=argv)

def run_predictors(argv=None):
    # TODO add evaluate_all option
    run_evaluate_predictors(argv=argv)

