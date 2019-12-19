"""Experiment base definition.

To create a new experiment, create a python module with the following global variables:

data_sample_generator = None

OPENER_SCRIPT = None
METRICS_SCRIPT = None
ALGO_SCRIPT = None
COMPOSITE_ALGO_SCRIPT = None
AGGREGATE_ALGO_SCRIPT = None

METRICS_DOCKERFILE = None
ALGO_DOCKERFILE = None
"""
import dataclasses
import typing


@dataclasses.dataclass
class Experiment:
    opener_script: str
    metrics_script: str
    algo_script: str
    aggregate_algo_script: str
    composite_algo_script: str
    metrics_dockerfile: str
    algo_dockerfile: str
    data_sample_generator: typing.Callable


def create(module):
    return Experiment(
        # data sample / dataset
        data_sample_generator=module.data_sample_generator(),
        opener_script=module.OPENER_SCRIPT,
        # algos
        algo_script=module.ALGO_SCRIPT,
        aggregate_algo_script=module.AGGREGATE_ALGO_SCRIPT,
        composite_algo_script=module.COMPOSITE_ALGO_SCRIPT,
        algo_dockerfile=module.ALGO_DOCKERFILE,
        # objective
        metrics_script=module.METRICS_SCRIPT,
        metrics_dockerfile=module.METRICS_DOCKERFILE,
    )
