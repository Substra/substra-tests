"""Configuration of substratest package.

This is not the targetted network configuration.
"""
import os

_ENV_PREFIX = "SUBSTRA_TESTS"


def _getenv(name, default_env_value, converter=None):
    assert isinstance(default_env_value, str)
    env_name = f"{_ENV_PREFIX}_{name}"
    value = os.getenv(env_name, default_env_value)
    return converter(value) if converter else value


FUTURE_TIMEOUT = _getenv("FUTURE_TIMEOUT", "300", converter=int)  # seconds
FUTURE_POLLING_PERIOD = _getenv("FUTURE_POLLING_PERIOD", "0.5", converter=float)  # seconds
