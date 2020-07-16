"""Global settings for all tests environment."""
import dataclasses
import os
import typing
import yaml

CURRENT_DIR = os.path.dirname(__file__)

DEFAULT_NETWORK_CONFIGURATION_PATH = os.path.join(CURRENT_DIR, '../', 'values.yaml')
SUBSTRA_TESTS_CONFIG_FILEPATH = os.getenv('SUBSTRA_TESTS_CONFIG_FILEPATH', DEFAULT_NETWORK_CONFIGURATION_PATH)

MIN_NODES = 2


@dataclasses.dataclass(frozen=True)
class NodeCfg:
    name: str
    msp_id: str
    address: str
    user: str = None
    password: str = None
    shared_path: str = None


@dataclasses.dataclass(frozen=True)
class Options:
    celery_task_max_retries: int
    enable_intermediate_model_removal: bool


@dataclasses.dataclass(frozen=True)
class Settings:
    path: str
    options: Options
    nodes: typing.List[NodeCfg] = dataclasses.field(default_factory=list)


_SETTINGS = None


def _load_yaml(path):
    """Load configuration from yaml file."""
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.Loader)
    nodes = [NodeCfg(**kw) for kw in data['nodes']]
    assert len(nodes) >= MIN_NODES, f'not enough nodes: {len(nodes)}'
    return Settings(
        path=path,
        nodes=nodes,
        options=Options(**data['options'])
    )


def load():
    """Loads settings static configuration.

    As the configuration is static and immutable, it is loaded only once from the disk.

    Returns an instance of the `Settings` class.
    """
    global _SETTINGS
    if _SETTINGS is not None:
        return _SETTINGS

    s = _load_yaml(SUBSTRA_TESTS_CONFIG_FILEPATH)
    _SETTINGS = s
    return _SETTINGS


# TODO that's a bad idea to expose the static configuration, it has been done to allow
#      tests parametrization but this won't work for specific tests written with more
#      nodes

# load configuration at module load time to allow tests parametrization depending on
# network static configuration
load()


MSP_IDS = [n.msp_id for n in _SETTINGS.nodes]
HAS_SHARED_PATH = bool(_SETTINGS.nodes[0].shared_path)
CELERY_TASK_MAX_RETRIES = _SETTINGS.options.celery_task_max_retries
