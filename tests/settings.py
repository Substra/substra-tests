"""Global settings for all tests environment."""
import dataclasses
import os
import typing
import yaml

SUBSTRAT_SKAFFOLD_FILEPATH = os.getenv('SUBSTRAT_SKAFFOLD_FILEPATH')

CURRENT_DIR = os.path.dirname(__file__)

DEFAULT_NETWORK_CONFIGURATION_PATH = os.path.join(CURRENT_DIR, '../', 'values.yaml')
SUBSTRAT_CONFIG_FILEPATH = os.getenv('SUBSTRAT_CONFIG_FILE')

NETWORK_CONFIGURATION_PATH = SUBSTRAT_CONFIG_FILEPATH or DEFAULT_NETWORK_CONFIGURATION_PATH
MIN_NODES = 2


@dataclasses.dataclass(frozen=True)
class NodeCfg:
    name: str
    msp_id: str
    address: str
    user: str = None
    password: str = None


@dataclasses.dataclass(frozen=True)
class Settings:
    path: str
    nodes: typing.List[NodeCfg] = dataclasses.field(default_factory=list)


_SETTINGS = None


def _load_yaml(path):
    """Load configuration from yaml file."""
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.Loader)
    nodes = [NodeCfg(**kw) for kw in data['nodes']]
    assert len(nodes) >= MIN_NODES, f'not enought nodes: {len(nodes)}'
    return Settings(
        path=path,
        nodes=nodes,
    )


def _load_skaffold(path):
    """Load configuration from a skaffold yaml file."""
    with open(path) as f:
        skaffold = yaml.load(f, Loader=yaml.Loader)

    services = skaffold['deploy']['helm']['releases']
    backends = [s for s in services if s['name'].startswith('substrabac-peer')]

    nodes = [NodeCfg(
        name=b['name'],
        msp_id=b['overrides']['peer']['mspID'],
        address=b['overrides']['backend']['defaultDomain'],
        user=b['overrides']['backend']['auth']['user'],
        password=b['overrides']['backend']['auth']['password'],
    ) for b in backends]

    return Settings(
        path=path,
        nodes=nodes,
    )


def load():
    """Loads settings static configuration.

    As the configuration is static and immutable, it is loaded only once from the disk.

    Returns an instance of the `Settings` class.
    """
    global _SETTINGS
    if _SETTINGS is not None:
        return _SETTINGS

    if SUBSTRAT_SKAFFOLD_FILEPATH:
        s = _load_skaffold(SUBSTRAT_SKAFFOLD_FILEPATH)
    else:
        s = _load_yaml(NETWORK_CONFIGURATION_PATH)
    _SETTINGS = s
    return _SETTINGS


# TODO that's a bad idea to expose the static configuration, it has been done to allow
#      tests parametrization but this won't work for specific tests written with more
#      more nodes

# load configuration at module load time to allow tests parametrization depending on
# network static configuration
load()


MSP_IDS = [n.msp_id for n in _SETTINGS.nodes]
