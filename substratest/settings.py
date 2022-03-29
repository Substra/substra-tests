"""Global settings for all tests environment."""
import os
from typing import List
from typing import Optional

import pydantic
import yaml

CURRENT_DIR = os.path.dirname(__file__)

DEFAULT_NETWORK_CONFIGURATION_PATH = os.path.join(CURRENT_DIR, "../", "values.yaml")
SUBSTRA_TESTS_CONFIG_FILEPATH = os.getenv("SUBSTRA_TESTS_CONFIG_FILEPATH", DEFAULT_NETWORK_CONFIGURATION_PATH)

DEFAULT_NETWORK_LOCAL_CONFIGURATION_PATH = os.path.join(CURRENT_DIR, "../", "local-backend-values.yaml")

MIN_NODES = 1


class NodeCfg(pydantic.BaseModel):
    name: str
    msp_id: str
    address: str
    user: Optional[str] = None
    password: Optional[str] = None
    shared_path: Optional[str] = None


class Options(pydantic.BaseModel):
    enable_intermediate_model_removal: bool
    enable_model_download: bool
    minikube: bool = False


class Settings(pydantic.BaseModel):
    path: str
    options: Options
    nodes: List[NodeCfg]

    @classmethod
    def from_yaml_file(cls, path: str) -> "Settings":
        """Load configuration from yaml file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        data["path"] = path
        return Settings.parse_obj(data)


_SETTINGS = None
_LOCAL_SETTINGS = None


def load() -> Settings:
    """Loads settings static configuration.

    As the configuration is static and immutable, it is loaded only once from the disk.

    Returns an instance of the `Settings` class.
    """
    global _SETTINGS
    if _SETTINGS is not None:
        return _SETTINGS

    s = Settings.from_yaml_file(SUBSTRA_TESTS_CONFIG_FILEPATH)
    assert len(s.nodes) >= MIN_NODES, f"not enough nodes: {len(s.nodes)}"
    _SETTINGS = s
    return _SETTINGS


def load_local_backend() -> Settings:
    """Loads settings static configuration.

    As the configuration is static and immutable, it is loaded only once from the disk.

    Returns an instance of the `Settings` class.
    """
    global _LOCAL_SETTINGS
    if _LOCAL_SETTINGS is None:
        _LOCAL_SETTINGS = Settings.from_yaml_file(DEFAULT_NETWORK_LOCAL_CONFIGURATION_PATH)
    return _LOCAL_SETTINGS


# TODO that's a bad idea to expose the static configuration, it has been done to allow
#      tests parametrization but this won't work for specific tests written with more
#      nodes

# load configuration at module load time to allow tests parametrization depending on
# network static configuration
load()

MSP_IDS = [n.msp_id for n in _SETTINGS.nodes]
HAS_SHARED_PATH = bool(_SETTINGS.nodes[0].shared_path)
IS_MINIKUBE = bool(_SETTINGS.options.minikube)
