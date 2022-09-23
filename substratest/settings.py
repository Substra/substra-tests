"""Global settings for all tests environment."""
import os
import sys
from typing import List
from typing import Optional

import pydantic
import yaml

_CURRENT_DIR = os.path.dirname(__file__)

_DEFAULT_NETWORK_CONFIGURATION_PATH = os.path.join(_CURRENT_DIR, "../", "values.yaml")
_SUBSTRA_TESTS_CONFIG_FILEPATH = os.getenv("SUBSTRA_TESTS_CONFIG_FILEPATH", _DEFAULT_NETWORK_CONFIGURATION_PATH)

_DEFAULT_SUBSTRA_TOOLS_TAG_LOCAL = (
    f"latest-nvidiacuda11.6.0-base-ubuntu20.04-python{sys.version_info.major}.{sys.version_info.minor}"
)
_DEFAULT_SUBSTRA_TOOLS_TAG_REMOTE = "latest"

_DEFAULT_SUBSTRA_TOOLS_IMAGE_REMOTE = f"substra/substra-tools:{_DEFAULT_SUBSTRA_TOOLS_TAG_REMOTE}-minimal"
_DEFAULT_SUBSTRA_TOOLS_IMAGE_LOCAL = f"ghcr.io/substra/substra-tools:{_DEFAULT_SUBSTRA_TOOLS_TAG_LOCAL}-minimal"

_DEFAULT_SUBSTRA_TOOLS_IMAGE_WORKFLOWS = f"ghcr.io/substra/substra-tools:{_DEFAULT_SUBSTRA_TOOLS_TAG_REMOTE}-workflows"

_DEFAULT_NETWORK_LOCAL_CONFIGURATION_PATH = os.path.join(_CURRENT_DIR, "../", "local-backend-values.yaml")

_MIN_ORGANIZATIONS = 1

_DEFAULT_MNIST_TRAIN_SAMPLES = 500

_DEFAULT_MNIST_TEST_SAMPLES = 200


class OrganizationCfg(pydantic.BaseModel):
    name: str
    msp_id: str
    address: str
    user: Optional[str] = None
    password: Optional[str] = None
    shared_path: Optional[str] = None


class MnistWorkflowCfg(pydantic.BaseModel):
    train_samples: int = _DEFAULT_MNIST_TRAIN_SAMPLES
    test_samples: int = _DEFAULT_MNIST_TEST_SAMPLES


class SubstraToolsCfg(pydantic.BaseModel):
    image_remote: str = _DEFAULT_SUBSTRA_TOOLS_IMAGE_REMOTE
    image_local: str = _DEFAULT_SUBSTRA_TOOLS_IMAGE_LOCAL
    image_workflows: str = _DEFAULT_SUBSTRA_TOOLS_IMAGE_WORKFLOWS


class Options(pydantic.BaseModel):
    enable_intermediate_model_removal: bool
    enable_model_download: bool
    minikube: bool = False
    future_timeout: int = 300
    future_polling_period: float = 0.5
    organization_sync_timeout: int = 300


class Settings(pydantic.BaseModel):
    _SETTINGS: Optional["Settings"] = None
    _LOCAL_SETTINGS: Optional["Settings"] = None

    path: str
    options: Options
    substra_tools: SubstraToolsCfg = SubstraToolsCfg()
    organizations: List[OrganizationCfg]
    mnist_workflow: MnistWorkflowCfg = MnistWorkflowCfg()

    @classmethod
    def _from_yaml_file(cls, path: str) -> "Settings":
        """Load configuration from yaml file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        data["path"] = path
        return Settings.parse_obj(data)

    @classmethod
    def load(cls) -> "Settings":
        """Loads settings static configuration.

        As the configuration is static and immutable, it is loaded only once from the disk.
        """
        if cls._SETTINGS is None:
            s = Settings._from_yaml_file(_SUBSTRA_TESTS_CONFIG_FILEPATH)
            assert len(s.organizations) >= _MIN_ORGANIZATIONS, f"not enough organizations: {len(s.organizations)}"
            cls._SETTINGS = s

        return cls._SETTINGS

    @classmethod
    def load_local_backend(cls) -> "Settings":
        """Loads settings static configuration.

        As the configuration is static and immutable, it is loaded only once from the disk.
        """
        if cls._LOCAL_SETTINGS is None:
            cls._LOCAL_SETTINGS = Settings._from_yaml_file(_DEFAULT_NETWORK_LOCAL_CONFIGURATION_PATH)

        return cls._LOCAL_SETTINGS
