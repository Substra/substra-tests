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
    f"latest-nvidiacuda11.8.0-base-ubuntu22.04-python{sys.version_info.major}.{sys.version_info.minor}"
)
_DEFAULT_SUBSTRA_TOOLS_TAG_REMOTE = "latest"

_DEFAULT_SUBSTRA_TOOLS_IMAGE_REMOTE = f"substra/substra-tools:{_DEFAULT_SUBSTRA_TOOLS_TAG_REMOTE}"
_DEFAULT_SUBSTRA_TOOLS_IMAGE_LOCAL = f"ghcr.io/substra/substra-tools:{_DEFAULT_SUBSTRA_TOOLS_TAG_LOCAL}"

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


class Options(pydantic.BaseModel):
    enable_intermediate_model_removal: bool
    enable_model_download: bool
    minikube: bool = False
    future_timeout: int = 300
    future_polling_period: float = 0.5
    organization_sync_timeout: int = 300


class Settings(pydantic.BaseModel):
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
        return Settings.model_validate(data)


class PytestConfig(pydantic.BaseModel):
    static_settings: Optional[Settings] = None
    local_settings: Optional[Settings] = None

    def load(self) -> Settings:
        """Loads settings static configuration.

        As the configuration is static and immutable, it is loaded only once from the disk.
        """
        if self.static_settings is None:
            s = Settings._from_yaml_file(_SUBSTRA_TESTS_CONFIG_FILEPATH)
            assert len(s.organizations) >= _MIN_ORGANIZATIONS, f"not enough organizations: {len(s.organizations)}"
            self.static_settings = s

        return self.static_settings

    def load_local_backend(self) -> Settings:
        """Loads settings static configuration.

        As the configuration is static and immutable, it is loaded only once from the disk.
        """
        if self.local_settings is None:
            self.local_settings = Settings._from_yaml_file(_DEFAULT_NETWORK_LOCAL_CONFIGURATION_PATH)

        return self.local_settings
