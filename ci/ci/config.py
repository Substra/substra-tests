import os
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import make_dataclass
from enum import Enum
from typing import List
from typing import Optional

_CONNECT_TEST_REGISTRY = "eu.gcr.io/connect-314908"


class OrchestratorMode(Enum):
    DISTRIBUTED = "distributed"
    STANDALONE = "standalone"

    def __str__(self):
        return str(self.value)


@dataclass
class Image:
    name: str
    repo_subdir: str = ""
    kaniko_cache: bool = True
    registry: str = "substrafoundation"
    skaffold_artifact: str = ""


@dataclass()
class Repository:
    name: str = ""
    repo_name: str = ""
    commit: str = ""
    skaffold_profile: str = None
    skaffold_dir: str = ""
    skaffold_filename: str = "skaffold.yaml"
    # ref can be eiher a branch or a tag
    ref: str = "master"
    # In order to build them we need a list of the docker images in the repo
    images: List[Image] = field(default_factory=list)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Repository):
            return self.repo_name == o.repo_name
        return False


@dataclass()
class GCPConfigCluster:
    name: str = ""
    pvc_volume_name_prefix: str = ""
    machine_type: str = "n1-standard-16"
    kube_context: str = ""
    # Zone must be specific (e.g. "europe-west1-b" and not "europe-west1")
    # or else several kubernetes nodes will be created instead of just one,
    # which can lead to pod/volume affinity issues at runtime.
    zone: str = "us-east1-b"
    node_disk_type = "pd-ssd"
    node_disk_size = "100GB"
    nodes: int = 1


@dataclass()
class GCPConfigServiceAccount:
    name: str = "e2e-tests@connect-314908.iam.gserviceaccount.com"
    key_file: str = "connect-314908-3902714646d9.json"
    key_dir: str = os.path.realpath(os.path.join(os.getenv("HOME"), ".local/"))


@dataclass()
class GCPConfig:
    cluster: GCPConfigCluster = GCPConfigCluster()
    service_account: GCPConfigServiceAccount = GCPConfigServiceAccount()
    project: str = "connect-314908"
    kaniko_cache_ttl: str = "168h"  # A week
    ssh_key_secret: str = "projects/101637030044/secrets/connect-e2e-deploy-key/versions/2"
    no_cleanup: bool = False
    create_cluster: bool = True


@dataclass()
class GitConfig:
    git_token: str = ""
    clone_method: str = "ssh"
    use_token: bool = False


repos = [
    Repository(
        name="tests",
        repo_name="owkin/connect-tests.git",
        images=[Image("connect-tests")],
    ),
    Repository(
        name="backend",
        repo_name="owkin/connect-backend.git",
        images=[Image("connect-backend", skaffold_artifact="substra-backend"), Image("metrics-exporter")],
    ),
    Repository(
        name="frontend",
        repo_name="owkin/connect-frontend.git",
        images=[
            Image(
                "connect-frontend",
                # We need to use another registry here because the connect-frontend skaffold file uses
                # "eu.gcr.io/connect-314908/connect-frontend" and not "substra-front/connect-frontend"
                # in the build.artifacts section.
                kaniko_cache=False,  # breaks build process if enabled
                registry=_CONNECT_TEST_REGISTRY,
            ),
            Image(
                "connect-frontend-tests",
                repo_subdir="automated-e2e-tests",
                registry=_CONNECT_TEST_REGISTRY,
            ),
        ],
    ),
    Repository(
        name="connect_tools",
        repo_name="owkin/connect-tools.git",
        ref="main",
    ),
    Repository(
        name="connectlib",
        repo_name="owkin/connectlib.git",
        images=[Image("connectlib")],
        ref="main",
    ),
    Repository(
        name="sdk",
        repo_name="owkin/substra.git",
        ref="main",
    ),
    Repository(
        name="hlf_k8s",
        repo_name="owkin/connect-hlf-k8s.git",
        images=[Image("fabric-tools"), Image("fabric-peer")],
        # use 2-orgs-policy-any instead of 2-orgs-policy-any-no-ca provided with root skaffold file
        # the aim is to test also hlf-ca certificates generation in distributed mode
        skaffold_dir="examples/2-orgs-policy-any/",
    ),
    Repository(
        name="orchestrator",
        repo_name="owkin/orchestrator.git",
        ref="main",
        images=[
            Image("orchestrator-chaincode"),
            Image("orchestrator-chaincode-init"),
            Image("orchestrator-forwarder"),
            Image("orchestrator-server"),
            Image("orchestrator-rabbitmq-operator"),
        ],
    ),
]

Repositories = make_dataclass(
    "Repositories",
    [(r.name, Repository, r) for r in repos],
    namespace={"get_all": lambda self: [f.default for f in fields(self)]},
)


@dataclass
class SdkTestConfig:
    concurrency: int = 5
    future_timeout: int = 400
    make_command: str = "test-ci"


@dataclass
class FrontendTestConfig:
    namespace: str = "org-1"  # this is set in connect-frontend/automated-e2e-tests/skaffold.yaml
    enabled: bool = False


@dataclass
class ConnectlibTestConfig:
    future_timeout: int = 600
    make_command: str = "test-ci"  # This is set in connectlib/Makefile
    enabled: bool = False


@dataclass
class TestConfig:
    sdk: SdkTestConfig = SdkTestConfig()
    frontend: FrontendTestConfig = FrontendTestConfig()
    connectlib: ConnectlibTestConfig = ConnectlibTestConfig()


@dataclass()
class Config:
    gcp: GCPConfig = GCPConfig()
    git: GitConfig = GitConfig()
    repos: Repositories = Repositories()
    backend_celery_concurrency: int = 4
    orchestrator_mode: OrchestratorMode = OrchestratorMode.STANDALONE
    test: TestConfig = TestConfig()
    write_summary_to_file: Optional[str] = None

    @property
    def is_ci_runner(self):
        # In a GH action the CI env variable is always set to `true`
        ci = os.environ.get("CI", default="false")
        return ci == "true"

    def __str__(self):
        out = (
            f"CLUSTER_MACHINE_TYPE\t\t= {self.gcp.cluster.machine_type}\n"
            f"CLUSTER_NAME\t\t\t= {self.gcp.cluster.name}\n"
            f"E2E_TESTS_BRANCH\t\t= {self.repos.tests.ref}\n"
            f"SDK_BRANCH\t\t\t= {self.repos.sdk.ref}\n"
            f"BACKEND_BRANCH\t\t\t= {self.repos.backend.ref}\n"
            f"FRONTEND_BRANCH\t\t\t= {self.repos.frontend.ref}\n"
            f"CONNECT_TOOLS_BRANCH\t\t= {self.repos.connect_tools.ref}\n"
            f"CONNECTLIB_BRANCH\t\t= {self.repos.connectlib.ref}\n"
            f"HLF_K8S_BRANCH\t\t\t= {self.repos.hlf_k8s.ref}\n"
            f"ORCHESTRATOR_BRANCH\t\t= {self.repos.orchestrator.ref}\n"
            f"KANIKO_CACHE_TTL\t\t= {self.gcp.kaniko_cache_ttl}\n"
            f"BACKEND_CELERY_CONCURRENCY\t= {self.backend_celery_concurrency}\n"
            f"SDK_TESTS_CONCURRENCY\t\t= {self.test.sdk.concurrency}\n"
            f"SDK_TESTS_FUTURE_TIMEOUT\t= {self.test.sdk.future_timeout}\n"
            f"SDK_TESTS_MAKE_COMMAND\t\t= {self.test.sdk.make_command}\n"
            f"ORCHESTRATOR_MODE\t\t= {self.orchestrator_mode}\n"
        )
        if self.is_ci_runner:
            out += f"KEYS_DIR\t\t\t= {self.gcp.service_account.key_dir}\n"

        return out

    def get_repos(self) -> List[Repository]:
        res = []

        if self.orchestrator_mode == OrchestratorMode.DISTRIBUTED:
            res.append(self.repos.hlf_k8s)

        res.append(self.repos.orchestrator)
        res.append(self.repos.backend)
        res.append(self.repos.sdk)
        res.append(self.repos.tests)

        if self.test.frontend.enabled:
            res.append(self.repos.frontend)

        if self.test.connectlib.enabled:
            res.append(self.repos.connect_tools)
            res.append(self.repos.connectlib)

        return res
