import os
from dataclasses import dataclass, field
from typing import List
from enum import Enum


class OrchestratorMode(Enum):
    DISTRIBUTED = "distributed"
    STANDALONE = "standalone"

    def __str__(self):
        return str(self.value)


@dataclass()
class Repository:
    name: str = ""
    repo_name: str = ""
    commit: str = ""
    skaffold_artifact: str = ""
    skaffold_profile: str = None
    # ref can be eiher a branch or a tag
    ref: str = "master"
    # In order to build them we need a list of the docker images in the repo
    images: List[str] = field(default_factory=list)

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
    zone: str = "europe-west4-a"


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
    nodes: int = 1


@dataclass()
class GitConfig:
    git_token: str = ""
    clone_method: str = "ssh"
    use_token: bool = False


@dataclass()
class Repositories:
    tests: Repository = Repository(
        name="tests", repo_name="owkin/connect-tests.git", images=["connect-tests"],
    )
    backend: Repository = Repository(
        name="backend",
        repo_name="owkin/connect-backend.git",
        images=["connect-backend"],
        skaffold_artifact="substra-backend",
    )
    sdk: Repository = Repository(
        name="sdk", repo_name="owkin/substra.git",
    )
    hlf_k8s: Repository = Repository(
        name="hlf_k8s", repo_name="owkin/connect-hlf-k8s.git", images=["fabric-tools", "fabric-peer"],
    )
    orchestrator: Repository = Repository(
        name="orchestrator",
        repo_name="owkin/orchestrator.git",
        images=["chaincode", "forwarder", "server", "rabbitmq-operator"],
    )

    def get_all(self) -> List[Repository]:
        return [self.tests, self.backend, self.sdk, self.hlf_k8s, self.orchestrator]


@dataclass()
class Config:
    gcp: GCPConfig = GCPConfig()
    git: GitConfig = GitConfig()
    repos: Repositories = Repositories()
    backend_celery_concurrency: int = 4
    tests_concurrency: int = 5
    tests_future_timeout: int = 400
    tests_make_command: str = "test-ci"
    orchestrator_mode: OrchestratorMode = OrchestratorMode.DISTRIBUTED

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
            f"HLF_K8S_BRANCH\t\t\t= {self.repos.hlf_k8s.ref}\n"
            f"ORCHESTRATOR_BRANCH\t\t= {self.repos.orchestrator.ref}\n"
            f"KANIKO_CACHE_TTL\t\t= {self.gcp.kaniko_cache_ttl}\n"
            f"BACKEND_CELERY_CONCURRENCY\t= {self.backend_celery_concurrency}\n"
            f"TESTS_CONCURRENCY\t\t= {self.tests_concurrency}\n"
            f"TESTS_FUTURE_TIMEOUT\t\t= {self.tests_future_timeout}\n"
            f"TESTS_MAKE_COMMAND\t\t= {self.tests_make_command}\n"
            f"ORCHESTRATOR_MODE\t\t= {self.orchestrator_mode}\n"
        )
        if self.is_ci_runner:
            out += f"KEYS_DIR\t\t\t= {self.gcp.service_account.key_dir}\n"

        return out
