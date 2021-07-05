#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import shutil
import string
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, TextIO

import yaml

CLUSTER_NAME_ALLOWED_PREFIX = "connect-tests"
DIR = os.path.dirname(os.path.realpath(__file__))
RUN_TAG = "".join(random.choice(string.ascii_letters + "0123456789") for _ in range(10))
SOURCE_DIR = os.path.realpath(os.path.join(DIR, "src", RUN_TAG))
LOG_DIR = os.path.realpath(os.path.join(DIR, "logs", RUN_TAG))


@dataclass()
class Response:
    status: int
    body: str

    def json(self) -> Any:
        return json.loads(self.body)


@dataclass()
class Cluster:
    name: str = ""
    pvc_volume_name_prefix: str = ""
    machine_type: str = "n1-standard-8"
    kube_context: str = ""
    # Zone must be specific (e.g. "europe-west1-b" and not "europe-west1")
    # or else several kubernetes nodes will be created instead of just one,
    # which can lead to pod/volume affinity issues at runtime.
    zone: str = "europe-west4-a"


@dataclass()
class ServiceAccount:
    name: str = "e2e-tests@connect-314908.iam.gserviceaccount.com"
    key_file: str = "connect-314908-3902714646d9.json"
    key_dir: str = os.path.realpath(os.path.join(os.getenv("HOME"), ".local/"))


@dataclass()
class GCPConfig:
    cluster: Cluster = Cluster()
    service_account: ServiceAccount = ServiceAccount()
    project: str = "connect-314908"
    kaniko_cache_ttl: str = "168h"  # A week
    ssh_key_secret: str = "projects/101637030044/secrets/connect-e2e-deploy-key/versions/2"


@dataclass()
class Repository:
    name: str = ""
    repo_name: str = ""
    commit: str = ""
    skaffold_artifact: str = ""
    # ref can be eiher a branch or a tag
    ref: str = "master"
    # In order to build them we need a list of the docker images in the repo
    images: List[str] = field(default_factory=list)

    @property
    def dirname(self) -> str:
        return os.path.join(SOURCE_DIR, self.name)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Repository):
            return self.repo_name == o.repo_name
        return False


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
    chaincode: Repository = Repository(
        name="chaincode",
        repo_name="owkin/connect-chaincode.git",
        images=["connect-chaincode"],
    )
    hlf_k8s: Repository = Repository(
        name="hlf_k8s",
        repo_name="owkin/connect-hlf-k8s.git",
        images=["fabric-tools", "fabric-peer"],
    )

    def get_all(self) -> List[Repository]:
        return [self.tests, self.backend, self.sdk, self.chaincode, self.hlf_k8s]


@dataclass()
class Config:
    gcp: GCPConfig = GCPConfig()
    git: GitConfig = GitConfig()
    repos: Repositories = Repositories()
    backend_celery_concurrency: int = 4
    tests_concurrency: int = 5
    tests_future_timeout: int = 400

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
            f"CHAINCODE_BRANCH\t\t= {self.repos.chaincode.ref}\n"
            f"HLF_K8S_BRANCH\t\t\t= {self.repos.hlf_k8s.ref}\n"
            f"KANIKO_CACHE_TTL\t\t= {self.gcp.kaniko_cache_ttl}\n"
            f"BACKEND_CELERY_CONCURRENCY\t= {self.backend_celery_concurrency}\n"
            f"TESTS_CONCURRENCY\t\t= {self.tests_concurrency}\n"
            f"TESTS_FUTURE_TIMEOUT\t\t= {self.tests_future_timeout}\n"
        )

        if self.is_ci_runner:
            out += f"KEYS_DIR\t\t\t= {self.gcp.service_account.key_dir}\n"

        return out


def call(
    cmd: str, print_cmd: bool = True, secrets: List[str] = None, stdout: TextIO = None,
) -> int:
    if not secrets:
        secrets = []

    if print_cmd:
        printed_cmd = f"+ {cmd}"

        for secret in secrets:
            printed_cmd = printed_cmd.replace(secret, "****")

        print(printed_cmd)
    return subprocess.check_call([cmd], shell=True, stdout=stdout, stderr=stdout)


def call_output(cmd: str, print_cmd: bool = True, no_stderr: bool = False) -> str:
    if print_cmd:
        print(f"+ {cmd}")

    if no_stderr:
        stderr = subprocess.DEVNULL
    else:
        stderr = subprocess.STDOUT

    res = subprocess.check_output([cmd], shell=True, stderr=stderr)
    return res.decode().strip()


def cluster_name_format(value: str) -> str:
    """
    Validate the --cluster-name argument
    The cluster name must start with 'connect-tests'.
    This is to ensure the cluster gets picked up by the stale cluster deletion script.
    """

    if not value.startswith(CLUSTER_NAME_ALLOWED_PREFIX):
        raise argparse.ArgumentTypeError(
            f'Invalid cluster name "{value}". '
            f'The cluster name must start with "{CLUSTER_NAME_ALLOWED_PREFIX}".'
        )

    if len(value) > 35:
        raise argparse.ArgumentTypeError(
            f'Invalid cluster name "{value}". '
            f"The cluster name must not be longer than 35 characters."
        )

    return value


def arg_parse() -> Config:
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine-type",
        type=str,
        default=config.gcp.cluster.machine_type,
        help="The GKE machine type to use",
    )
    parser.add_argument(
        "-N",
        "--cluster-name",
        type=cluster_name_format,
        default=CLUSTER_NAME_ALLOWED_PREFIX,
        help="The prefix name if the GKE kubernetes cluster to create",
    )
    parser.add_argument(
        "-K",
        "--gcp-keys-directory",
        type=str,
        default=config.gcp.service_account.key_dir,
        help="The path to a folder containing the GKE service account credentials",
    )
    parser.add_argument(
        "--gcp-key-filename",
        type=str,
        default=config.gcp.service_account.key_file,
        help="the filename of the service account key",
    )
    parser.add_argument(
        "--e2e-tests",
        "--substra-tests",
        "--connect-tests",
        type=str,
        default=config.repos.tests.ref,
        help="e2e tests repo branch or tag",
        metavar="GIT_BRANCH",
    )
    parser.add_argument(
        "--sdk",
        "--substra",
        type=str,
        default=config.repos.sdk.ref,
        help="sdk/client branch or tag",
        metavar="GIT_BRANCH",
    )
    parser.add_argument(
        "--backend",
        "--connect-backend",
        "--substra-backend",
        type=str,
        default=config.repos.backend.ref,
        help="backend branch or tag",
        metavar="GIT_BRANCH",
    )
    parser.add_argument(
        "--chaincode",
        "--connect-chaincode",
        "--substra-chaincode",
        type=str,
        default=config.repos.chaincode.ref,
        help="chaincode branch or tag",
        metavar="GIT_BRANCH",
    )
    parser.add_argument(
        "--hlf-k8s",
        type=str,
        default=config.repos.hlf_k8s.ref,
        help="hlf-k8s branch or tag",
        metavar="GIT_BRANCH",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Use this option to disable kaniko caching",
    )
    parser.add_argument(
        "--backend-celery-concurrency",
        type=int,
        default=config.backend_celery_concurrency,
        help="The backend worker task concurrency",
    )
    parser.add_argument(
        "--tests-concurrency",
        type=int,
        default=config.tests_concurrency,
        help="The number of parallel test runners",
    )
    parser.add_argument(
        "--tests-future-timeout",
        type=int,
        default=config.tests_future_timeout,
        help="In e2e-tests, the number of seconds to wait for a training task to complete",
    )
    parser.add_argument(
        "--git-clone-method",
        type=str,
        default=config.git.clone_method,
        choices=["ssh", "https"],
        help="Method used to clone repositories",
    )
    parser.add_argument(
        "--git-use-token",
        action="store_true",
        help="Use a private access token stored in the env var GIT_TOKEN",
    )

    args = vars(parser.parse_args())

    # GCP Config
    cluster_name = args["cluster_name"]
    # Add RUN_TAG to cluster name to make it non-deterministic in case of retry
    cluster_name += f"-{RUN_TAG[:40-len(cluster_name)-1]}"
    config.gcp.cluster.name = (
        cluster_name.lower()
    )  # Make it lower for gcloud compatibility
    # Only the 18 first characters are taken into account
    config.gcp.cluster.pvc_volume_name_prefix = cluster_name[:18].lower()
    config.gcp.cluster.machine_type = args["machine_type"]
    config.gcp.service_account.key_dir = args["gcp_keys_directory"]
    config.gcp.service_account.key_file = args["gcp_key_filename"]
    if args["no_cache"]:
        config.gcp.kaniko_cache_ttl = "-1h"

    # Repo config
    config.repos.tests.ref = args["e2e_tests"]
    config.repos.sdk.ref = args["sdk"]
    config.repos.backend.ref = args["backend"]
    config.repos.chaincode.ref = args["chaincode"]
    config.repos.hlf_k8s.ref = args["hlf_k8s"]

    # Git config
    config.git.clone_method = args["git_clone_method"]
    config.git.use_token = args["git_use_token"]
    if config.git.use_token is True:
        config.git.git_token = os.getenv("GIT_TOKEN")

    # Tests config
    config.backend_celery_concurrency = args["backend_celery_concurrency"]
    config.tests_concurrency = args["tests_concurrency"]
    config.tests_future_timeout = args["tests_future_timeout"]

    print("💃💃💃\n")
    print(config)
    return config


def gcloud_login(cfg: GCPConfig) -> None:
    print("# Log into Google Cloud")
    call(
        f"gcloud auth activate-service-account {cfg.service_account.name} "
        f"--key-file={cfg.service_account.key_dir}/{cfg.service_account.key_file}"
    )


def gcloud_set_project(project: str) -> None:
    print("# Switching GCP project")
    if project is not None:
        call(f"gcloud config set project {project}")


def gcloud_get_project() -> str:
    return call_output(
        "gcloud config get-value project", print_cmd=False, no_stderr=True
    )


def gcloud_get_auth_token() -> str:
    try:
        token = call_output(
            "gcloud auth print-access-token", print_cmd=False, no_stderr=True,
        )
    except subprocess.CalledProcessError as exc:
        raise Exception(
            "Could not retrieve an access token, have you tried running `gcloud auth login` ?"
        ) from exc

    return token


def gcloud_test_permissions(cfg: GCPConfig) -> None:
    auth_token = gcloud_get_auth_token()

    # We validate only two of the 60 permissions required to execute this script
    # to validate that the user is authenticated.
    expected_permissions = [
        "cloudbuild.builds.create",
        "container.clusters.create",
    ]

    res = post_request(
        url=f"https://cloudresourcemanager.googleapis.com/v1/projects/{cfg.project}:testIamPermissions",
        headers={"Authorization": f"Bearer {auth_token}"},
        data={"permissions": expected_permissions},
    )

    if res.status != 200:
        return Exception("Failed to query GCP IAM")

    perms = res.json().get("permissions", None)

    if perms:
        for permission in expected_permissions:
            if permission not in perms:
                raise Exception(
                    f"Missing required permission {permission}, "
                    "have you tried running `gcloud auth login` ?"
                )
    else:
        raise Exception(
            "Missing required permissions, have you tried running `gcloud auth login` ?"
        )


def post_request(url: str, data: Dict, headers: Dict = None) -> Response:
    headers = {"Accept": "application/json", **headers}
    request_data = json.dumps(data).encode()
    headers["Content-Type"] = "application/json; charset=UTF-8"

    httprequest = urllib.request.Request(
        url, data=request_data, headers=headers, method="POST"
    )

    try:
        with urllib.request.urlopen(httprequest) as httpresponse:
            response = Response(
                status=httpresponse.status,
                body=httpresponse.read().decode(
                    httpresponse.headers.get_content_charset("utf-8")
                ),
            )
    except urllib.error.HTTPError as e:
        response = Response(body=str(e.reason), status=e.code,)

    return response


def create_cluster_async(cfg: GCPConfig) -> None:
    print("\n# Create GKE cluster")
    cmd = (
        f"gcloud container clusters create {cfg.cluster.name} "
        f"--machine-type {cfg.cluster.machine_type} "
        f"--service-account {cfg.service_account.name} "
        f"--num-nodes=1 "
        f"--zone={cfg.cluster.zone} "
        f"--project={cfg.project} "
        f"--enable-ip-alias "
        f"--no-enable-autoupgrade "
        f"--enable-network-policy "
        f"--async"
    )
    call(cmd)


def get_kube_context(cfg: GCPConfig) -> GCPConfig:
    old_ctx = None
    print("\n# Fetch kubernetes context")

    try:
        if call_output("kubectl config get-contexts --no-headers"):
            old_ctx = call_output("kubectl config current-context")
    except subprocess.CalledProcessError:
        pass

    call(
        f"gcloud container clusters get-credentials {cfg.cluster.name} "
        f"--zone {cfg.cluster.zone} --project {cfg.project}"
    )

    if old_ctx is not None:
        call(f"kubectl config use-context {old_ctx}")  # Restore old context

    cfg.kube_context = f"gke_{cfg.project}_{cfg.cluster.zone}_{cfg.cluster.name}"

    return cfg


def delete_cluster(cfg: GCPConfig) -> None:
    wait_for_cluster(cfg)
    print("# Delete cluster")
    cmd = (
        f"yes | gcloud container clusters delete {cfg.cluster.name} --zone "
        f"{cfg.cluster.zone} --project {cfg.project} --quiet"
    )
    call(cmd)


def delete_disks(cfg: GCPConfig) -> None:
    try:
        # the filter AND is implicit
        disk_filter = f"name~^gke-{cfg.cluster.pvc_volume_name_prefix}-pvc-.* zone~{cfg.cluster.zone}"
        cmd = (
            f'gcloud compute disks list --project {cfg.project} --format="table(name)" '
            f'--filter="{disk_filter}" | sed 1d'
        )
        disks = call_output(cmd)
        disks = disks.replace("\n", " ")
        if disks:
            call(
                f"gcloud compute disks delete --zone {cfg.cluster.zone} "
                f"--project {cfg.project} --quiet {disks}"
            )
    except subprocess.CalledProcessError as ex:
        print("ERROR: Deletion of the GCP disks failed", ex)


def wait_for_cluster(cfg: GCPConfig) -> None:
    print("# Waiting for GKE cluster to be ready ...", end="")

    while True:
        output = call_output(
            f'gcloud container clusters list --filter="name={cfg.cluster.name}" --project {cfg.project}',
            print_cmd=False,
        )

        try:
            status = output.split("\n")[1].split(" ")[-1]
            if status not in ["RUNNING", "PROVISIONING"]:
                raise Exception(f"Unknown status {status}")
        except Exception as ex:
            print("\nFATAL: Error retrieving cluster status. Output was:")
            print(output)
            raise ex

        if status == "RUNNING":
            print("done.")
            break

        print(".", end="", flush=True)
        time.sleep(5)


def setup_helm() -> None:
    print("\n# Setup Helm")
    call("helm repo add stable https://charts.helm.sh/stable")
    call("helm repo add owkin https://owkin.github.io/charts")
    call("helm repo add bitnami https://charts.bitnami.com/bitnami")


def clone_repos(cfg: Config) -> Config:
    if os.path.exists(SOURCE_DIR):
        shutil.rmtree(SOURCE_DIR)

    os.makedirs(SOURCE_DIR)

    print(f"\n# Clone repos in {SOURCE_DIR}")

    for repo in cfg.repos.get_all():
        repo = clone_repository(cfg.git, repo)

    print("\nCommit hashes:\n")
    for repo in cfg.repos.get_all():
        print(f"- {repo.repo_name}: \t{repo.commit}")
    print("\n")

    return cfg


def build_repo_url(cfg: GitConfig, repo: Repository) -> str:
    url = ""
    if cfg.clone_method == "https":
        url = "https://{}github.com/{}"
        if cfg.use_token:
            url = url.format(f"{cfg.git_token}:x-oauth-basic@", repo.repo_name)
        else:
            url = url.format("", repo.repo_name)
    elif cfg.clone_method == "ssh":
        url = f"git@github.com:{repo.repo_name}"
    return url


def clone_repository(cfg: GitConfig, repo: Repository) -> Repository:
    extra_args = {}
    if cfg.use_token:
        extra_args["print_cmd"] = False

    url = build_repo_url(cfg, repo)

    call(
        f'git clone -q --depth 1 {url} --branch "{repo.ref}" {repo.dirname}',
        **extra_args,
    )

    if not repo.commit:
        try:
            commit = call_output(
                f"git --git-dir={repo.dirname}/.git rev-parse origin/{repo.ref}"
            )
        except subprocess.CalledProcessError:
            # It didn't work with a branch name. Try with a tag name.
            commit = call_output(
                f"git --git-dir={repo.dirname}/.git rev-list {repo.ref}"
            )
        repo.commit = commit

    return repo


def build_images(cfg: Config) -> None:
    tag = f"connect-tests-{RUN_TAG}"
    images = {}

    print("# Queue docker image builds")
    for repo in cfg.repos.get_all():
        for image in repo.images:
            build_id = build_image(cfg=cfg, tag=tag, image=image, repo=repo)
            images[build_id] = image

    wait_for_builds(cfg.gcp, tag, images)


def build_image(cfg: Config, tag: str, image: str, repo: Repository) -> str:
    config_file = os.path.join(DIR, f"cloudbuild/{repo.name}.yaml")

    extra_substitutions = ""
    if repo == cfg.repos.tests:
        extra_substitutions = f",_SUBSTRA_GIT_COMMIT={cfg.repos.sdk.commit}"

    cmd = (
        f"gcloud builds submit "
        "cloudbuild/known_host.tgz "
        f"--config={config_file} "
        f"--async "
        f"--project={cfg.gcp.project} "
        f"--substitutions=_BUILD_TAG={tag},_IMAGE={image},_BRANCH={repo.ref},_COMMIT={repo.commit},"
        f"_KANIKO_CACHE_TTL={cfg.gcp.kaniko_cache_ttl},_GIT_REPOSITORY={repo.repo_name},"
        f"_SSH_KEY_SECRET={cfg.gcp.ssh_key_secret}{extra_substitutions}"
    )

    output = call_output(cmd)
    print(output)

    build_id = output.split("\n")[-1].split(" ")[0]

    return build_id


def wait_for_builds(cfg: GCPConfig, tag: str, images: List[str]) -> None:
    print("\n# Waiting for builds to complete ...", end="")
    do_wait = True
    while do_wait:
        build_list = call_output(
            f'gcloud builds list --filter="tags={tag}" --project={cfg.project}',
            print_cmd=False,
        )

        builds = build_list.split("\n")[1:]

        num_builds = len(builds)
        num_success = build_list.count("SUCCESS")
        num_failed = (
            build_list.count("TIMEOUT") +
            build_list.count("CANCELLED") +
            build_list.count("FAIL")
        )

        do_wait = num_builds != (num_success + num_failed)

        time.sleep(5)
        print(".", end="", flush=True)

    print("done.")

    if num_failed:
        print("FATAL: One or more builds failed. See logs for more details")
        for build in builds:
            if "TIMEOUT" in build or "CANCELLED" in build or "FAIL" in build:
                build_id = build.split(" ")[0]
                image = images[build_id]
                print(
                    f"- [{image}]: "
                    f"https://console.cloud.google.com/cloud-build/builds/{build_id}?project={cfg.project}"
                )
        raise Exception("docker image build(s) failed.")


def deploy_all(cfg: Config) -> None:
    print("\n# Deploy helm charts")

    for repo in cfg.repos.get_all():
        # Chaincode does not need to be deployed
        if repo in [cfg.repos.chaincode, cfg.repos.sdk]:
            continue

        wait = (
            repo != cfg.repos.hlf_k8s
        )  # don't wait for hlf-k8s deployment to complete
        deploy(cfg, repo, wait)


def deploy(cfg: Config, repo: Repository, wait=True) -> None:
    artifacts_file = create_build_artifacts(cfg, repo)
    skaffold_file = patch_skaffold_file(cfg, repo)

    path = os.path.dirname(skaffold_file)

    if repo == cfg.repos.hlf_k8s:
        call(
            f"KUBE_CONTEXT={cfg.gcp.kube_context} {path}/examples/dev-secrets.sh create"
        )

    call(
        f"cd {path} && skaffold deploy --kube-context={cfg.gcp.kube_context} "
        f'-f=skaffold.yaml -a={artifacts_file} --status-check={"true" if wait else "false"}'
    )


def create_build_artifacts(cfg: Config, repo: Repository) -> str:
    # Gcloud Build artifacts
    artifacts_file = os.path.join(repo.dirname, "tags.json")

    with open(artifacts_file, "w") as file:
        tags = {"builds": []}

        for image in repo.images:

            tag = f"eu.gcr.io/{cfg.gcp.project}/{image}:ci-{repo.commit}"

            if image == "connect-tests":
                tag += f"-{cfg.repos.sdk.commit}"

            ref = repo.skaffold_artifact if repo.skaffold_artifact else image

            tags["builds"].append({"imageName": f"substrafoundation/{ref}", "tag": tag})

            print(f"Created build artifact for {tag}")

        json.dump(tags, file)

    return artifacts_file


def patch_skaffold_file(cfg: Config, repo: Repository) -> str:

    skaffold_file = os.path.join(repo.dirname, "skaffold.yaml")

    with open(skaffold_file) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    values_files = []

    for release in data["deploy"]["helm"]["releases"]:
        if release["chartPath"].startswith("charts/"):
            release["chartPath"] = os.path.join(repo.dirname, release["chartPath"])
        if "valuesFiles" in release:
            values_files.extend(release["valuesFiles"])

    with open(skaffold_file, "w") as file:
        yaml.dump(data, file)

    for values_file in values_files:
        patch_values_file(cfg, repo, os.path.join(repo.dirname, values_file))
    return skaffold_file


def patch_values_file(cfg: Config, repo: Repository, value_file: str) -> None:
    with open(value_file) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    if repo == cfg.repos.backend:
        data["celeryworker"]["concurrency"] = cfg.backend_celery_concurrency
        data["backend"]["kaniko"][
            "dockerConfigSecretName"
        ] = ""  # remove docker-config secret
    if repo == cfg.repos.hlf_k8s:
        if "chaincodes" in data:
            data["chaincodes"][0]["image"][
                "repository"
            ] = f"eu.gcr.io/{cfg.gcp.project}/connect-chaincode"
            data["chaincodes"][0]["image"]["tag"] = f"ci-{cfg.repos.chaincode.commit}"

        # remove docker-config secret
        if (
            "fabric-tools" in data and
            "pullImageSecret" in data["fabric-tools"]["image"]
        ):
            del data["fabric-tools"]["image"]["pullImageSecret"]
        if (
            "image" in data["hlf-peer"] and
            "pullImageSecret" in data["hlf-peer"]["image"]
        ):
            del data["hlf-peer"]["image"]["pullImageSecret"]
        if "chaincodes" in data:
            if "pullImageSecret" in data["chaincodes"][0]["image"]:
                del data["chaincodes"][0]["image"]["pullImageSecret"]
            if "pullImageSecret" in data["chaincodes"][1]["image"]:
                del data["chaincodes"][1]["image"]["pullImageSecret"]

    with open(value_file, "w") as file:
        yaml.dump(data, file)


def retrieve_logs(cfg: GCPConfig) -> None:
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)

    os.makedirs(LOG_DIR)

    print(f"\n# Retrieve logs in {LOG_DIR}")

    orgs = ["org-1", "org-2"]
    for org in orgs:
        retrieve_logs_single_org(cfg, org)


def retrieve_logs_single_org(cfg: GCPConfig, namespace: str) -> None:
    ns_log_dir = os.path.join(LOG_DIR, namespace)
    if os.path.exists(ns_log_dir):
        shutil.rmtree(ns_log_dir)
    os.makedirs(ns_log_dir)

    backend_pod = call_output(
        cmd=(
            f"kubectl --context {cfg.kube_context} get pod -n {namespace}"
            " -l app.kubernetes.io/name=substra-backend-server -o name"
        )
    )

    backend_log_path = os.path.join(ns_log_dir, "backend-server")
    with open(backend_log_path, "w") as f:
        try:
            call(
                cmd=f"kubectl --context {cfg.kube_context} logs -n {namespace} {backend_pod}",
                stdout=f,
            )
        except subprocess.CalledProcessError:
            # If we can't retrieve the logs of this pod we still want to get the logs of the others
            print(f"Could not retrieve logs for server pod {backend_pod}")

    worker_pod = call_output(
        cmd=(
            f"kubectl --context {cfg.kube_context} get pod -n {namespace}"
            " -l app.kubernetes.io/name=substra-backend-worker -o name"
        )
    )

    worker_log_path = os.path.join(ns_log_dir, "backend-worker")
    with open(worker_log_path, "w") as f:
        try:
            call(
                cmd=f"kubectl --context {cfg.kube_context} logs -n {namespace} {worker_pod}",
                stdout=f,
            )
        except subprocess.CalledProcessError:
            print(f"Could not retrieve logs for worker pod {worker_pod}")


def run_tests(cfg: Config) -> bool:
    print("# Wait for the connect-tests pod to be ready")
    substra_tests_pod = call_output(
        f"kubectl --context {cfg.gcp.kube_context} get pods -n connect-tests | grep connect-tests"
    ).split(" ")[0]

    try:
        call(
            f"kubectl --context {cfg.gcp.kube_context} wait pod/{substra_tests_pod} "
            f"-n connect-tests --for=condition=ready --timeout=590s"
        )
    except subprocess.CalledProcessError:
        print(
            "ERROR: Timeout while waiting for the connect-tests pod. "
            'This means the `connect-backend-server` pods never reached the "ready" state.'
        )

    try:
        time.sleep(5)
        token = gcloud_get_auth_token()
        call(
            f"kubectl --context {cfg.gcp.kube_context} exec {substra_tests_pod} -n connect-tests -- "
            f"docker login -u oauth2accesstoken -p {token} https://gcr.io",
            secrets=[token],
        )
    except subprocess.CalledProcessError:
        print("FATAL: could not log in onto the image registry")
        return False

    print("\n# Run tests")

    try:
        # Run the tests on the remote and local backend
        call(
            f"kubectl --context {cfg.gcp.kube_context} exec {substra_tests_pod} -n connect-tests -- "
            f"env SUBSTRA_TESTS_FUTURE_TIMEOUT={cfg.tests_future_timeout} "
            f"make test-remote PARALLELISM={cfg.tests_concurrency}"
        )
        return True
    except subprocess.CalledProcessError:
        print(
            "FATAL: `make test-remote` completed with a non-zero exit code. Did some test(s) fail?"
        )
        return False


def main() -> None:
    is_success = False
    config = arg_parse()
    current_project = None
    permissions_validated = False
    app_deployed = False

    try:
        if config.is_ci_runner:
            gcloud_login(config.gcp)
        else:
            current_project = gcloud_get_project()
            gcloud_set_project(config.gcp.project)
            gcloud_test_permissions(config.gcp)
        permissions_validated = True
        create_cluster_async(config.gcp)
        config = clone_repos(config)
        build_images(config)
        wait_for_cluster(config.gcp)
        config.gcp = get_kube_context(config.gcp)
        setup_helm()
        deploy_all(config)
        app_deployed = True
        is_success = run_tests(config)
        print("Completed test run:")
        print(config)
        # We delete the log dir here because we want to keep the log dir if something happens
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)

    except Exception as ex:
        print(f"FATAL: {ex}")
        logging.exception(ex)
        if app_deployed:
            retrieve_logs(config.gcp)
        is_success = False

    finally:
        print("\n# Perform final teardown")
        if os.path.exists(SOURCE_DIR):
            shutil.rmtree(SOURCE_DIR)
        if permissions_validated:
            delete_cluster(config.gcp)
            delete_disks(config.gcp)
            gcloud_set_project(current_project)
    sys.exit(0 if is_success else 1)


if __name__ == "__main__":
    main()
