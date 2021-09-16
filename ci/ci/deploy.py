import json
import os

from ci.config import Config, OrchestratorMode, Repository
from ci.call import call

import yaml


def deploy_all(cfg: Config, source_dir: str) -> None:
    print("\n# Deploy helm charts")

    for repo in cfg.repos.get_all():
        if repo == cfg.repos.sdk:
            continue
        if cfg.orchestrator_mode == OrchestratorMode.STANDALONE and repo == cfg.repos.hlf_k8s:
            continue

        wait = repo != cfg.repos.hlf_k8s  # don't wait for hlf-k8s deployment to complete
        _deploy(cfg, repo, source_dir, wait)


def _deploy(cfg: Config, repo: Repository, source_dir: str, wait=True) -> None:
    artifacts_file = _create_build_artifacts(cfg, repo, source_dir)
    skaffold_file = _patch_skaffold_file(cfg, repo, source_dir)

    path = os.path.dirname(skaffold_file)

    if repo == cfg.repos.hlf_k8s:
        call(f"KUBE_CONTEXT={cfg.gcp.kube_context} {path}/examples/dev-secrets.sh create")

    skaffold_profile = ""
    if repo.skaffold_profile:
        skaffold_profile = f"--profile {repo.skaffold_profile}"

    call(
        f"cd {path} && skaffold deploy --kube-context={cfg.gcp.kube_context} "
        f'-f=skaffold.yaml -a={artifacts_file} --status-check={"true" if wait else "false"} {skaffold_profile}'
    )


def _create_build_artifacts(cfg: Config, repo: Repository, source_dir: str) -> str:
    # Gcloud Build artifacts

    artifacts_file = os.path.join(source_dir, repo.name, "tags.json")

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


def _patch_skaffold_file(cfg: Config, repo: Repository, source_dir: str) -> str:

    repo_dir = os.path.join(source_dir, repo.name)

    skaffold_file = os.path.join(repo_dir, "skaffold.yaml")

    with open(skaffold_file) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    values_files = []

    for release in data["deploy"]["helm"]["releases"]:
        if release["chartPath"].startswith("charts/"):
            release["chartPath"] = os.path.join(repo_dir, release["chartPath"])
        if "valuesFiles" in release:
            values_files.extend(release["valuesFiles"])

    with open(skaffold_file, "w") as file:
        yaml.dump(data, file)

    for values_file in values_files:
        _patch_values_file(cfg, repo, os.path.join(repo_dir, values_file))
    return skaffold_file


def _patch_values_file(cfg: Config, repo: Repository, value_file: str) -> None:
    with open(value_file) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    if repo == cfg.repos.backend:
        data["celeryworker"]["replicaCount"] = max(1, cfg.gcp.nodes - 1)
        data["celeryworker"]["concurrency"] = cfg.backend_celery_concurrency
        data["backend"]["kaniko"]["dockerConfigSecretName"] = ""  # remove docker-config secret
        for elt in data["registry"]["prepopulate"]:
            elt["dockerConfigSecretName"] = ""  # remove docker-config secret
    if repo == cfg.repos.hlf_k8s:
        if "chaincodes" in data:
            data["chaincodes"][0]["image"]["repository"] = f"eu.gcr.io/{cfg.gcp.project}/orchestrator-chaincode"
            data["chaincodes"][0]["image"]["tag"] = f"ci-{cfg.repos.orchestrator.commit}"

            data["chaincodes"][0]["init"]["image"][
                "repository"] = f"eu.gcr.io/{cfg.gcp.project}/orchestrator-chaincode-init"
            data["chaincodes"][0]["init"]["image"]["tag"] = f"ci-{cfg.repos.orchestrator.commit}"

        # remove docker-config secret
        if "fabric-tools" in data and "pullImageSecret" in data["fabric-tools"]["image"]:
            del data["fabric-tools"]["image"]["pullImageSecret"]
        if "image" in data["hlf-peer"] and "pullImageSecret" in data["hlf-peer"]["image"]:
            del data["hlf-peer"]["image"]["pullImageSecret"]
        if "chaincodes" in data:
            if "pullImageSecret" in data["chaincodes"][0]["image"]:
                del data["chaincodes"][0]["image"]["pullImageSecret"]
            if "pullImageSecret" in data["chaincodes"][1]["image"]:
                del data["chaincodes"][1]["image"]["pullImageSecret"]

    with open(value_file, "w") as file:
        yaml.dump(data, file)
