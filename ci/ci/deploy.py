import json
import os
from typing import List

import yaml

from ci.config import Config, Repository
from ci.call import call


def deploy_all(cfg: Config, source_dir: str) -> None:
    print("\n# Deploy helm charts")

    _create_namespaces(cfg, ["org-1", "org-2"])

    for repo in cfg.get_repos():
        if repo == cfg.repos.sdk:
            continue

        # For some projects don't wait for the deployment to complete
        wait = repo not in [cfg.repos.hlf_k8s, cfg.repos.orchestrator]
        deploy(cfg, repo, source_dir, wait)


def _create_namespaces(cfg: Config, namespaces: List[str]) -> None:
    for namespace in namespaces:
        call(f"kubectl --context {cfg.gcp.kube_context} create namespace {namespace}")


def deploy(cfg: Config, repo: Repository, source_dir: str, wait=True, repo_subdir: str = "") -> None:
    artifacts_file = _create_build_artifacts(cfg, repo, source_dir, repo_subdir)
    skaffold_file = _patch_skaffold_file(cfg, repo, source_dir, repo_subdir)

    path = os.path.dirname(skaffold_file)

    skaffold_profile = ""
    if repo.skaffold_profile:
        skaffold_profile = f"--profile {repo.skaffold_profile}"

    call(
        f"cd {path} && skaffold deploy --kube-context={cfg.gcp.kube_context} "
        f'-f=skaffold.yaml -a={artifacts_file} --status-check={"true" if wait else "false"} {skaffold_profile}'
    )


def _create_build_artifacts(cfg: Config, repo: Repository, source_dir: str, repo_subdir: str = "") -> str:
    # Gcloud Build artifacts

    artifacts_file = os.path.join(source_dir, repo.name, repo_subdir, "tags.json")
    images = [i for i in repo.images if i.repo_subdir == repo_subdir]

    with open(artifacts_file, "w") as file:
        tags = {"builds": []}

        for image in images:

            tag = f"eu.gcr.io/{cfg.gcp.project}/{image.name}:ci-{repo.commit}"

            if image.name == "connect-tests":
                tag += f"-{cfg.repos.sdk.commit}"

            ref = repo.skaffold_artifact or image.name

            name = f"{image.registry}/{ref}"
            tags["builds"].append({"imageName": name, "tag": tag})

            print(f"Created build artifact for {tag}")

        json.dump(tags, file)

    return artifacts_file


def _patch_skaffold_file(cfg: Config, repo: Repository, source_dir: str, repo_subdir: str = "") -> str:
    # this code will break if the release names are modified
    # in the skaffold conf of the backend or frontend repos

    repo_dir = os.path.join(source_dir, repo.name, repo_subdir)

    skaffold_file = os.path.join(repo_dir, repo.skaffold_dir, repo.skaffold_filename)

    with open(skaffold_file) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    values_files = []

    for release in data["deploy"]["helm"]["releases"]:
        # use 2-orgs-policy-any instead of 2-orgs-policy-any-no-ca provided with root skaffold file
        # which means that chartPath is not properly defined like the one in the root dir of hlf-k8s
        # the aim is to test also hlf-ca certificates generation in distributed mode
        if (release.get("chartPath", "").startswith("charts/") or
           release.get("chartPath", "").startswith("../../charts/")):
            release["chartPath"] = os.path.join(repo_dir, release["chartPath"].replace("../../", ""))
        if "valuesFiles" in release:
            values_files.extend([(release, vf) for vf in release["valuesFiles"]])
        if repo == cfg.repos.frontend:
            if "setValues" not in release:
                release["setValues"] = {}
            if repo_subdir == "automated-e2e-tests":
                release["setValues"]["cypress.config.baseUrl"] = (
                    f'http://frontend-{release["namespace"]}-connect-frontend'
                    f'.{release["namespace"]}.svc.cluster.local'
                )
                release["setValues"]["cypress.config.env.BACKEND_API_URL"] = (
                    f'http://backend-{release["namespace"]}-substra-backend-server'
                    f'.{release["namespace"]}.svc.cluster.local:8000'
                )
                release["setValues"]["cypress.screenshotsPvc.enabled"] = True
                release["setValues"]["cypress.screenshotsPvc.retrieverEnabled"] = True
            else:
                release["setValues"]["api.url"] = (
                    f'http://backend-{release["namespace"]}-substra-backend-server'
                    f'.{release["namespace"]}.svc.cluster.local:8000'
                )

    with open(skaffold_file, "w") as file:
        yaml.dump(data, file)

    for (release, values_file) in values_files:
        _patch_values_file(
            cfg,
            repo,
            os.path.join(repo_dir, repo.skaffold_dir, values_file),
            release
        )
    return skaffold_file


def _patch_values_file(cfg: Config, repo: Repository, value_file: str, release: dict) -> None:
    with open(value_file) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    if repo == cfg.repos.backend:
        data["worker"]["replicaCount"] = cfg.gcp.nodes
        data["worker"]["concurrency"] = cfg.backend_celery_concurrency
        data["kaniko"]["dockerConfigSecretName"] = ""  # remove docker-config secret
        data["server"]["commonHostDomain"] = "cluster.local"
        for elt in data["containerRegistry"]["prepopulate"]:
            elt["dockerConfigSecretName"] = ""  # remove docker-config secret

        if "extraEnv" not in data:
            data["extraEnv"] = []

        data["server"]["defaultDomain"] = (
            f'http://backend-{release["namespace"]}-substra-backend-server'
            f'.{release["namespace"]}.svc.cluster.local:8000'
        )
        allowed_hosts = [
            f'.{release["namespace"]}',
            f'.{release["namespace"]}.svc.cluster.local',
        ]
        data["config"]["ALLOWED_HOSTS"] = json.dumps(allowed_hosts)
        allowed_cors_origins = [
            f'http://frontend-{release["namespace"]}-connect-frontend.{release["namespace"]}.svc.cluster.local'
        ]
        data["config"]["CORS_ORIGIN_WHITELIST"] = json.dumps(allowed_cors_origins)

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
