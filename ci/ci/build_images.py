import os
import time
import json
from typing import Dict, Tuple

from ci.config import Config, Repository, Image
from ci.call import call_output

GCR_HOST = "gcr.io"


def build_images(cfg: Config, known_host_file_path: str, run_tag: str, dir: str) -> None:
    tag = f"connect-tests-{run_tag}"
    images = {}

    print("# Queue docker image builds")
    for repo in cfg.get_repos():
        for image in repo.images:
            build_id = _build_image(cfg, tag, image, repo, known_host_file_path, dir)
            images[build_id] = (repo, image)
    print(f"{len(images)} image builds queued under the tag {tag}")
    _wait_for_builds(cfg, tag, images, known_host_file_path, dir)


def _build_image(
    cfg: Config,
    tag: str,
    image: Image,
    repo: Repository,
    known_host_file_path: str,
    dir: str
) -> str:
    config_file = os.path.join(dir, f"cloudbuild/{repo.name}.yaml")

    substitutions = {
        "GCR_HOST": GCR_HOST,
        "BUILD_TAG": tag,
        "IMAGE": image.name,
        "GIT_REPOSITORY": repo.repo_name,
        "BRANCH": repo.ref,
        "COMMIT": repo.commit,
        "KANIKO_CACHE_ENABLED": str(image.kaniko_cache).lower(),
        "KANIKO_CACHE_TTL": cfg.gcp.kaniko_cache_ttl,
        "SSH_KEY_SECRET": cfg.gcp.ssh_key_secret,
    }
    if repo == cfg.repos.tests:
        substitutions["SUBSTRA_GIT_COMMIT"] = cfg.repos.sdk.commit

    if repo == cfg.repos.connectlib:
        substitutions["SUBSTRA_GIT_COMMIT"] = cfg.repos.sdk.commit
        substitutions["CONNECTTOOLS_GIT_COMMIT"] = cfg.repos.connect_tools.commit

    cmd = (
        f"gcloud builds submit "
        f"{known_host_file_path} "
        f"--config={config_file} "
        f"--async "
        f"--project={cfg.gcp.project} "
        f"--substitutions=" + ",".join([f"_{k}={v}" for k, v in substitutions.items()])
    )

    output = call_output(cmd, print_cmd=False)

    build_id = output.split("\n")[-1].split(" ")[0]

    return build_id


def _wait_for_builds(
    cfg: Config,
    tag: str,
    images: Dict[str, Tuple[Repository, Image]],
    known_host_file_path: str,
    dir: str,
) -> None:
    print("\n# Waiting for builds to complete ...", end="")

    max_retries_per_image = 5
    retries = {}

    do_wait = True
    while do_wait:
        builds = json.loads(
            call_output(
                f'gcloud builds list --format=json --filter="tags={tag}" --project={cfg.gcp.project}',
                print_cmd=False,
            )
        )

        nb_successful = 0
        nb_abandoned = 0
        nb_retried = 0
        for b in builds:
            if b["id"] in images:
                if b["status"] == "FAILURE":
                    repo, image = images[b["id"]]
                    if retries.get(f"{repo.name}/{image.name}", 0) < max_retries_per_image:
                        retries[f"{repo.name}/{image.name}"] = retries.get(f"{repo.name}/{image.name}", 0) + 1
                        new_id = _build_image(cfg, tag, image, repo, known_host_file_path, dir)
                        images[new_id] = images.pop(b["id"])
                        nb_retried += 1
                    else:
                        nb_abandoned += 1
                elif b["status"] == "SUCCESS":
                    nb_successful += 1
                elif b["status"] in ["TIMEOUT", "CANCELLED"]:
                    nb_abandoned += 1

        do_wait = len(images) != (nb_successful + nb_abandoned)

        time.sleep(5)
        print("R" if nb_retried else ".", end="", flush=True)

    print("done.")

    if nb_abandoned:
        print("FATAL: One or more builds failed. See logs for more details")
        for build in builds:
            if build["id"] in images and build["status"] in ["TIMEOUT", "CANCELLED", "FAILURE"]:
                repo, image = images[build["id"]]
                print(
                    f"- [{image.name}]: "
                    f'https://console.cloud.google.com/cloud-build/builds/{build["id"]}?project={cfg.gcp.project}'
                )
        raise Exception("docker image build(s) failed.")
