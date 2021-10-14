import os
import time
import json
from typing import Dict

from ci.config import Config, Repository
from ci.call import call_output


def build_images(cfg: Config, known_host_file_path: str, run_tag: str, dir: str) -> None:
    tag = f"connect-tests-{run_tag}"
    images = {}

    print("# Queue docker image builds")
    for repo in cfg.repos.get_all():
        for image in repo.images:
            build_id = _build_image(cfg, tag, image, repo, known_host_file_path, dir)
            images[build_id] = image
    print(f"{len(images)} queued under the tag {tag}")
    _wait_for_builds(cfg, tag, images, repo, known_host_file_path, dir)


def _build_image(cfg: Config, tag: str, image: str, repo: Repository, known_host_file_path: str, dir: str) -> str:
    config_file = os.path.join(dir, f"cloudbuild/{repo.name}.yaml")

    extra_substitutions = ""
    if repo == cfg.repos.tests:
        extra_substitutions = f",_SUBSTRA_GIT_COMMIT={cfg.repos.sdk.commit}"

    cmd = (
        f"gcloud builds submit "
        f"{known_host_file_path} "
        f"--config={config_file} "
        f"--async "
        f"--project={cfg.gcp.project} "
        f"--substitutions=_BUILD_TAG={tag},_IMAGE={image},_BRANCH={repo.ref},_COMMIT={repo.commit},"
        f"_KANIKO_CACHE_TTL={cfg.gcp.kaniko_cache_ttl},_GIT_REPOSITORY={repo.repo_name},"
        f"_SSH_KEY_SECRET={cfg.gcp.ssh_key_secret}{extra_substitutions}"
    )

    output = call_output(cmd, print_cmd=False)

    build_id = output.split("\n")[-1].split(" ")[0]

    return build_id


def _wait_for_builds(
    cfg: Config,
    tag: str,
    images: Dict[str, str],
    repo: Repository,
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
                    image = images[b["id"]]
                    if retries.get(image, 0) < max_retries_per_image:
                        retries[image] = retries.get(image, 0) + 1
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
                image = images[build["id"]]
                print(
                    f"- [{image}]: "
                    f'https://console.cloud.google.com/cloud-build/builds/{build["id"]}?project={cfg.gcp.project}'
                )
        raise Exception("docker image build(s) failed.")
