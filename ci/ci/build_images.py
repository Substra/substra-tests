import os
import time
from typing import List

from ci.config import Config, GCPConfig, Repository
from ci.call import call_output


def build_images(cfg: Config, known_host_file_path: str, run_tag: str, dir: str) -> None:
    tag = f"connect-tests-{run_tag}"
    images = {}

    print("# Queue docker image builds")
    for repo in cfg.repos.get_all():
        for image in repo.images:
            build_id = _build_image(cfg, tag, image, repo, known_host_file_path, dir)
            images[build_id] = image

    _wait_for_builds(cfg.gcp, tag, images)


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

    output = call_output(cmd)
    print(output)

    build_id = output.split("\n")[-1].split(" ")[0]

    return build_id


def _wait_for_builds(cfg: GCPConfig, tag: str, images: List[str]) -> None:
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
        num_failed = build_list.count("TIMEOUT") + build_list.count("CANCELLED") + build_list.count("FAIL")

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
