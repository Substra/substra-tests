#!/usr/bin/env python3
import argparse
import logging
import os
import random
import shutil
import string
import sys

from ci import gcloud
from ci.config import Config
from ci.logs import retrieve_logs
from ci.deploy import deploy_all
from ci.helm import setup_helm
from ci.git import clone_repos
from ci.build_images import build_images
from ci.run_tests import run_tests


CLUSTER_NAME_ALLOWED_PREFIX = "connect-tests"
DIR = os.path.dirname(os.path.realpath(__file__))
RUN_TAG = "".join(random.choice(string.ascii_letters + "0123456789") for _ in range(10))
SOURCE_DIR = os.path.realpath(os.path.join(DIR, "src", RUN_TAG))
LOG_DIR = os.path.realpath(os.path.join(DIR, "logs", RUN_TAG))
KNOWN_HOST_FILE_PATH = os.path.join(DIR, "cloudbuild", "known_host.tgz")


def cluster_name_format(value: str) -> str:
    """
    Validate the --cluster-name argument
    The cluster name must start with 'connect-tests'.
    This is to ensure the cluster gets picked up by the stale cluster deletion script.
    """

    if not value.startswith(CLUSTER_NAME_ALLOWED_PREFIX):
        raise argparse.ArgumentTypeError(
            f'Invalid cluster name "{value}". ' f'The cluster name must start with "{CLUSTER_NAME_ALLOWED_PREFIX}".'
        )

    if len(value) > 35:
        raise argparse.ArgumentTypeError(
            f'Invalid cluster name "{value}". ' f"The cluster name must not be longer than 35 characters."
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
        "--no-cleanup",
        action="store_true",
        help="Do not delete the GKE k8s cluster nor volumes at the end of tests",
    )
    parser.add_argument(
        "--use-cluster",
        help="Skip cluster creation, and instead connect to the cluster of this name",
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
        "--tests-make-command",
        type=str,
        default=config.tests_make_command,
        help="Override the make command to execute the tests, If set to \"\", no test will be run.",
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
    parser.add_argument(
        "--nodes",
        type=int,
        default=config.gcp.nodes,
        help=(
            "Number of cluster nodes (default value is 1). "
            "If nodes > 1, then worker's replicas = nodes - 1."
        )
    )

    args = vars(parser.parse_args())

    config.gcp.create_cluster = not args["use_cluster"]
    if config.gcp.create_cluster:
        cluster_name = args["cluster_name"]
        # Add RUN_TAG to cluster name to make it non-deterministic in case of retry
        cluster_name += f"-{RUN_TAG[:40-len(cluster_name)-1]}"
        config.gcp.cluster.name = cluster_name.lower()  # Make it lower for gcloud compatibility
    else:
        config.gcp.cluster.name = args["use_cluster"]

    # Only the 18 first characters are taken into account
    config.gcp.cluster.pvc_volume_name_prefix = config.gcp.cluster.name[:18].lower()
    config.gcp.cluster.machine_type = args["machine_type"]
    config.gcp.no_cleanup = args["no_cleanup"]

    config.gcp.service_account.key_dir = args["gcp_keys_directory"]
    config.gcp.service_account.key_file = args["gcp_key_filename"]
    if args["no_cache"]:
        config.gcp.kaniko_cache_ttl = "-1h"
    config.gcp.nodes = args["nodes"]

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
    config.tests_make_command = args["tests_make_command"]

    print("💃💃💃\n")
    print(config)
    return config


def main() -> None:
    is_success = False
    config = arg_parse()
    current_project = None
    permissions_validated = False
    app_deployed = False

    try:
        if config.is_ci_runner:
            gcloud.login(config.gcp)
        else:
            current_project = gcloud.get_project()
            gcloud.set_project(config.gcp.project)
            gcloud.test_permissions(config.gcp)
        permissions_validated = True
        if config.gcp.create_cluster:
            gcloud.create_cluster_async(config.gcp)
        config = clone_repos(config, SOURCE_DIR)
        build_images(config, KNOWN_HOST_FILE_PATH, RUN_TAG, DIR)
        gcloud.wait_for_cluster(config.gcp)
        config.gcp = gcloud.get_kube_context(config.gcp)
        setup_helm()
        gcloud.label_nodes(config.gcp)
        deploy_all(config, SOURCE_DIR)
        app_deployed = True
        if config.gcp.nodes > 1:
            gcloud.print_nodes(config.gcp)
        if config.tests_make_command:
            run_tests(config)
        print("Completed:")
        print(config)
        is_success = True

    except Exception as ex:
        print(f"FATAL: {ex}")
        logging.exception(ex)
        if app_deployed:
            retrieve_logs(config.gcp, LOG_DIR)
        is_success = False

    finally:
        if os.path.exists(SOURCE_DIR):
            shutil.rmtree(SOURCE_DIR)

        if config.gcp.no_cleanup:
            print(
                f"Skipping teardown of cluster {config.gcp.cluster.name}. \n",
                "⚠️💸 DON'T FORGET TO RUN cleanup.py WHEN YOU'RE DONE 💸⚠️"
            )
        else:
            print("\n# Perform final teardown")
            if permissions_validated:
                gcloud.delete_cluster(config.gcp)
                gcloud.delete_disks(config.gcp)
                gcloud.set_project(current_project)

    sys.exit(0 if is_success else 1)


if __name__ == "__main__":
    main()
