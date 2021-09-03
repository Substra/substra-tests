#!/usr/bin/env python3
import argparse

from ci import gcloud
from ci.config import Config


def arg_parse() -> Config:
    config = Config()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "cluster-name",
        help="The name of the cluster",
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

    args = vars(parser.parse_args())

    config.gcp.cluster.name = args["cluster-name"]
    config.gcp.cluster.pvc_volume_name_prefix = config.gcp.cluster.name[:18].lower()
    config.gcp.service_account.key_dir = args["gcp_keys_directory"]
    config.gcp.service_account.key_file = args["gcp_key_filename"]

    return config


def main() -> None:
    config = arg_parse()

    current_project = gcloud.get_project()
    gcloud.set_project(config.gcp.project)
    gcloud.test_permissions(config.gcp)

    gcloud.delete_cluster(config.gcp)
    gcloud.delete_disks(config.gcp)
    gcloud.set_project(current_project)


if __name__ == "__main__":
    main()
