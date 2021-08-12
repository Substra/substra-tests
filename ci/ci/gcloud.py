#!/usr/bin/env python3
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict

from ci.config import GCPConfig
from ci.call import call, call_output


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
    return call_output("gcloud config get-value project", print_cmd=False, no_stderr=True)


def gcloud_get_auth_token() -> str:
    try:
        token = call_output(
            "gcloud auth print-access-token",
            print_cmd=False,
            no_stderr=True,
        )
    except subprocess.CalledProcessError as exc:
        raise Exception("Could not retrieve an access token, have you tried running `gcloud auth login` ?") from exc

    return token


def gcloud_test_permissions(cfg: GCPConfig) -> None:
    auth_token = gcloud_get_auth_token()

    # We validate only two of the 60 permissions required to execute this script
    # to validate that the user is authenticated.
    expected_permissions = [
        "cloudbuild.builds.create",
        "container.clusters.create",
    ]

    res = _post_request(
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
                    f"Missing required permission {permission}, " "have you tried running `gcloud auth login` ?"
                )
    else:
        raise Exception("Missing required permissions, have you tried running `gcloud auth login` ?")


def gcloud_create_cluster_async(cfg: GCPConfig) -> None:
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


def gcloud_get_kube_context(cfg: GCPConfig) -> GCPConfig:
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


def gcloud_delete_cluster(cfg: GCPConfig) -> None:
    gcloud_wait_for_cluster(cfg)
    print("# Delete cluster")
    cmd = (
        f"yes | gcloud container clusters delete {cfg.cluster.name} --zone "
        f"{cfg.cluster.zone} --project {cfg.project} --quiet"
    )
    call(cmd)


def gcloud_delete_disks(cfg: GCPConfig) -> None:
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
            call(f"gcloud compute disks delete --zone {cfg.cluster.zone} " f"--project {cfg.project} --quiet {disks}")
    except subprocess.CalledProcessError as ex:
        print("ERROR: Deletion of the GCP disks failed", ex)


def gcloud_wait_for_cluster(cfg: GCPConfig) -> None:
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


@dataclass()
class _GCPResponse:
    status: int
    body: str

    def json(self) -> Any:
        return json.loads(self.body)


def _post_request(url: str, data: Dict, headers: Dict = None) -> _GCPResponse:
    headers = {"Accept": "application/json", **headers}
    request_data = json.dumps(data).encode()
    headers["Content-Type"] = "application/json; charset=UTF-8"

    httprequest = urllib.request.Request(url, data=request_data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(httprequest) as httpresponse:
            response = _GCPResponse(
                status=httpresponse.status,
                body=httpresponse.read().decode(httpresponse.headers.get_content_charset("utf-8")),
            )
    except urllib.error.HTTPError as e:
        response = _GCPResponse(
            body=str(e.reason),
            status=e.code,
        )

    return response
