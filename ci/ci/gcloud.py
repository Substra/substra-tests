#!/usr/bin/env python3
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from typing import Dict

from ci.config import GCPConfig
from ci.call import call, call_output


SERVER_LABEL, WORKER_LABEL = "server=true", "worker=true"


def login(cfg: GCPConfig) -> None:
    print("# Log into Google Cloud")
    call(
        f"gcloud auth activate-service-account {cfg.service_account.name} "
        f"--key-file={cfg.service_account.key_dir}/{cfg.service_account.key_file}"
    )


def set_project(project: str) -> None:
    print("# Switching GCP project")
    if project is not None:
        call(f"gcloud config set project {project}")


def get_project() -> str:
    return call_output("gcloud config get-value project", print_cmd=False, no_stderr=True)


def get_auth_token() -> str:
    try:
        token = call_output(
            "gcloud auth print-access-token",
            print_cmd=False,
            no_stderr=True,
        )
    except subprocess.CalledProcessError as exc:
        raise Exception("Could not retrieve an access token, have you tried running `gcloud auth login` ?") from exc

    return token


def test_permissions(cfg: GCPConfig) -> None:
    auth_token = get_auth_token()

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


def create_cluster_async(cfg: GCPConfig) -> None:
    print("\n# Create GKE cluster")
    cmd = (
        f"gcloud container clusters create {cfg.cluster.name} "
        f"--machine-type {cfg.cluster.machine_type} "
        f"--service-account {cfg.service_account.name} "
        f"--num-nodes={cfg.cluster.nodes} "
        f"--zone={cfg.cluster.zone} "
        f"--project={cfg.project} "
        f"--enable-ip-alias "
        f"--disk-size={cfg.cluster.node_disk_size} "
        f"--disk-type={cfg.cluster.node_disk_type} "
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


def delete_all(cfg: GCPConfig) -> None:
    """Delete cluster and PVC disks"""

    # Step 1. Delete the disks
    #
    # Deleting the namespaces will automatically delete the PVC disks.
    # Deleting the cluster will automatically delete the disks of the kubernetes nodes (1 per node).
    #
    # The scheduled Google Cloud Function "cluster_cleaning_function" acts as a fallback in case something goes wrong:
    # it will periodically identify and delete the orphaned disks, if any. See the cluster_cleaning_function source
    # code and documentation for more info.
    namespaces = "org-1 org-2"
    # "foreground" deletion should maximize the chances the command returns only after everything is deleted (I think).
    call(f"kubectl --context {cfg.kube_context} delete ns {namespaces} --ignore-not-found --cascade=foreground")

    # Step 2. Delete the cluster
    _delete_cluster_async(cfg)


def wait_for_cluster(cfg: GCPConfig) -> None:
    print("# Waiting for GKE cluster to be ready ...", end="")

    while True:
        output = call_output(
            f'gcloud container clusters list --filter="name={cfg.cluster.name}" --project {cfg.project} '
            f'--zone={cfg.cluster.zone}',
            print_cmd=False,
        )

        try:
            cluster_info = list(filter(lambda l: cfg.cluster.name in l, output.splitlines()))[0]
            status = cluster_info.split(" ")[-1]
            if status not in ["RUNNING", "PROVISIONING", "RECONCILING"]:
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


def label_nodes(cfg: GCPConfig) -> None:
    print("\n# Label nodes")
    data = json.loads(call_output(
        cmd=f"kubectl --context {cfg.kube_context} get nodes -A -o json",
        print_cmd=False,
    ))
    nodes = [node["metadata"]["name"] for node in data["items"]]
    first_node, *other_nodes = nodes
    # label first node for server
    call(
        cmd=(
            f"kubectl --context {cfg.kube_context} "
            f"label nodes {first_node} {SERVER_LABEL} --overwrite"
        ),
    )
    # label other nodes for worker (or first node in case of single node)
    for node in (other_nodes or [first_node]):
        call(
            cmd=(
                f"kubectl --context {cfg.kube_context} "
                f"label nodes {node} {WORKER_LABEL} --overwrite"
            ),
        )


def print_nodes(cfg: GCPConfig):
    print("******* Backend server and worker nodes *******")
    print(call_output(
        cmd=(
            f"kubectl --context {cfg.kube_context} get pods -A "
            "-l 'app.kubernetes.io/name in (substra-backend-server,substra-backend-worker)' "
            "-o custom-columns='pod:metadata.name,node:spec.nodeName'"
        ),
        print_cmd=False,
    ))


def _delete_cluster_async(cfg: GCPConfig) -> None:
    wait_for_cluster(cfg)
    print("# Delete cluster")

    args = [
        f"--zone {cfg.cluster.zone}",
        f"--project {cfg.project}",
        "--quiet",
        "--async",
    ]

    args = " ".join(args)
    cmd = f"yes | gcloud container clusters delete {cfg.cluster.name} {args}"
    call(cmd)


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
