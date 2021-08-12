import os
import shutil
import subprocess

from ci.config import GCPConfig
from ci.call import call, call_output


def retrieve_logs(cfg: GCPConfig, log_dir: str) -> None:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    os.makedirs(log_dir)

    print(f"\n# Retrieve logs in {log_dir}")

    orgs = ["org-1", "org-2"]
    for org in orgs:
        retrieve_logs_single_org(cfg, org, log_dir)


def retrieve_logs_single_org(cfg: GCPConfig, namespace: str, log_dir: str) -> None:
    ns_log_dir = os.path.join(log_dir, namespace)
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
