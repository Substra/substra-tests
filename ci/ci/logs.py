import os
import subprocess

from ci.config import Config, GCPConfig, OrchestratorMode
from ci.call import call, call_output
from ci.k8s import get_single_k8s_object, get_k8s_objects
from ci.tests import get_done_frontend_tests_job


def retrieve_logs(cfg: Config, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n# Retrieve logs in {log_dir}")

    orgs = ["org-1", "org-2"]
    for org in orgs:
        ns_log_dir = _create_log_dir(log_dir, org)
        retrieve_logs_single_org(cfg.gcp, org, ns_log_dir)
        if cfg.orchestrator_mode == OrchestratorMode.DISTRIBUTED:
            retrieve_logs_orchestrator(cfg.gcp, org, ns_log_dir)

    if cfg.orchestrator_mode == OrchestratorMode.STANDALONE:
        orc_log_dir = _create_log_dir(log_dir, "orchestrator")
        retrieve_logs_orchestrator(cfg.gcp, "org-1", orc_log_dir)

    if cfg.test.frontend.enabled:
        try:
            retrieve_frontend_test_logs(cfg, log_dir)
        except Exception as e:
            print(f"Couldn't retrieve frontend logs: {e}")


def _create_log_dir(log_dir: str, org: str) -> str:
    """Builds the log directory for a specific org.

    Args:
        log_dir (str): root log directory.
        org (str): org name.

    Returns:
        str: log directory for the selected org.
    """
    ns_log_dir = os.path.join(log_dir, org)
    os.makedirs(ns_log_dir, exist_ok=True)

    return ns_log_dir


def retrieve_logs_orchestrator(cfg: GCPConfig, namespace: str, log_dir: str) -> None:
    """Retrieves the orchestrator pod logs and output the content to a file under the log_dir.

    Args:
        cfg (GCPConfig): GCP configuration object.
        namespace (str): Namespace in which the orchestrator pod is located.
        log_dir (str): Directory in which the logs are outputed.
    """
    orchestrator_pod = _get_pod_name(cfg, namespace, "app.kubernetes.io/component=server")
    orchestrator_logs_path = os.path.join(log_dir, "orchestrator")
    _retrieve_pod_logs(cfg, orchestrator_logs_path, namespace, orchestrator_pod)


def retrieve_logs_single_org(cfg: GCPConfig, namespace: str, ns_log_dir: str) -> None:
    """Retrieves the backend and worker pod logs and output the content to files under the ns_logs_dir.

    This function creates two files under the ns_log_dir, one named backend-server for the server
    logs and one named backend-worker for the worker logs.

    Args:
        cfg (GCPConfig): GCP configuration object.
        namespace (str): Namespace in which the connect-backend pods are located.
        ns_log_dir (str): Directory in which the logs files will be created.
    """
    backend_pod = _get_pod_name(cfg, namespace, "app.kubernetes.io/name=substra-backend-server")
    backend_log_path = os.path.join(ns_log_dir, "backend-server")
    _retrieve_pod_logs(cfg, backend_log_path, namespace, backend_pod)

    worker_pod = _get_pod_name(cfg, namespace, "app.kubernetes.io/name=substra-backend-worker")
    worker_log_path = os.path.join(ns_log_dir, "backend-worker")
    _retrieve_pod_logs(cfg, worker_log_path, namespace, worker_pod)


def _get_pod_name(cfg: GCPConfig, namespace: str, label_selector: str) -> str:
    """Retrieves a pod name based on a label selector.

    Args:
        cfg (GCPConfig): GCP configuration object.
        namespace (str): namespace where the pod you want is located.
        label_selector (str): label selector used to filter pods.

    Returns:
        str: a pod name.
    """
    return call_output(
        cmd=(
            f"kubectl --context {cfg.kube_context} get pod -n {namespace}"
            f" -l {label_selector} -o name"
        )
    )


def _retrieve_pod_logs(cfg: GCPConfig, log_file_path: str, namespace: str, pod_name: str):
    """Retrieves logs from a Kubernetes pod.

    Args:
        cfg (GCPConfig): GCP configuration object.
        log_file_path (str): output path of the log file.
        namespace (str): namespace of the pod you want to retrieve logs from.
        pod_name (str): pod from which you want to retrieve logs.
    """
    with open(log_file_path, "w") as f:
        try:
            call(
                cmd=f"kubectl --context {cfg.kube_context} logs -n {namespace} {pod_name}",
                stdout=f,
                print_cmd=False
            )
        except subprocess.CalledProcessError:
            print(f"Failed to retrieve logs for pod {pod_name}")


def retrieve_frontend_test_logs(cfg: Config, log_dir: str) -> None:
    job = get_done_frontend_tests_job(cfg, 2 * 60)

    namespace = job["metadata"]["namespace"]
    cypress_pods = get_k8s_objects(
        cfg,
        "pod",
        namespace,
        extra_k8s_args=f"--selector job-name={job['metadata']['name']} --sort-by '{{status.startTime}}'",
        desc="cypress pods"
    )

    frontend_pods = get_k8s_objects(
        cfg,
        "pod",
        namespace,
        lambda m: "connect-frontend" in m["name"],
        desc="frontend pods"
    )

    ns_log_dir = os.path.join(log_dir, namespace)
    os.makedirs(ns_log_dir, exist_ok=True)
    for p in frontend_pods + cypress_pods:
        _retrieve_pod_logs(cfg.gcp, os.path.join(ns_log_dir, p["metadata"]["name"]), namespace, p["metadata"]["name"])

    pod = get_single_k8s_object(
        cfg,
        "pod",
        namespace,
        lambda m: "cypress-screenshots-retriever" in m["name"],
        desc="pod with frontend screenshots"
    )
    sc_dir = os.path.join(log_dir, "cypress-screenshots")
    os.makedirs(sc_dir, exist_ok=True)
    call(f"kubectl --context {cfg.gcp.kube_context} cp org-1/{pod['metadata']['name']}:/screenshots/ {sc_dir}")
