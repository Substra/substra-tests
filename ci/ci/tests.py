import subprocess
import time
from traceback import print_exc
from typing import Dict


from ci.config import Config
from ci.call import call, call_output
from ci.deploy import deploy
from ci import gcloud
from ci.k8s import get_single_k8s_object, NoK8sObjectsMatchError

# TODO: when splitting deployment scripts and test (cf https://app.asana.com/0/1201044465977493/1201519666278453/f)
# the functions run_sdk, run_connectlib (and run_frontend if possible) should be refactored.


def run_sdk(cfg: Config):
    print("# Wait for the connect-tests pod to be ready")
    substra_tests_pod = call_output(
        f"kubectl --context {cfg.gcp.kube_context} get pods -n connect-tests | grep connect-tests"
    ).split(" ")[0]

    try:
        call(
            f"kubectl --context {cfg.gcp.kube_context} wait pod/{substra_tests_pod} "
            f"-n connect-tests --for=condition=ready --timeout=590s"
        )
    except subprocess.CalledProcessError:
        print(
            "ERROR: Timeout while waiting for the connect-tests pod. "
            'This means the `connect-backend-server` pods never reached the "ready" state.'
        )

    try:
        time.sleep(5)
        token = gcloud.get_auth_token()
        call(
            f"kubectl --context {cfg.gcp.kube_context} exec {substra_tests_pod} -n connect-tests -- "
            f"docker login -u oauth2accesstoken -p {token} https://gcr.io",
            secrets=[token],
        )
    except subprocess.CalledProcessError:
        print("FATAL: could not log in onto the image registry")
        raise

    print("\n# Run tests")

    try:
        # Run the tests on the remote and local backend
        call(
            f"kubectl --context {cfg.gcp.kube_context} exec {substra_tests_pod} -n connect-tests -- "
            f"env SUBSTRA_TESTS_FUTURE_TIMEOUT={cfg.test.sdk.future_timeout} "
            f"make {cfg.test.sdk.make_command} "
            f"PARALLELISM={cfg.test.sdk.concurrency} "
        )
        return True
    except subprocess.CalledProcessError:
        print(f"FATAL: `make {cfg.test.sdk.make_command}` completed with a non-zero exit code. Did some test(s) fail?")
        raise


class FrontendTestsException(Exception):
    pass


def run_frontend(cfg: Config, source_dir):
    print("\n# Run frontend tests")
    # this deployment actually runs the tests (k8s job)
    deploy(cfg, cfg.repos.frontend, source_dir, False, "automated-e2e-tests")

    job = get_done_frontend_tests_job(cfg, 30 * 60)

    if job["status"]["conditions"][0]["type"] == "Complete":
        return  # Success !
    raise FrontendTestsException("Frontend tests failed")


def get_done_frontend_tests_job(cfg: Config, timeout_s: int) -> Dict:
    seconds_between_tries = 60
    time_elapsed = 0
    last_exc = None
    while time_elapsed < timeout_s:
        time.sleep(seconds_between_tries)
        try:
            job = get_single_k8s_object(
                cfg,
                "job",
                cfg.test.frontend.namespace,
                lambda m: "connect-frontend-tests" in m["name"],
                desc="frontend tests job"
            )
            try:
                if job["status"]["conditions"][0]["status"] == "True":
                    # https://github.com/kubernetes/kubernetes/issues/68712
                    return job
            except (KeyError, IndexError) as e:
                last_exc = e
        except NoK8sObjectsMatchError as e:
            last_exc = e
        time_elapsed += seconds_between_tries
    if last_exc:
        raise Exception("Couldn't get connect-frontend-tests Job") from last_exc
    raise Exception("Couldn't get connect-frontend-tests Job")


def run_connectlib(cfg: Config):
    print("# Wait for the connect-tests pod to be ready")
    connectlib_tests_pod = call_output(
        f"kubectl --context {cfg.gcp.kube_context} get pods -n connect-tests | grep connectlib"
    ).split(" ")[0]

    try:
        call(
            f"kubectl --context {cfg.gcp.kube_context} wait pod/{connectlib_tests_pod} "
            f"-n connect-tests --for=condition=ready --timeout=590s"
        )
    except subprocess.CalledProcessError:
        print(
            "ERROR: Timeout while waiting for the connect-tests pod. "
            'This means the `connect-backend-server` pods never reached the "ready" state.'
        )

    try:
        time.sleep(5)
        token = gcloud.get_auth_token()
        call(
            f"kubectl --context {cfg.gcp.kube_context} exec {connectlib_tests_pod} -n connect-tests -- "
            f"docker login -u oauth2accesstoken -p {token} https://gcr.io",
            secrets=[token],
        )
    except subprocess.CalledProcessError:
        print("FATAL: could not log in onto the image registry")
        raise

    print("\n# Run tests")

    try:
        # Run the tests on the remote and local backend
        call(
            f"kubectl --context {cfg.gcp.kube_context} exec {connectlib_tests_pod} -n connect-tests -- "
            f"env SUBSTRA_TESTS_FUTURE_TIMEOUT={cfg.test.connectlib.future_timeout} "
            f"make {cfg.test.connectlib.make_command}"
        )
        return True
    except subprocess.CalledProcessError:
        print(
            f"FATAL: `make {cfg.test.connectlib.make_command}` "
            "completed with a non-zero exit code. Did some test(s) fail?"
        )
        raise


def run(cfg: Config, source_dir) -> Dict[str, bool]:
    SDK_LABEL = "sdk"
    FRONTEND_LABEL = "frontend"
    CONNECTLIB_LABEL = "connectlib"

    test_passed: Dict[str, bool] = {
        SDK_LABEL: None,
        FRONTEND_LABEL: None,
        CONNECTLIB_LABEL: None
    }

    cfg.test.frontend

    # SDK and frontend tests
    try:
        if cfg.test.sdk.make_command:
            test_passed[SDK_LABEL] = False
            run_sdk(cfg)
            test_passed[SDK_LABEL] = True

            # frontend tests depend on backend tests
            if cfg.test.frontend.enabled:
                test_passed[FRONTEND_LABEL] = False
                run_frontend(cfg, source_dir)
                test_passed[FRONTEND_LABEL] = True

    except Exception:
        print_exc()

    # Connectlib tests
    try:
        if cfg.test.connectlib.enabled:
            test_passed[CONNECTLIB_LABEL] = False
            run_connectlib(cfg)
            test_passed[CONNECTLIB_LABEL] = True

    except Exception:
        print_exc()

    return test_passed
