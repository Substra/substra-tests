import subprocess
import time

from ci.config import Config
from ci.call import call, call_output
from ci.gcloud import gcloud_get_auth_token


def run_tests(cfg: Config):
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
        token = gcloud_get_auth_token()
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
            f"env SUBSTRA_TESTS_FUTURE_TIMEOUT={cfg.tests_future_timeout} "
            f"make {cfg.tests_make_command} PARALLELISM={cfg.tests_concurrency}"
        )
        return True
    except subprocess.CalledProcessError:
        print(f"FATAL: `make {cfg.tests_make_command}` completed with a non-zero exit code. Did some test(s) fail?")
        raise
