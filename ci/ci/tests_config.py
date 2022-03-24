import tempfile

import yaml
from ci.call import call
from ci.config import Config

SUBSTRA_TESTS_CONFIG_FILEPATH = "/usr/src/app/values.yaml"


def inject_config_file(
    cfg: Config,
    namespace: str,
    substra_tests_pod: str,
    future_timeout: int,
) -> None:

    tests_config = _get_config(future_timeout)
    with tempfile.NamedTemporaryFile("w") as f:
        yaml.dump(tests_config, f)
        f.seek(0)
        call(
            f"kubectl --context {cfg.gcp.cluster.kube_context} cp -n {namespace} "
            f"{f.name} {substra_tests_pod}:{SUBSTRA_TESTS_CONFIG_FILEPATH}"
        )


def _get_config(
    future_timeout: int,
) -> object:
    return {
        "future_timeout": future_timeout,
        "options": {
            "enable_intermediate_model_removal": False,
            "enable_model_download": True,
        },
        "nodes": [
            {
                "name": "node-1",
                "msp_id": "MyOrg1MSP",
                "address": "http://backend-org-1-substra-backend-server.org-1.svc.cluster.local:8000",
                "user": "node-1",
                "password": "p@sswr0d44",
            },
            {
                "name": "node-2",
                "msp_id": "MyOrg2MSP",
                "address": "http://backend-org-2-substra-backend-server.org-2.svc.cluster.local:8000",
                "user": "node-2",
                "password": "p@sswr0d45",
            },
        ],
    }
