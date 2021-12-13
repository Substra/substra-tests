import json
from typing import List, Dict, Callable

from ci.config import Config
from ci.call import call_output


class NoK8sObjectsMatchError(Exception):
    pass


class TooManyK8sObjectsMatchError(Exception):
    pass


def get_k8s_objects(
    cfg: Config,
    type: str,
    namespace: str,
    metadata_predicate: Callable[[Dict], bool] = lambda m: True,
    extra_k8s_args: str = "",
    desc=""
) -> List:
    objects = json.loads(
        call_output(f"kubectl --context {cfg.gcp.kube_context} get {type} -n {namespace} -o json " +
                    extra_k8s_args, print_cmd=False)
    )["items"]
    objects = [j for j in objects if metadata_predicate(j["metadata"])]

    if len(objects) < 1:
        raise NoK8sObjectsMatchError(desc)
    return objects


def get_single_k8s_object(
    cfg: Config,
    type: str,
    namespace: str,
    metadata_predicate: Callable[[Dict], bool] = lambda m: True,
    extra_k8s_args: str = "",
    desc=""
) -> Dict:
    objects = get_k8s_objects(cfg, type, namespace, metadata_predicate, extra_k8s_args, desc)
    if len(objects) > 1:
        raise TooManyK8sObjectsMatchError(desc)
    return objects[0]
