import os
import time

import pytest

from substratest.factory import DEFAULT_FUNCTION_NAME
from substratest.fl_interface import FunctionCategory


def get_dockerfile(base_image: str, function_category: FunctionCategory, extra_instructions: str = "") -> str:
    substra_git_ref = os.getenv("SUBSTRA_GIT_REF", "main")
    return f"""
FROM {base_image}

RUN apt-get update -y && apt-get install -y git
RUN python3 -m pip install -U pip
RUN python3 -m pip install git+https://github.com/Substra/substra.git@{substra_git_ref}

COPY function.py .
{extra_instructions}
ENTRYPOINT ["python3", "function.py", "--function-name", "{DEFAULT_FUNCTION_NAME[function_category]}"]
"""


@pytest.mark.subprocess_skip
def test_base_docker_image(factory, cfg, client, default_dataset, worker):
    """Test that a function created with the base docker image works"""

    docker_image = cfg.base_docker_image

    function_category = FunctionCategory.simple
    dockerfile = get_dockerfile(docker_image, function_category)
    spec = factory.create_function(function_category, dockerfile=dockerfile)
    function = client.add_function(spec)
    spec = factory.create_traintask(
        function=function,
        inputs=default_dataset.train_data_inputs,
        worker=worker,
    )
    traintask = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    client.wait_task(traintask.key, raise_on_failure=True)


@pytest.mark.remote_only
def test_function_build_when_submitted(factory, cfg, client, worker):
    docker_image = cfg.base_docker_image
    function_category = FunctionCategory.simple
    dockerfile = get_dockerfile(docker_image, function_category, extra_instructions="ENV test=0\nRUN sleep 1")
    spec = factory.create_function(function_category, dockerfile=dockerfile)
    function = client.add_function(spec)

    function = client.wait_function(function.key, raise_on_failure=True)
    assert function.status == "FUNCTION_STATUS_READY"


@pytest.mark.remote_only
def test_function_build_order(factory, cfg, client, worker):
    docker_image = cfg.base_docker_image
    function_category = FunctionCategory.simple

    dockerfile_1 = get_dockerfile(docker_image, function_category, extra_instructions="ENV test=7\nRUN sleep 1")
    spec_1 = factory.create_function(function_category, dockerfile=dockerfile_1)
    function_1 = client.add_function(spec_1)
    dockerfile_2 = get_dockerfile(docker_image, function_category, extra_instructions="ENV test=8\nRUN sleep 1")
    spec_2 = factory.create_function(function_category, dockerfile=dockerfile_2)
    function_2 = client.add_function(spec_2)
    dockerfile_3 = get_dockerfile(docker_image, function_category, extra_instructions="ENV test=9\nRUN sleep 1")
    spec_3 = factory.create_function(function_category, dockerfile=dockerfile_3)
    function_3 = client.add_function(spec_3)

    # Cannot use `get_function` as status is not yet exposed through substra SDK
    function_getter = client._backend._client.get
    function_1 = function_getter("function", function_1.key)
    function_2 = function_getter("function", function_2.key)
    function_3 = function_getter("function", function_3.key)

    assert function_2["status"] == "FUNCTION_STATUS_WAITING"
    assert function_3["status"] == "FUNCTION_STATUS_WAITING"

    while function_1["status"] != "FUNCTION_STATUS_READY":
        time.sleep(1)
        function_1 = function_getter("function", function_1["key"])

    function_2 = function_getter("function", function_2["key"])
    function_3 = function_getter("function", function_3["key"])
    assert function_2["status"] == "FUNCTION_STATUS_BUILDING"
    assert function_3["status"] == "FUNCTION_STATUS_WAITING"


@pytest.mark.remote_only
def test_function_cancelled_cp(factory, cfg, client, worker, default_dataset):
    """
    Test that if you cancelled a CP and a function is linked to tasks that are oinly in CP canceled or faiuled,
    the function will be canceled
    """
    docker_image = cfg.base_docker_image
    function_category = FunctionCategory.simple

    # We create 2 functions, so when the second one should start building, the CP is already been canceled
    dockerfile_wait = get_dockerfile(
        docker_image, function_category, extra_instructions="ENV test=test_function_cancelled_cp_wait\nRUN sleep 5"
    )
    dockerfile_test = get_dockerfile(
        docker_image, function_category, extra_instructions="ENV test=test_function_cancelled_cp_test"
    )

    function_wait_spec = factory.create_function(function_category, dockerfile=dockerfile_wait)
    function_test_spec = factory.create_function(function_category, dockerfile=dockerfile_test)

    inputs = default_dataset.opener_input + default_dataset.train_data_sample_inputs
    cp = factory.create_compute_plan()
    function_wait = client.add_function(function_wait_spec)
    cp.create_traintask(function=function_wait, worker=worker, inputs=inputs)
    function_test = client.add_function(function_test_spec)
    cp.create_traintask(function=function_test, worker=worker, inputs=inputs)
    client.add_compute_plan(cp)

    client.cancel_compute_plan(cp.key)
    cp = client.wait_compute_plan(cp.key, raise_on_failure=False, timeout=cfg.options.future_timeout)
    assert cp.status == "PLAN_STATUS_CANCELED"

    # Check that `function_wait` (built first) reaches status DONE or CANCELED
    function_wait = client.wait_function(function_wait.key, raise_on_failure=False, timeout=cfg.options.future_timeout)
    assert function_wait.status != "FUNCTION_STATUS_FAILED"
    function_test = client.wait_function(function_test.key, raise_on_failure=False, timeout=cfg.options.future_timeout)
    assert function_test.status == "FUNCTION_STATUS_CANCELED"
