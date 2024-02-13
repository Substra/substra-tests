import time

import pytest

from substratest.factory import DEFAULT_FUNCTION_NAME
from substratest.fl_interface import FunctionCategory


def get_dockerfile(substra_tools_image: str, function_category: FunctionCategory, extra_instructions: str = "") -> str:
    return f"""
FROM {substra_tools_image}

COPY function.py .
{extra_instructions}
ENTRYPOINT ["python3", "function.py", "--function-name", "{DEFAULT_FUNCTION_NAME[function_category]}"]
"""


@pytest.mark.subprocess_skip
def test_base_substra_tools_image(factory, cfg, client, default_dataset, worker):
    """Test that an function created with the base substra-tools image works"""

    substra_tools_image = cfg.substra_tools.image_local

    function_category = FunctionCategory.simple
    dockerfile = get_dockerfile(substra_tools_image, function_category)
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
    substra_tools_image = cfg.substra_tools.image_local
    function_category = FunctionCategory.simple
    dockerfile = get_dockerfile(substra_tools_image, function_category, extra_instructions="ENV test=0\nsleep 1")
    spec = factory.create_function(function_category, dockerfile=dockerfile)
    function = client.add_function(spec)

    function = client._backend._client.get("function", function.key)
    assert function["status"] == "FUNCTION_STATUS_READY"


@pytest.mark.remote_only
def test_function_build_order(factory, cfg, client, worker):
    substra_tools_image = cfg.substra_tools.image_local
    function_category = FunctionCategory.simple

    dockerfile_1 = get_dockerfile(substra_tools_image, function_category, extra_instructions="ENV test=7\nRUN sleep 1")
    spec_1 = factory.create_function(function_category, dockerfile=dockerfile_1)
    function_1 = client.add_function(spec_1)
    dockerfile_2 = get_dockerfile(substra_tools_image, function_category, extra_instructions="ENV test=8\nRUN sleep 1")
    spec_2 = factory.create_function(function_category, dockerfile=dockerfile_2)
    function_2 = client.add_function(spec_2)
    dockerfile_3 = get_dockerfile(substra_tools_image, function_category, extra_instructions="ENV test=9\nRUN sleep 1")
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
