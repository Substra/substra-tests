import pytest

from substratest.factory import DEFAULT_FUNCTION_NAME
from substratest.fl_interface import FunctionCategory


@pytest.mark.subprocess_skip
def test_base_substra_tools_image(factory, cfg, client, default_dataset, worker):
    """Test that an function created with the base substra-tools image works"""

    substra_tools_image = cfg.substra_tools.image_local

    function_category = FunctionCategory.simple

    dockerfile = f"""
FROM {substra_tools_image}

COPY function.py .

ENTRYPOINT ["python3", "function.py", "--function-name", "{DEFAULT_FUNCTION_NAME[function_category]}"]
"""
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
