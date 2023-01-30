import pytest

from substratest.factory import DEFAULT_ALGO_METHOD_NAME
from substratest.fl_interface import AlgoCategory


@pytest.mark.subprocess_skip
def test_base_substra_tools_image(factory, cfg, client, default_dataset, worker):
    """Test that an algo created with the base substra-tools image instead of the minimal works"""

    suffix = "-minimal"
    if cfg.substra_tools.image_local.endswith(suffix):
        substra_tools_image = cfg.substra_tools.image_local[: -len(suffix)]
    else:
        substra_tools_image = cfg.substra_tools.image_local

    algo_category = AlgoCategory.simple

    dockerfile = f"""
FROM {substra_tools_image}

COPY algo.py .

ENTRYPOINT ["python3", "algo.py", "--function-name", "{DEFAULT_ALGO_METHOD_NAME[algo_category]}"]
"""
    spec = factory.create_algo(algo_category, dockerfile=dockerfile)
    algo = client.add_algo(spec)
    spec = factory.create_traintask(
        algo=algo,
        inputs=default_dataset.train_data_inputs,
        worker=worker,
    )
    traintask = client.add_task(spec)
    traintask = client.wait(traintask)
