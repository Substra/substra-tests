import pytest

from substratest.factory import AlgoCategory


@pytest.mark.subprocess_skip
def test_base_connect_tools_image(factory, client, default_dataset):
    """Test that an algo created with the base connect-tools image instead of the minimal works"""

    suffix = "-minimal"
    if factory.default_tools_image.endswith(suffix):
        connect_tool_image = factory.default_tools_image[: -len(suffix)]
    else:
        connect_tool_image = factory.default_tools_image

    dockerfile = f"""
FROM {connect_tool_image}

COPY algo.py .

ENTRYPOINT ["python3", "algo.py"]
"""
    spec = factory.create_algo(AlgoCategory.simple, dockerfile=dockerfile)
    algo = client.add_algo(spec)
    spec = factory.create_traintuple(algo=algo, inputs=default_dataset.train_data_inputs)
    traintuple = client.add_traintuple(spec)
    traintuple = client.wait(traintuple)
