import pytest

from substratest.factory import AlgoCategory


@pytest.mark.subprocess_skip
def test_base_connect_tools_image(factory, client, default_dataset):
    """Test that an algo created with the base connect-tools image instead of the minimal works"""
    dockerfile = """
FROM gcr.io/connect-314908/connect-tools:latest-nvidiacuda11.6.0-base-ubuntu20.04-python3.7

COPY algo.py .

ENTRYPOINT ["python3", "algo.py"]
"""
    spec = factory.create_algo(AlgoCategory.simple, dockerfile=dockerfile)
    algo = client.add_algo(spec)
    spec = factory.create_traintuple(algo=algo, inputs=default_dataset.train_data_inputs)
    traintuple = client.add_traintuple(spec)
    traintuple = client.wait(traintuple)
