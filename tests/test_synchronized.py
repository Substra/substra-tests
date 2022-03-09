import pytest

from substratest import settings
from substratest.factory import AlgoCategory

MSP_IDS = settings.MSP_IDS


@pytest.fixture
def current_client(clients):
    """Reference client."""
    return clients[0]


@pytest.mark.remote_only
@pytest.mark.skipif(len(MSP_IDS) < 2, reason="requires at least 2 nodes")
def test_synchronized_algo_on_creation(factory, channel, current_client):
    spec = factory.create_algo(AlgoCategory.simple)
    algo = current_client.add_algo(spec)
    channel.wait_for_asset_synchronized(algo)


@pytest.mark.remote_only
@pytest.mark.skipif(len(MSP_IDS) < 2, reason="requires at least 2 nodes")
def test_synchronized_metric_on_creation(factory, channel, current_client):
    spec = factory.create_metric()
    metric = current_client.add_metric(spec)
    channel.wait_for_asset_synchronized(metric)


@pytest.mark.remote_only
@pytest.mark.skipif(len(MSP_IDS) < 2, reason="requires at least 2 nodes")
def test_synchronized_dataset_on_creation(factory, channel, current_client):
    spec = factory.create_dataset()
    dataset = current_client.add_dataset(spec)
    channel.wait_for_asset_synchronized(dataset)


@pytest.mark.remote_only
@pytest.mark.skipif(len(MSP_IDS) < 2, reason="requires at least 2 nodes")
def test_synchronized_dataset_on_update(factory, channel, current_client):
    # create dataset
    spec = factory.create_dataset()
    dataset = current_client.add_dataset(spec)
    channel.wait_for_asset_synchronized(dataset)

    # update dataset by adding a new datasample
    spec = factory.create_data_sample(datasets=[dataset])
    current_client.add_data_sample(spec)
