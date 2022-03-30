import pytest

from substratest.factory import AlgoCategory


@pytest.fixture
def current_client(clients):
    """Reference client."""
    return clients[0]


@pytest.mark.remote_only
def test_synchronized_algo_on_creation(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 nodes")

    spec = factory.create_algo(AlgoCategory.simple)
    algo = current_client.add_algo(spec)
    channel.wait_for_asset_synchronized(algo)


@pytest.mark.remote_only
def test_synchronized_metric_on_creation(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 nodes")

    spec = factory.create_metric()
    metric = current_client.add_metric(spec)
    channel.wait_for_asset_synchronized(metric)


@pytest.mark.remote_only
def test_synchronized_dataset_on_creation(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 nodes")

    spec = factory.create_dataset()
    dataset = current_client.add_dataset(spec)
    channel.wait_for_asset_synchronized(dataset)


@pytest.mark.remote_only
def test_synchronized_dataset_on_update(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 nodes")

    # create dataset
    spec = factory.create_dataset()
    dataset = current_client.add_dataset(spec)
    channel.wait_for_asset_synchronized(dataset)

    # update dataset by adding a new datasample
    spec = factory.create_data_sample(datasets=[dataset])
    current_client.add_data_sample(spec)
