import pytest

from substratest.factory import AlgoCategory


@pytest.fixture
def current_client(clients):
    """Reference client."""
    return clients[0]


@pytest.mark.remote_only
def test_synchronized_algo(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 nodes")

    spec = factory.create_algo(AlgoCategory.simple)
    algo = current_client.add_algo(spec)
    channel.wait_for_asset_synchronized(algo)


@pytest.mark.remote_only
def test_synchronized_metric(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 nodes")

    spec = factory.create_metric()
    metric = current_client.add_metric(spec)
    channel.wait_for_asset_synchronized(metric)


@pytest.mark.remote_only
def test_synchronized_dataset(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 nodes")

    spec = factory.create_dataset()
    dataset = current_client.add_dataset(spec)
    channel.wait_for_asset_synchronized(dataset)


@pytest.mark.remote_only
def test_synchronized_datasample(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 nodes")

    # create dataset
    spec = factory.create_dataset()
    dataset = current_client.add_dataset(spec)
    channel.wait_for_asset_synchronized(dataset)  # required by datasample

    # create datasample
    spec = factory.create_data_sample(datasets=[dataset])
    datasample_key = current_client.add_data_sample(spec)
    datasample = current_client.get_data_sample(datasample_key)
    channel.wait_for_asset_synchronized(datasample)


@pytest.mark.remote_only
def test_synchronized_traintuple(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 nodes")

    # create algo
    spec = factory.create_algo(AlgoCategory.simple)
    algo = current_client.add_algo(spec)

    # create dataset
    spec = factory.create_dataset()
    dataset = current_client.add_dataset(spec)
    channel.wait_for_asset_synchronized(dataset)  # required by datasample

    # create datasample
    spec = factory.create_data_sample(datasets=[dataset])
    datasample_key = current_client.add_data_sample(spec)
    datasample = current_client.get_data_sample(datasample_key)
    channel.wait_for_asset_synchronized(datasample)  # required by traintuple

    # create traintuple
    spec = factory.create_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=[datasample.key],
    )
    traintuple = current_client.add_traintuple(spec)
    channel.wait_for_asset_synchronized(traintuple)
