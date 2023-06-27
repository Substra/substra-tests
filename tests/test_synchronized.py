import pytest

from substratest.factory import AugmentedDataset
from substratest.factory import FunctionCategory


@pytest.fixture
def current_client(clients):
    """Reference client."""
    return clients[0]


@pytest.mark.remote_only
def test_synchronized_function(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 organizations")

    spec = factory.create_function(FunctionCategory.simple)
    function = current_client.add_function(spec)
    channel.wait_for_asset_synchronized(function)


@pytest.mark.remote_only
def test_synchronized_metric(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 organizations")

    spec = factory.create_function(category=FunctionCategory.metric)
    metric = current_client.add_function(spec)
    channel.wait_for_asset_synchronized(metric)


@pytest.mark.remote_only
def test_synchronized_dataset(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 organizations")

    spec = factory.create_dataset()
    dataset = current_client.add_dataset(spec)
    channel.wait_for_asset_synchronized(dataset)


@pytest.mark.remote_only
def test_synchronized_datasample(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 organizations")

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
def test_synchronized_traintask(clients, factory, channel, current_client, worker):
    if len(clients) < 2:
        pytest.skip("requires at least 2 organizations")

    # create function
    spec = factory.create_function(FunctionCategory.simple)
    function = current_client.add_function(spec)

    # create dataset
    spec = factory.create_dataset()
    dataset = current_client.add_dataset(spec)
    channel.wait_for_asset_synchronized(dataset)  # required by datasample

    # create datasample
    spec = factory.create_data_sample(datasets=[dataset])
    datasample_key = current_client.add_data_sample(spec)
    datasample = current_client.get_data_sample(datasample_key)
    channel.wait_for_asset_synchronized(datasample)  # required by traintask

    dataset = AugmentedDataset(current_client.get_dataset(dataset.key))
    dataset.set_train_test_dasamples(train_data_sample_keys=[datasample_key])

    # create traintask
    spec = factory.create_traintask(function=function, inputs=dataset.train_data_inputs, worker=worker)
    traintask = current_client.add_task(spec)
    traintask = current_client.wait_task(traintask.key, raises=True)
    channel.wait_for_asset_synchronized(traintask)


@pytest.mark.remote_only
def test_synchronized_computeplan(clients, factory, channel, current_client):
    if len(clients) < 2:
        pytest.skip("requires at least 2 organizations")

    # create function
    cp_spec = factory.create_compute_plan()
    compute_plan = current_client.add_compute_plan(cp_spec)
    assert compute_plan.creator != "external"

    # Compute plan are created with None value as duration, but for filtering and sorting purpose
    # remote api's get_compute_plan returns 0 as duration instead. We need to set it to 0
    # here instead of None for wait_for_asset_synchronized to pass on compute plan.
    compute_plan.duration = 0
    channel.wait_for_asset_synchronized(compute_plan)

    external_cp = clients[1].get_compute_plan(key=compute_plan.key)
    # creator should not be propagated to another org
    assert external_cp.creator == "external"
