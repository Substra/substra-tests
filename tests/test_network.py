import os

import substra

import pytest

import substratest as sbt
from . import settings


def test_connection_to_nodes(clients):
    """Connect to each substra nodes using the client."""
    for client in clients:
        client.list_algo()


def test_add_dataset(factory, client):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)
    assert dataset.metadata == {}

    dataset_copy = client.get_dataset(dataset.key)
    assert dataset == dataset_copy


def test_download_opener(factory, client):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    content = client.download_opener(dataset.key)
    assert content == spec.read_opener()


def test_describe_dataset(factory, client):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    content = client.describe_dataset(dataset.key)
    assert content == spec.read_description()


def test_add_dataset_conflict(factory, client):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    with pytest.raises(substra.exceptions.AlreadyExists):
        client.add_dataset(spec)

    dataset_copy = client.add_dataset(spec, exist_ok=True)
    assert dataset == dataset_copy


def test_link_dataset_with_objective(factory, client):
    spec = factory.create_objective()
    objective_1 = client.add_objective(spec)

    spec = factory.create_objective()
    objective_2 = client.add_objective(spec)

    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    # link dataset with objective
    client.link_dataset_with_objective(dataset, objective_1)
    dataset = client.get_dataset(dataset.key)
    assert dataset.objective_key == objective_1.key

    # ensure an existing dataset cannot be linked to more than one objective
    with pytest.raises(substra.exceptions.InvalidRequest):
        client.link_dataset_with_objective(dataset, objective_2)


def test_add_data_sample(factory, client):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    client.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    client.add_data_sample(spec)


def test_link_dataset_with_datasamples(factory, client):
    # create data sample and link it to a dataset
    spec = factory.create_dataset()
    dataset_tmp = client.add_dataset(spec)

    spec = factory.create_data_sample(datasets=[dataset_tmp])
    data_sample = client.add_data_sample(spec)

    # create a new dataset and link existing data sample to the new dataset
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    dataset = client.get_dataset(dataset.key)
    assert dataset.train_data_sample_keys == []

    client.link_dataset_with_data_samples(dataset, [data_sample])

    dataset = client.get_dataset(dataset.key)
    assert dataset.train_data_sample_keys == [data_sample]


@pytest.mark.remote_only
@pytest.mark.skipif(not settings.HAS_SHARED_PATH, reason='requires a shared path')
def test_add_data_sample_located_in_shared_path(factory, client, node_cfg):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    spec.move_data_to_server(node_cfg.shared_path, settings.IS_MINIKUBE)
    client.add_data_sample(spec, local=False)  # should not raise


@pytest.mark.skip(reason='may fill up disk as shared folder is not cleanup')
@pytest.mark.parametrize('filesize', [1, 10, 100, 1000])  # in mega
def test_add_data_sample_path_big_files(filesize, factory, client, node_cfg):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    content = os.urandom(filesize * 1000 * 1000)
    spec = factory.create_data_sample(content=content, datasets=[dataset])
    spec.move_data_to_server(node_cfg.shared_path, settings.IS_MINIKUBE)
    client.add_data_sample(spec, local=False)  # should not raise


def test_add_objective(factory, client):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    data_sample = client.add_data_sample(spec)

    spec = factory.create_objective(dataset=dataset, data_samples=[data_sample])
    objective = client.add_objective(spec)
    objective_copy = client.get_objective(objective.key)
    assert objective == objective_copy

    # check dataset has been linked with the objective after objective creation
    dataset = client.get_dataset(dataset.key)
    assert dataset.objective_key == objective.key

    # add a new dataset and ensure it is linked with the created objective
    spec = factory.create_dataset(objective=objective)
    dataset = client.add_dataset(spec)
    assert dataset.objective_key == objective.key


@pytest.mark.parametrize('asset_name', [
    'dataset',
    'objective',
    'algo',
    'aggregate_algo',
    'composite_algo',
])
@pytest.mark.parametrize('metadata,metadata_output', [
    ({'foo': 'bar'}, {'foo': 'bar'}),
    (None, {}),
    ({}, {}),
])
def test_asset_with_metadata(factory, client, asset_name, metadata, metadata_output):
    create_spec = getattr(factory, f"create_{asset_name}")
    add_asset = getattr(client, f"add_{asset_name}")

    spec = create_spec(metadata=metadata)
    asset = add_asset(spec)

    assert asset.metadata == metadata_output


@pytest.mark.parametrize('asset_name', [
    'dataset',
    'objective',
    'algo',
    'aggregate_algo',
    'composite_algo',
])
@pytest.mark.parametrize('metadata', [
    {'foo' * 40: "bar"},
    {"foo": 'bar' * 40},
])
def test_asset_with_invalid_metadata(factory, client, asset_name, metadata):
    create_spec = getattr(factory, f"create_{asset_name}")
    add_asset = getattr(client, f"add_{asset_name}")

    spec = create_spec(metadata=metadata)

    with pytest.raises(substra.exceptions.InvalidRequest):
        add_asset(spec)


def test_add_algo(factory, client):
    spec = factory.create_algo()
    algo = client.add_algo(spec)

    algo_copy = client.get_algo(algo.key)
    assert algo == algo_copy


def test_add_composite_algo(factory, client):
    spec = factory.create_composite_algo()
    algo = client.add_composite_algo(spec)

    algo_copy = client.get_composite_algo(algo.key)
    assert algo == algo_copy


@pytest.mark.remote_only  # No node saved in the local backend
def test_list_nodes(client, network):
    """Nodes are properly registered and list nodes returns expected nodes."""
    nodes = client.list_node()
    node_ids = [n.id for n in nodes]
    network_node_ids = [c.node_id for c in network.clients]
    # check all nodes configured are correctly registered
    assert set(network_node_ids).issubset(set(node_ids))


def test_query_algos(factory, client):
    spec = factory.create_algo()
    algo = client.add_algo(spec)

    spec = factory.create_composite_algo()
    compo_algo = client.add_composite_algo(spec)

    # check the created composite algo is not returned when listing algos
    algo_keys = [a.key for a in client.list_algo()]
    assert algo.key in algo_keys
    assert compo_algo.key not in algo_keys

    # check the created algo is not returned when listing composite algos
    compo_algo_keys = [a.key for a in client.list_composite_algo()]
    assert compo_algo.key in compo_algo_keys
    assert algo.key not in compo_algo_keys


@pytest.mark.parametrize(
    'asset_type', sbt.assets.AssetType.can_be_listed(),
)
def test_list_asset(asset_type, client):
    """Simple check that list_asset method can be called without raising errors."""
    method = getattr(client, f'list_{asset_type.name}')
    method()  # should not raise


@pytest.mark.remote_only
@pytest.mark.parametrize(
    'asset_type', sbt.assets.AssetType.can_be_get(),
)
def test_error_get_asset_invalid_request(asset_type, client):
    method = getattr(client, f'get_{asset_type.name}')
    with pytest.raises(substra.exceptions.InvalidRequest):
        method('invalid-id')


@pytest.mark.parametrize(
    'asset_type', sbt.assets.AssetType.can_be_get(),
)
def test_error_get_asset_not_found(asset_type, client):
    method = getattr(client, f'get_{asset_type.name}')
    with pytest.raises(substra.exceptions.NotFound):
        method('42' * 32)  # a valid key must have a 64 length
