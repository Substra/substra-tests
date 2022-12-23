import copy
import os
import uuid

import pytest
import substra

import substratest as sbt
from substratest.factory import AlgoCategory


def test_connection_to_organizations(clients):
    """Connect to each substra organizations using the client."""
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


@pytest.mark.parametrize("category", [AlgoCategory.simple, AlgoCategory.aggregate, AlgoCategory.composite])
def test_download_algo(factory, client, category):
    spec = factory.create_algo(category)
    algo = client.add_algo(spec)

    content = client.download_algo(algo.key)
    with open(spec.file, "rb") as f:
        expected_content = f.read()
    assert content == expected_content


def test_describe_dataset(factory, client):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    content = client.describe_dataset(dataset.key)
    assert content == spec.read_description()


def test_add_duplicate_dataset(factory, client):
    spec = factory.create_dataset()
    client.add_dataset(spec)

    # does not raise
    client.add_dataset(spec)


def test_add_data_sample(factory, client):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    spec = factory.create_data_sample(tdatasets=[dataset])
    client.add_data_sample(spec)

    spec = factory.create_data_sample(datasets=[dataset])
    client.add_data_sample(spec)


def test_add_data_samples_in_batch(factory, client):
    batch_size = 5
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    specs = [factory.create_data_sample(datasets=[dataset]) for _ in range(batch_size)]

    spec = sbt.factory.DataSampleBatchSpec.from_data_sample_specs(specs)

    keys = client.add_data_samples(spec)
    assert len(keys) == batch_size


def test_link_dataset_with_datasamples(factory, client):
    # create data sample and link it to a dataset
    spec = factory.create_dataset()
    dataset_tmp = client.add_dataset(spec)

    spec = factory.create_data_sample(datasets=[dataset_tmp])
    data_sample_key = client.add_data_sample(spec)

    # create a new dataset and link existing data sample to the new dataset
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    dataset = client.get_dataset(dataset.key)
    assert dataset.data_sample_keys == []

    client.link_dataset_with_data_samples(dataset, [data_sample_key])

    dataset = client.get_dataset(dataset.key)
    assert dataset.data_sample_keys == [data_sample_key]


@pytest.mark.skip(reason="may fill up disk as shared folder is not cleanup")
@pytest.mark.parametrize("filesize", [1, 10, 100, 1000])  # in mega
def test_add_data_sample_path_big_files(network, filesize, factory, client, organization_cfg):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    content = os.urandom(filesize * 1000 * 1000)
    spec = factory.create_data_sample(content=content, datasets=[dataset])
    spec.move_data_to_server(organization_cfg.shared_path, network.options.minikube)
    client.add_data_sample(spec, local=False)  # should not raise


@pytest.mark.parametrize(
    "asset_name,params",
    [
        ("dataset", {}),
        ("algo", {"category": AlgoCategory.simple}),
    ],
)
@pytest.mark.parametrize(
    "metadata,metadata_output",
    [
        ({"foo": "bar"}, {"foo": "bar"}),
        (None, {}),
        ({}, {}),
    ],
)
def test_asset_with_metadata(factory, client, asset_name, params, metadata, metadata_output):
    create_spec = getattr(factory, f"create_{asset_name}")
    add_asset = getattr(client, f"add_{asset_name}")

    spec_params = {}
    spec_params.update(params)
    spec_params.update({"metadata": metadata})
    spec = create_spec(**spec_params)
    asset = add_asset(spec)

    assert asset.metadata == metadata_output


@pytest.mark.parametrize(
    "asset_name,params",
    [
        ("dataset", {}),
        ("algo", {"category": AlgoCategory.simple}),
    ],
)
@pytest.mark.parametrize(
    "metadata",
    [
        {"foo" * 40: "bar"},
        {"foo": "bar" * 40},
        {"foo__bar": "40"},
    ],
)
def test_asset_with_invalid_metadata(factory, client, asset_name, params, metadata):
    create_spec = getattr(factory, f"create_{asset_name}")
    add_asset = getattr(client, f"add_{asset_name}")

    spec_params = {}
    spec_params.update(params)
    spec_params.update({"metadata": metadata})
    spec = create_spec(**spec_params)

    with pytest.raises(substra.exceptions.InvalidRequest):
        add_asset(spec)


def test_add_algo(factory, client):
    spec = factory.create_algo(category=AlgoCategory.simple)
    algo = client.add_algo(spec)

    algo_copy = client.get_algo(algo.key)
    assert algo == algo_copy


@pytest.mark.remote_only  # No organization saved in the local backend
def test_list_organizations(client, network):
    """Organizations are properly registered and list organizations returns expected organizations."""
    organizations = client.list_organization()
    organization_ids = [n.id for n in organizations]
    network_organization_ids = [c.organization_id for c in network.clients]
    # check all organizations configured are correctly registered
    assert set(network_organization_ids).issubset(set(organization_ids))


def test_query_algos(factory, client):
    """Check we can find a newly created algo through the list method."""
    spec = factory.create_algo(category=AlgoCategory.simple)
    algo = client.add_algo(spec)

    matching_algos = [a for a in client.list_algo() if a.key == algo.key]
    assert len(matching_algos) == 1

    # ensure the list method returns the same information as the add method
    assert algo == matching_algos[0]


@pytest.mark.parametrize(
    "asset_type",
    sbt.assets.AssetType.can_be_listed(),
)
def test_list_asset(asset_type, client):
    """Simple check that list_asset method can be called without raising errors."""
    method = getattr(client, f"list_{asset_type.name}")
    method()  # should not raise


@pytest.mark.parametrize(
    "asset_type",
    sbt.assets.AssetType.can_be_get(),
)
def test_error_get_asset_not_found(asset_type, client):
    method = getattr(client, f"get_{asset_type.name}")
    with pytest.raises(substra.exceptions.NotFound):
        method(str(uuid.uuid4()))


@pytest.mark.parametrize(
    "asset_name,asset_params",
    [
        ("algo", {"category": AlgoCategory.simple}),
        ("compute_plan", {}),
        ("dataset", {}),
    ],
)
def test_update_name(asset_name, asset_params, client, factory, channel):
    create_spec_method = getattr(factory, f"create_{asset_name}")

    add_method = getattr(client, f"add_{asset_name}")
    update_method = getattr(client, f"update_{asset_name}")
    get_method = getattr(client, f"get_{asset_name}")

    # create asset
    spec = create_spec_method(**asset_params)
    asset = add_method(spec)
    # Compute plan are created with None value as duration, but for filtering and sorting purpose
    # remote api's get_compute_plan returns 0 as duration instead. We need to set it to 0
    # here instead of None for wait_for_asset_synchronized to pass on compute plan.
    if asset_name == "compute_plan":
        asset.duration = 0
    channel.wait_for_asset_synchronized(asset)

    # update asset
    new_name = "updated name"
    update_method(asset, new_name)

    expected_asset = copy.deepcopy(asset)
    expected_asset.name = new_name

    channel.wait_for_asset_synchronized(expected_asset)
    updated_asset = get_method(asset.key)

    assert updated_asset.name != asset.name
    assert updated_asset.name == new_name


@pytest.mark.remote_only
@pytest.mark.parametrize(
    "asset_name,asset_params",
    [
        ("algo", {"category": AlgoCategory.simple}),
        ("compute_plan", {}),
        ("dataset", {}),
    ],
)
def test_update_name_unauthorized(asset_name, asset_params, clients, factory, channel):
    if len(clients) < 2:
        pytest.skip("requires at least 2 organizations")
    create_spec_method = getattr(factory, f"create_{asset_name}")

    add_method = getattr(clients[0], f"add_{asset_name}")
    update_method = getattr(clients[1], f"update_{asset_name}")

    # create asset
    spec = create_spec_method(**asset_params)
    asset = add_method(spec)

    # Compute plan are created with None value as duration, but for filtering and sorting purpose
    # remote api's get_compute_plan returns 0 as duration instead. We need to set it to 0
    # here instead of None for wait_for_asset_synchronized to pass on compute plan.
    if asset_name == "compute_plan":
        asset.duration = 0
    channel.wait_for_asset_synchronized(asset)

    # update asset from different user
    new_name = "updated name"
    with pytest.raises(substra.exceptions.AuthorizationError):
        update_method(asset, new_name)
