import pytest
import substra

from substratest.factory import AlgoCategory
from substratest.factory import Permissions


@pytest.fixture
def public():
    return Permissions(public=True, authorized_ids=[])


@pytest.fixture
def private():
    return Permissions(public=False, authorized_ids=[])


@pytest.fixture
def all_organizations(clients):
    return Permissions(public=False, authorized_ids=[c.organization_id for c in clients])


@pytest.fixture
def organization_1_only(client_1):
    return Permissions(public=False, authorized_ids=[client_1.organization_id])


@pytest.fixture
def organization_2_only(client_2):
    return Permissions(public=False, authorized_ids=[client_2.organization_id])


@pytest.fixture
def organizations_1_and_2_only(client_1, client_2):
    return Permissions(public=False, authorized_ids=[client_1.organization_id, client_2.organization_id])


@pytest.mark.remote_only  # no check on permissions with the local backend
@pytest.mark.parametrize("is_public", [True, False])
def test_permission_creation(is_public, factory, client):
    """Test asset creation with simple permission."""
    permissions = Permissions(public=is_public, authorized_ids=[])
    spec = factory.create_dataset(permissions=permissions)
    dataset = client.add_dataset(spec)
    assert dataset.permissions.process.public is is_public


@pytest.mark.remote_only  # no check on permissions with the local backend
@pytest.mark.parametrize(
    "permissions",
    [
        pytest.lazy_fixture("public"),
        pytest.lazy_fixture("private"),
        pytest.lazy_fixture("all_organizations"),
        pytest.lazy_fixture("organization_1_only"),
        pytest.lazy_fixture("organization_2_only"),
    ],
)
def test_get_metadata(permissions, factory, clients, channel):
    """Test get metadata assets with various permissions."""
    clients = clients[:2]

    # add 1 dataset per organization
    datasets = []
    for client in clients:
        spec = factory.create_dataset(permissions=permissions)
        d = client.add_dataset(spec)
        datasets.append(d)

    for d in datasets:
        channel.wait_for_asset_synchronized(d)

    # check that all clients can get access to all metadata
    for client in clients:
        for d in datasets:
            d = client.get_dataset(d.key)
            assert d.permissions.process.public == permissions.public


@pytest.mark.remote_only  # no check on permissions with the local backend
def test_permission_invalid_organization_id(factory, client):
    """Test asset creation with invalid permission."""
    invalid_organization = "unknown-organization"
    invalid_permissions = Permissions(public=False, authorized_ids=[invalid_organization])
    spec = factory.create_dataset(permissions=invalid_permissions)
    with pytest.raises(substra.exceptions.InvalidRequest) as exc:
        client.add_dataset(spec)
    assert "invalid permission input values" in str(exc.value)


@pytest.mark.remote_only  # no check on permissions with the local backend
@pytest.mark.parametrize(
    "permissions",
    [
        pytest.lazy_fixture("public"),
        pytest.lazy_fixture("organization_2_only"),
    ],
)
def test_download_asset_access_granted(permissions, factory, client_1, client_2, channel):
    """Test asset can be downloaded by all permitted organizations."""
    spec = factory.create_dataset(permissions=permissions)
    dataset = client_1.add_dataset(spec)

    content = client_1.download_opener(dataset.key)
    assert content == spec.read_opener()

    channel.wait_for_asset_synchronized(dataset)
    content = client_2.download_opener(dataset.key)
    assert content == spec.read_opener()


@pytest.mark.remote_only  # no check on permissions with the local backend
def test_download_asset_access_restricted(factory, client_1, client_2, channel):
    """Test public asset can be downloaded by all organizations."""
    permissions = Permissions(public=False, authorized_ids=[])
    spec = factory.create_dataset(permissions=permissions)
    dataset = client_1.add_dataset(spec)

    content = client_1.download_opener(dataset.key)
    assert content == spec.read_opener()
    channel.wait_for_asset_synchronized(dataset)

    with pytest.raises(substra.exceptions.AuthorizationError):
        client_2.download_opener(dataset.key)


@pytest.mark.remote_only  # no check on permissions with the local backend
@pytest.mark.parametrize(
    "permissions_1,permissions_2,expected_permissions",
    [
        (
            pytest.lazy_fixture("organization_2_only"),
            pytest.lazy_fixture("organization_1_only"),
            pytest.lazy_fixture("organizations_1_and_2_only"),
        ),
        (
            pytest.lazy_fixture("public"),
            pytest.lazy_fixture("organization_1_only"),
            pytest.lazy_fixture("organizations_1_and_2_only"),
        ),
    ],
)
def test_merge_permissions(permissions_1, permissions_2, expected_permissions, factory, client_1, client_2, channel):
    """Test merge permissions from dataset and algo asset located on different organizations.

    - dataset and metrics located on organization 1
    - algo located on organization 2
    - traintuple created on organization 2
    """
    # add train data samples / dataset / metric on organization 1
    spec = factory.create_dataset(permissions=permissions_1)
    dataset_1 = client_1.add_dataset(spec)
    spec = factory.create_data_sample(test_only=False, datasets=[dataset_1])
    train_data_sample_1 = client_1.add_data_sample(spec)

    # add algo on organization 2
    spec = factory.create_algo(category=AlgoCategory.simple, permissions=permissions_2)
    algo_2 = client_2.add_algo(spec)

    # add traintuple from organization 2
    spec = factory.create_traintuple(
        algo=algo_2,
        dataset=dataset_1,
        data_samples=[train_data_sample_1],
    )
    channel.wait_for_asset_synchronized(algo_2)
    traintuple = client_1.add_traintuple(spec)
    traintuple = client_1.wait(traintuple)
    assert traintuple.train.models is not None
    assert traintuple.worker == client_1.organization_id
    tuple_permissions = traintuple.train.model_permissions.process
    assert tuple_permissions.public == expected_permissions.public
    assert set(tuple_permissions.authorized_ids) == set(expected_permissions.authorized_ids)


@pytest.mark.remote_only  # no check on permissions with the local backend
def test_permissions_denied_process(factory, client_1, client_2, channel):
    # setup data

    spec = factory.create_dataset(permissions=Permissions(public=False, authorized_ids=[]))
    dataset_1 = client_1.add_dataset(spec)

    spec = factory.create_data_sample(
        test_only=False,
        datasets=[dataset_1],
    )
    train_data_sample_1 = client_1.add_data_sample(spec)

    # setup algo

    spec = factory.create_algo(category=AlgoCategory.simple, permissions=Permissions(public=False, authorized_ids=[]))
    algo_2 = client_2.add_algo(spec)
    channel.wait_for_asset_synchronized(algo_2)

    # traintuples

    spec = factory.create_traintuple(
        algo=algo_2,
        dataset=dataset_1,
        data_samples=[train_data_sample_1],
    )

    with pytest.raises(substra.exceptions.AuthorizationError):
        client_2.add_traintuple(spec)

    with pytest.raises(substra.exceptions.AuthorizationError):
        client_1.add_traintuple(spec)


@pytest.mark.remote_only  # no check on permissions with the local backend
@pytest.mark.slow
@pytest.mark.parametrize(
    "client_1_permissions,client_2_permissions,expected_success",
    [
        (pytest.lazy_fixture("private"), pytest.lazy_fixture("private"), False),
        (pytest.lazy_fixture("organization_2_only"), pytest.lazy_fixture("private"), True),
    ],
)
def test_permissions_model_process(
    client_1_permissions, client_2_permissions, expected_success, factory, client_1, client_2, network, channel
):
    """Test that a traintuple can/cannot process an in-model depending on permissions."""
    datasets = []
    algos = []
    for client, permissions in zip([client_1, client_2], [client_1_permissions, client_2_permissions]):
        # dataset
        spec = factory.create_dataset(permissions=permissions)
        dataset = client.add_dataset(spec)
        spec = factory.create_data_sample(
            test_only=False,
            datasets=[dataset],
        )
        client.add_data_sample(spec)
        datasets.append(client.get_dataset(dataset.key))

        # algo
        spec = factory.create_algo(category=AlgoCategory.simple, permissions=permissions)
        algo = client.add_algo(spec)
        channel.wait_for_asset_synchronized(algo)
        algos.append(algo)

    dataset_1, dataset_2 = datasets
    algo_1, algo_2 = algos

    # traintuples
    spec = factory.create_traintuple(
        algo=algo_1,
        dataset=dataset_1,
        data_samples=dataset_1.train_data_sample_keys,
    )
    traintuple_1 = client_1.add_traintuple(spec)
    traintuple_1 = client_1.wait(traintuple_1)

    print(spec)

    assert not traintuple_1.train.model_permissions.process.public
    assert set(traintuple_1.train.model_permissions.process.authorized_ids) == set(
        [client_1.organization_id] + client_1_permissions.authorized_ids
    )

    spec = factory.create_traintuple(
        algo=algo_2, dataset=dataset_2, data_samples=dataset_2.train_data_sample_keys, traintuples=[traintuple_1]
    )

    if expected_success:
        traintuple_2 = client_2.add_traintuple(spec)

        client_2.wait(traintuple_2)
    else:
        with pytest.raises(substra.exceptions.AuthorizationError):
            traintuple_2 = client_2.add_traintuple(spec)


@pytest.mark.remote_only  # no check on permissions with the local backend
def test_merge_permissions_denied_process(factory, clients, channel):
    """Test to process asset with merged permissions from 2 other organizations

    - dataset and metrics located on organization 1
    - algo located on organization 2
    - traintuple created on organization 2
    - failed attempt to create testtuple using this traintuple from organization 3
    """
    if len(clients) < 3:
        pytest.skip("requires at least 3 organizations")

    # define clients once and for all
    client_1, client_2, client_3, *_ = clients

    permissions_list = [
        (
            Permissions(public=False, authorized_ids=[client_2.organization_id, client_3.organization_id]),
            Permissions(public=False, authorized_ids=[client_1.organization_id]),
        ),
        (
            Permissions(public=True, authorized_ids=[]),
            Permissions(public=False, authorized_ids=[client_1.organization_id]),
        ),
    ]
    for permissions_1, permissions_2 in permissions_list:

        # add train data samples / dataset / metric on organization 1
        spec = factory.create_dataset(permissions=permissions_1)
        dataset_1 = client_1.add_dataset(spec)
        channel.wait_for_asset_synchronized(dataset_1)  # used by client_2 and client_3
        spec = factory.create_data_sample(
            test_only=False,
            datasets=[dataset_1],
        )
        train_data_sample_1 = client_1.add_data_sample(spec)
        spec = factory.create_data_sample(
            test_only=True,
            datasets=[dataset_1],
        )
        _ = client_1.add_data_sample(spec)
        spec = factory.create_algo(category=AlgoCategory.metric, permissions=permissions_1)
        metric_1 = client_1.add_algo(spec)
        channel.wait_for_asset_synchronized(metric_1)  # used by client_3

        # add algo on organization 2
        spec = factory.create_algo(category=AlgoCategory.simple, permissions=permissions_2)
        algo_2 = client_2.add_algo(spec)

        # add traintuple from organization 2
        spec = factory.create_algo(category=AlgoCategory.predict, permissions=permissions_1)
        predict_algo_1 = client_1.add_algo(spec)

        # add traintuple from node 2
        spec = factory.create_traintuple(
            algo=algo_2,
            dataset=dataset_1,
            data_samples=[train_data_sample_1],
        )
        traintuple_2 = client_2.add_traintuple(spec)
        traintuple_2 = client_2.wait(traintuple_2)
        channel.wait_for_asset_synchronized(traintuple_2)  # used by client_3

        # failed to add predicttuple from organization 3
        spec = factory.create_predicttuple(
            algo=predict_algo_1,
            traintuple=traintuple_2,
            dataset=dataset_1,
            data_samples=dataset_1.test_data_sample_keys,
        )

        with pytest.raises(substra.exceptions.AuthorizationError):
            client_3.add_predicttuple(spec)


@pytest.mark.remote_only  # no check on permissions with the local backend
def test_permissions_denied_head_model_process(factory, client_1, client_2, channel):
    # setup data

    datasets = []
    for client in [client_1, client_2]:

        spec = factory.create_dataset(permissions=Permissions(public=False, authorized_ids=[client.organization_id]))
        dataset = client.add_dataset(spec)

        spec = factory.create_data_sample(
            test_only=False,
            datasets=[dataset],
        )
        client.add_data_sample(spec)

        dataset = client.get_dataset(dataset.key)
        channel.wait_for_asset_synchronized(dataset)
        datasets.append(dataset)

    dataset_1, dataset_2 = datasets

    # setup algo

    spec = factory.create_algo(category=AlgoCategory.composite)
    composite_algo = client_1.add_algo(spec)
    channel.wait_for_asset_synchronized(composite_algo)  # used by client_2

    # composite traintuples

    spec = factory.create_composite_traintuple(
        algo=composite_algo,
        dataset=dataset_1,
        data_samples=dataset_1.train_data_sample_keys,
    )

    composite_traintuple_1 = client_1.add_composite_traintuple(spec)

    composite_traintuple_1 = client_1.wait(composite_traintuple_1)

    channel.wait_for_asset_synchronized(composite_traintuple_1)  # used by client_2

    spec = factory.create_composite_traintuple(
        algo=composite_algo,
        dataset=dataset_2,
        data_samples=dataset_2.train_data_sample_keys,
        head_traintuple=composite_traintuple_1,
        trunk_traintuple=composite_traintuple_1,
    )
    with pytest.raises(substra.exceptions.AuthorizationError):
        client_2.add_composite_traintuple(spec)
