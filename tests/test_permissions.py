import substra

import pytest

from substratest.factory import Permissions
from substratest import assets
from . import settings

MSP_IDS = settings.MSP_IDS


@pytest.mark.parametrize('is_public', [True, False])
def test_permission_creation(is_public, factory, client):
    """Test asset creation with simple permission."""
    permissions = Permissions(public=is_public, authorized_ids=[])
    spec = factory.create_dataset(permissions=permissions)
    dataset = client.add_dataset(spec)
    assert dataset.permissions.process.public is is_public


@pytest.mark.parametrize('permissions', [
    Permissions(public=True, authorized_ids=[]),
    Permissions(public=False, authorized_ids=[]),
    Permissions(public=False, authorized_ids=MSP_IDS),
    Permissions(public=False, authorized_ids=[MSP_IDS[0]]),
    Permissions(public=False, authorized_ids=[MSP_IDS[1]]),
])
def test_get_metadata(permissions, factory, clients):
    """Test get metadata assets with various permissions."""
    clients = clients[:2]

    # add 1 dataset per node
    datasets = []
    for client in clients:
        spec = factory.create_dataset(permissions=permissions)
        d = client.add_dataset(spec)
        datasets.append(d)

    # check that all clients can get access to all metadata
    for client in clients:
        for d in datasets:
            d = client.get_dataset(d.key)
            assert d.permissions.process.public == permissions.public


def test_permission_invalid_node_id(factory, client):
    """Test asset creation with invalid permission."""
    invalid_node = 'unknown-node'
    invalid_permissions = Permissions(public=False, authorized_ids=[invalid_node])
    spec = factory.create_dataset(permissions=invalid_permissions)
    with pytest.raises(substra.exceptions.InvalidRequest) as exc:
        client.add_dataset(spec)
    assert "invalid permission input values" in str(exc.value)


@pytest.mark.parametrize('permissions', [
    Permissions(public=True, authorized_ids=[]),
    Permissions(public=False, authorized_ids=[MSP_IDS[1]]),
])
def test_download_asset_access_granted(permissions, factory, client_1, client_2):
    """Test asset can be downloaded by all permitted nodes."""
    spec = factory.create_dataset(permissions=permissions)
    dataset = client_1.add_dataset(spec)

    content = client_1.download_opener(dataset.key)
    assert content == spec.read_opener()

    content = client_2.download_opener(dataset.key)
    assert content == spec.read_opener()


def test_download_asset_access_restricted(factory, client_1, client_2):
    """Test public asset can be downloaded by all nodes."""
    permissions = Permissions(public=False, authorized_ids=[])
    spec = factory.create_dataset(permissions=permissions)
    dataset = client_1.add_dataset(spec)

    content = client_1.download_opener(dataset.key)
    assert content == spec.read_opener()

    with pytest.raises(substra.exceptions.AuthorizationError):
        client_2.download_opener(dataset.key)


@pytest.mark.parametrize('permissions_1,permissions_2,expected_permissions', [
    (
        Permissions(public=False, authorized_ids=[MSP_IDS[1]]),
        Permissions(public=False, authorized_ids=[MSP_IDS[0]]),
        Permissions(public=False, authorized_ids=[MSP_IDS[0], MSP_IDS[1]])
    ),
    (
        Permissions(public=True, authorized_ids=[]),
        Permissions(public=False, authorized_ids=[MSP_IDS[0]]),
        Permissions(public=False, authorized_ids=[MSP_IDS[0], MSP_IDS[1]])
    ),
])
def test_merge_permissions(permissions_1, permissions_2, expected_permissions,
                           factory, client_1, client_2):
    """Test merge permissions from dataset and algo asset located on different nodes.

    - dataset and objectives located on node 1
    - algo located on node 2
    - traintuple created on node 2
    """
    # add train data samples / dataset / objective on node 1
    spec = factory.create_dataset(permissions=permissions_1)
    dataset_1 = client_1.add_dataset(spec)
    spec = factory.create_data_sample(test_only=False, datasets=[dataset_1])
    train_data_sample_1 = client_1.add_data_sample(spec)

    # add algo on node 2
    spec = factory.create_algo(permissions=permissions_2)
    algo_2 = client_2.add_algo(spec)

    # add traintuple from node 2
    spec = factory.create_traintuple(
        algo=algo_2,
        dataset=dataset_1,
        data_samples=[train_data_sample_1],
    )
    traintuple = client_1.add_traintuple(spec).future().wait()
    assert traintuple.status == assets.Status.done
    assert traintuple.out_model is not None
    assert traintuple.dataset.worker == client_1.node_id
    tuple_permissions = traintuple.permissions.process
    assert tuple_permissions.public == expected_permissions.public
    assert set(tuple_permissions.authorized_ids) == set(expected_permissions.authorized_ids)


def test_permissions_denied_process(factory, client_1, client_2):
    # setup data

    spec = factory.create_dataset(permissions=Permissions(public=False, authorized_ids=[]))
    dataset_1 = client_1.add_dataset(spec)

    spec = factory.create_data_sample(
        test_only=False,
        datasets=[dataset_1],
    )
    train_data_sample_1 = client_1.add_data_sample(spec)

    # setup algo

    spec = factory.create_algo(permissions=Permissions(public=False, authorized_ids=[]))
    algo_2 = client_2.add_algo(spec)

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


@pytest.mark.slow
@pytest.mark.xfail(reason='permission check not yet implemented in the backend')
def test_permissions_denied_model_process(factory, client_1, client_2, network):
    # setup

    datasets = []
    algos = []
    for client in network.clients[:2]:
        # dataset
        spec = factory.create_dataset(permissions=Permissions(public=False, authorized_ids=[]))
        dataset = client.add_dataset(spec)
        spec = factory.create_data_sample(
            test_only=False,
            datasets=[dataset],
        )
        client.add_data_sample(spec)
        datasets.append(client.get_dataset(dataset.key))
        # algo
        spec = factory.create_algo(permissions=Permissions(public=False, authorized_ids=[]))
        algos.append(client.add_algo(spec))

    dataset_1, dataset_2 = datasets
    algo_1, algo_2 = algos

    # traintuples

    spec = factory.create_traintuple(
        algo=algo_1,
        dataset=dataset_1,
        data_samples=dataset_1.train_data_sample_keys,
        tag='foo',
    )
    traintuple_1 = client_1.add_traintuple(spec).future().wait()

    assert not traintuple_1.permissions.process.public
    assert traintuple_1.permissions.process.authorized_ids == [client_1.node_id]

    spec = factory.create_traintuple(
        algo=algo_2,
        dataset=dataset_2,
        data_samples=dataset_2.train_data_sample_keys,
        traintuples=[traintuple_1]
    )

    with pytest.raises(substra.exceptions.AuthenticationError):
        client_2.add_traintuple(spec)


@pytest.mark.skipif(len(MSP_IDS) < 3, reason='requires at least 3 nodes')
def test_merge_permissions_denied_process(factory, clients):
    """Test to process asset with merged permissions from 2 other nodes

    - dataset and objectives located on node 1
    - algo located on node 2
    - traintuple created on node 2
    - failed attempt to create testtuple using this traintuple from node 3
    """
    # define clients one and for all
    client_1 = clients[0]
    client_2 = clients[1]
    client_3 = clients[2]

    permissions_list = [(
        Permissions(public=False, authorized_ids=[MSP_IDS[1], MSP_IDS[2]]),
        Permissions(public=False, authorized_ids=[MSP_IDS[0]]),
    ), (
        Permissions(public=True, authorized_ids=[]),
        Permissions(public=False, authorized_ids=[MSP_IDS[0]]),
    )]
    for permissions_1, permissions_2 in permissions_list:

        # add train data samples / dataset / objective on node 1
        spec = factory.create_dataset(permissions=permissions_1)
        dataset_1 = client_1.add_dataset(spec)
        spec = factory.create_data_sample(
            test_only=False,
            datasets=[dataset_1],
        )
        train_data_sample_1 = client_1.add_data_sample(spec)
        spec = factory.create_data_sample(
            test_only=True,
            datasets=[dataset_1],
        )
        test_data_sample_1 = client_1.add_data_sample(spec)
        spec = factory.create_objective(
            dataset=dataset_1,
            data_samples=[test_data_sample_1],
            permissions=permissions_1,
        )
        objective_1 = client_1.add_objective(spec)

        # add algo on node 2
        spec = factory.create_algo(permissions=permissions_2)
        algo_2 = client_2.add_algo(spec)

        # add traintuple from node 2
        spec = factory.create_traintuple(
            algo=algo_2,
            dataset=dataset_1,
            data_samples=[train_data_sample_1],
        )
        traintuple_2 = client_2.add_traintuple(spec).future().wait()

        # failed to add testtuple from node 3
        spec = factory.create_testtuple(
            objective=objective_1,
            traintuple=traintuple_2,
        )

        with pytest.raises(substra.exceptions.AuthorizationError):
            client_3.add_testtuple(spec)


def test_permissions_denied_head_model_process(factory, client_1, client_2):
    # setup data

    datasets = []
    for client in [client_1, client_2]:

        spec = factory.create_dataset(permissions=Permissions(public=False, authorized_ids=[client.node_id]))
        dataset = client.add_dataset(spec)

        spec = factory.create_data_sample(
            test_only=False,
            datasets=[dataset],
        )
        client.add_data_sample(spec)

        datasets.append(client.get_dataset(dataset.key))

    dataset_1, dataset_2 = datasets

    # setup algo

    spec = factory.create_composite_algo()
    composite_algo = client_1.add_composite_algo(spec)

    # composite traintuples

    spec = factory.create_composite_traintuple(
        algo=composite_algo,
        dataset=dataset_1,
        data_samples=dataset_1.train_data_sample_keys,
    )
    composite_traintuple_1 = client_1.add_composite_traintuple(spec)

    spec = factory.create_composite_traintuple(
        algo=composite_algo,
        dataset=dataset_2,
        data_samples=dataset_2.train_data_sample_keys,
        head_traintuple=composite_traintuple_1,
        trunk_traintuple=composite_traintuple_1,
    )
    with pytest.raises(substra.exceptions.InvalidRequest):
        client_2.add_composite_traintuple(spec)


def test_permission_public_trunk(factory, client):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    spec = factory.create_data_sample(
        test_only=False,
        datasets=[dataset],
    )
    data_sample = client.add_data_sample(spec)

    spec = factory.create_composite_algo()
    composite_algo = client.add_composite_algo(spec)

    spec = factory.create_composite_traintuple(
        algo=composite_algo,
        dataset=dataset,
        data_samples=[data_sample],
    )
    spec_dict = spec.dict()
    spec_dict['out_trunk_model_permissions']['public'] = True

    res = client._client.add_composite_traintuple(spec_dict)
    composite_traintuple = assets.CompositeTraintuple.load(res).attach(client)
    assert composite_traintuple.out_trunk_model.permissions.process.public is False
    composite_traintuple.future().wait()
