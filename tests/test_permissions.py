import substra

import pytest

from substratest.factory import Permissions
from substratest import assets

from . import settings

MSP_IDS = settings.MSP_IDS


@pytest.mark.parametrize('is_public', [True, False])
def test_permission_creation(is_public, factory, session):
    """Test asset creation with simple permission."""
    permissions = Permissions(public=is_public, authorized_ids=[])
    spec = factory.create_dataset(permissions=permissions)
    dataset = session.add_dataset(spec)
    assert dataset.permissions.process.public is is_public


@pytest.mark.parametrize('permissions', [
    Permissions(public=True, authorized_ids=[]),
    Permissions(public=False, authorized_ids=[]),
    Permissions(public=False, authorized_ids=MSP_IDS),
    Permissions(public=False, authorized_ids=[MSP_IDS[0]]),
    Permissions(public=False, authorized_ids=[MSP_IDS[1]]),
])
def test_get_metadata(permissions, factory, network):
    """Test get metadata assets with various permissions."""
    sessions = network.sessions[:2]

    # add 1 dataset per node
    datasets = []
    for session in sessions:
        spec = factory.create_dataset(permissions=permissions)
        d = session.add_dataset(spec)
        datasets.append(d)

    # check that all sessions can get access to all metadata
    for session in sessions:
        for d in datasets:
            d = session.get_dataset(d.key)
            assert d.permissions.process.public == permissions.public


def test_permission_invalid_node_id(factory, session):
    """Test asset creation with invalid permission."""
    invalid_node = 'unknown-node'
    invalid_permissions = Permissions(public=False, authorized_ids=[invalid_node])
    spec = factory.create_dataset(permissions=invalid_permissions)
    with pytest.raises(substra.exceptions.InvalidRequest) as exc:
        session.add_dataset(spec)
    assert "invalid permission input values" in str(exc.value)


@pytest.mark.parametrize('permissions', [
    Permissions(public=True, authorized_ids=[]),
    Permissions(public=False, authorized_ids=[MSP_IDS[1]]),
])
def test_download_asset_access_granted(permissions, factory, session_1, session_2):
    """Test asset can be downloaded by all permitted nodes."""
    spec = factory.create_dataset(permissions=permissions)
    dataset = session_1.add_dataset(spec)

    content = session_1.download_opener(dataset.key)
    assert content == spec.read_opener()

    content = session_2.download_opener(dataset.key)
    assert content == spec.read_opener()


def test_download_asset_access_restricted(factory, session_1, session_2):
    """Test public asset can be downloaded by all nodes."""
    permissions = Permissions(public=False, authorized_ids=[])
    spec = factory.create_dataset(permissions=permissions)
    dataset = session_1.add_dataset(spec)

    content = session_1.download_opener(dataset.key)
    assert content == spec.read_opener()

    with pytest.raises(substra.exceptions.AuthorizationError):
        session_2.download_opener(dataset.key)


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
                           factory, session_1, session_2):
    """Test merge permissions from dataset and algo asset located on different nodes.

    - dataset and objectives located on node 1
    - algo located on node 2
    - traintuple created on node 2
    """
    # add train data samples / dataset / objective on node 1
    spec = factory.create_dataset(permissions=permissions_1)
    dataset_1 = session_1.add_dataset(spec)
    spec = factory.create_data_sample(test_only=False, datasets=[dataset_1])
    train_data_sample_1 = session_1.add_data_sample(spec)

    # add algo on node 2
    spec = factory.create_algo(permissions=permissions_2)
    algo_2 = session_2.add_algo(spec)

    # add traintuple from node 2
    spec = factory.create_traintuple(
        algo=algo_2,
        dataset=dataset_1,
        data_samples=[train_data_sample_1],
    )
    traintuple = session_1.add_traintuple(spec).future().wait()
    assert traintuple.status == assets.Status.done
    assert traintuple.out_model is not None
    assert traintuple.dataset.worker == session_1.node_id
    tuple_permissions = traintuple.permissions.process
    assert tuple_permissions.public == expected_permissions.public
    assert set(tuple_permissions.authorized_ids) == set(expected_permissions.authorized_ids)


@pytest.mark.skipif(len(MSP_IDS) < 3, reason='requires at least 3 nodes')
def test_merge_permissions_denied_process(factory, network):
    """Test to process asset with merged permissions from 2 other nodes

    - dataset and objectives located on node 1
    - algo located on node 2
    - traintuple created on node 2
    - failed attempt to create testtuple using this traintuple from node 3
    """
    # define sessions one and for all
    session_1 = network.sessions[0]
    session_2 = network.sessions[1]
    session_3 = network.sessions[2]

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
        dataset_1 = session_1.add_dataset(spec)
        spec = factory.create_data_sample(
            test_only=False,
            datasets=[dataset_1],
        )
        datasample_1 = session_1.add_data_sample(spec)
        spec = factory.create_objective(
            data_samples=[datasample_1],
            permissions=permissions_1,
        )
        objective_1 = session_1.add_objective(spec)

        # add algo on node 2
        spec = factory.create_algo(permissions=permissions_2)
        algo_2 = session_2.add_algo(spec)

        # add traintuple from node 2
        spec = factory.create_traintuple(
            algo=algo_2,
            dataset=dataset_1,
            data_samples=[datasample_1],
        )
        traintuple_2 = session_2.add_traintuple(spec).future().wait()

        # failed to add testtuple from node 3
        spec = factory.create_testtuple(
            objective=objective_1,
            traintuple=traintuple_2,
        )

        with pytest.raises(substra.exceptions.AuthorizationError):
            session_3.add_testtuple(spec)
