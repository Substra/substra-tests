import os
import shutil
import tempfile

import substra

import pytest

import substratest as sbt


def test_connection_to_nodes(network):
    """Connect to each substra nodes using the session."""
    for session in network.sessions:
        session.list_algo()


def test_add_dataset(factory, session):
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    dataset_copy = session.get_dataset(dataset.key)
    assert dataset == dataset_copy


def test_add_dataset_conflict(factory, session):
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    with pytest.raises(substra.exceptions.AlreadyExists):
        session.add_dataset(spec)

    dataset_copy = session.add_dataset(spec, exist_ok=True)
    assert dataset == dataset_copy


def test_add_data_sample(factory, session):
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    session.add_data_sample(spec)


def test_add_data_sample_located_in_shared_path(factory, session, node_cfg):
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])

    # move data sample to substra backend shared file system
    dst = node_cfg.shared_path
    dst = dst if dst.endswith('/') else dst + '/'
    dst = tempfile.mkdtemp(dir=dst)
    shutil.move(spec.path, dst)

    # update spec path and add data sample
    spec.path = os.path.join(dst, os.path.basename(spec.path))
    session.add_data_sample(spec, local=False)  # should not raise


def test_add_objective(factory, session):
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    data_sample = session.add_data_sample(spec)

    spec = factory.create_objective(dataset=dataset, data_samples=[data_sample])
    objective = session.add_objective(spec)
    objective_copy = session.get_objective(objective.key)
    assert objective == objective_copy


def test_add_algo(factory, session):
    spec = factory.create_algo()
    algo = session.add_algo(spec)

    algo_copy = session.get_algo(algo.key)
    assert algo == algo_copy


def test_list_nodes(network, session):
    """Nodes are properly registered and list nodes returns expected nodes."""
    nodes = session.list_node()
    node_ids = [n.id for n in nodes]
    network_node_ids = [s.node_id for s in network.sessions]
    # check all nodes configured are correctly registered
    assert set(network_node_ids).issubset(set(node_ids))


@pytest.mark.parametrize(
    'asset_type', sbt.assets.AssetType.can_be_listed(),
)
def test_list_asset(asset_type, session):
    """Simple check that list_asset method can be called without raising errors."""
    method = getattr(session, f'list_{asset_type.name}')
    method()  # should not raise
