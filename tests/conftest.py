import dataclasses
import typing

import pytest

import substratest as sbt
from . import settings


def pytest_report_header(config):
    """Print network configuration in pytest header to help configuration debugging."""
    cfg = settings.load()
    messages = [
        f"substra network configuration loaded from: '{cfg.path}'",
        "substra network setup:",
    ]
    for n in cfg.nodes:
        messages.append(f"  - node: name={n.name} msp_id={n.msp_id} address={n.address}")
    return messages


@dataclasses.dataclass
class Network:
    sessions: typing.List[sbt.Session] = dataclasses.field(default_factory=list)


def _get_network():
    """Create network instance from settings."""
    cfg = settings.load()
    sessions = [sbt.Session(
        node_name=n.name,
        node_id=n.msp_id,
        address=n.address,
        user=n.user,
        password=n.password,
    ) for n in cfg.nodes]
    return Network(
        sessions=sessions,
    )


@pytest.fixture
def factory(request):
    """Factory fixture.

    Provide class methods to simply create asset specification in order to add them
    to the substra framework.
    """
    with sbt.AssetsFactory(name=request.node.name) as f:
        yield f


@pytest.fixture
def network():
    """Network fixture.

    Network must started outside of the tests environment and the network is kept
    alive while running all tests.

    Returns an instance of the `Network` class.
    """
    return _get_network()


@pytest.yield_fixture(scope='session')
def data_network():
    """Network fixture with pre-existing assets in all nodes.

    The following asssets will be created for each node:
    - 4 train data samples
    - 1 test data sample
    - 1 dataset
    - 1 objective

    Network must started outside of the tests environment and the network is kept
    alive while running all tests.

    Returns a tuple (factory, Network)
    """
    n = _get_network()

    with sbt.AssetsFactory(name='data-network') as f:
        for sess in n.sessions:

            # create dataset
            spec = f.create_dataset()
            dataset = sess.add_dataset(spec)

            # create train data samples
            for i in range(4):
                spec = f.create_data_sample(datasets=[dataset], test_only=False)
                sess.add_data_sample(spec)

            # create test data sample
            spec = f.create_data_sample(datasets=[dataset], test_only=True)
            test_data_sample = sess.add_data_sample(spec)

            # create objective
            spec = f.create_objective(dataset=dataset, data_samples=[test_data_sample])
            sess.add_objective(spec)

        yield f, n


@pytest.fixture
def session_1(network):
    """Client fixture (first node)."""
    return network.sessions[0]


@pytest.fixture
def session_2(network):
    """Client fixture (second node)."""
    return network.sessions[1]


@pytest.fixture
def node_cfg():
    """Node configuration (first node)."""
    cfg = settings.load()
    return cfg.nodes[0]


@pytest.fixture
def session(network):
    """Client fixture (first node)."""
    return network.sessions[0]
