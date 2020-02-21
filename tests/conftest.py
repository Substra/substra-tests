import dataclasses
import typing
import uuid

import pytest
import pydantic

import substratest as sbt
from . import settings


TESTS_RUN_UUID = uuid.uuid4().hex  # unique uuid identifying the tests run


def pytest_report_header(config):
    """Print network configuration in pytest header to help configuration debugging."""
    cfg = settings.load()
    messages = [
        f"tests run uuid: {TESTS_RUN_UUID}'",
        f"substra network configuration loaded from: '{cfg.path}'",
        "substra network setup:",
    ]
    for n in cfg.nodes:
        messages.append(f"  - node: name={n.name} msp_id={n.msp_id} address={n.address}")
    return messages


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )


class _State(pydantic.BaseModel):
    """Current state.

    Represents all the assets that have been added during the tests.
    """
    datasets: typing.List[sbt.client.assets.Dataset] = []
    test_data_samples: typing.List[sbt.client.assets.DataSampleCreated] = []
    train_data_samples: typing.List[sbt.client.assets.DataSampleCreated] = []
    objectives: typing.List[sbt.client.assets.Objective] = []


@dataclasses.dataclass
class Network:
    options: settings.Options
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
        options=cfg.options,
        sessions=sessions,
    )


@pytest.fixture
def factory(request):
    """Factory fixture.

    Provide class methods to simply create asset specification in order to add them
    to the substra framework.
    """
    name = f"{TESTS_RUN_UUID}_{request.node.name}"
    with sbt.AssetsFactory(name=name) as f:
        yield f


@pytest.fixture
def network():
    """Network fixture.

    Network must started outside of the tests environment and the network is kept
    alive while running all tests.

    Returns an instance of the `Network` class.
    """
    return _get_network()


@pytest.fixture(scope='session')
def global_execution_env():
    """Network fixture with pre-existing assets in all nodes.

    The following asssets will be created for each node:
    - 4 train data samples
    - 1 test data sample
    - 1 dataset
    - 1 objective

    Network must started outside of the tests environment and the network is kept
    alive while running all tests.

    Returns a tuple (factory, state, Network)
    """
    n = _get_network()
    s = _State()
    factory_name = f"{TESTS_RUN_UUID}_global"

    with sbt.AssetsFactory(name=factory_name) as f:
        for sess in n.sessions:

            # create dataset
            spec = f.create_dataset()
            dataset = sess.add_dataset(spec)

            # create train data samples
            for i in range(4):
                spec = f.create_data_sample(datasets=[dataset], test_only=False)
                data_sample = sess.add_data_sample(spec)
                s.train_data_samples.append(data_sample)

            # create test data sample
            spec = f.create_data_sample(datasets=[dataset], test_only=True)
            test_data_sample = sess.add_data_sample(spec)
            s.test_data_samples.append(test_data_sample)

            # reload datasets (to ensure they are properly linked with the created data samples)
            dataset = sess.get_dataset(dataset.key)
            s.datasets.append(dataset)

            # create objective
            spec = f.create_objective(dataset=dataset, data_samples=[test_data_sample])
            objective = sess.add_objective(spec)
            s.objectives.append(objective)

        yield f, s, n


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
