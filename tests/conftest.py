import dataclasses
import typing
import uuid

import pytest

import substratest as sbt

from . import settings

TESTS_RUN_UUID = uuid.uuid4().hex  # unique uuid identifying the tests run


def pytest_report_header(config):
    """Print network configuration in pytest header to help configuration debugging."""
    cfg = settings.load()
    messages = [
        f"tests run uuid: {TESTS_RUN_UUID}",
        f"substra network configuration loaded from: '{cfg.path}'",
        "substra network setup:",
    ]
    for n in cfg.nodes:
        messages.append(f"  - node: name={n.name} msp_id={n.msp_id} address={n.address}")
    return messages


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "local_only: marks tests as local backend only (deselect with '-m \"not local_only\"')",
    )
    config.addinivalue_line(
        "markers",
        "remote_only: marks tests as remote backend only (deselect with '-m \"not remote_only\"')",
    )
    config.addinivalue_line(
        "markers",
        "workflows: marks tests as part of a production workflow (deselect with '-m \"not workflows\"')",
    )


def pytest_addoption(parser):
    """Command line arguments to configure the network to be local or remote"""
    parser.addoption(
        "--local",
        action="store_true",
        help="Run the tests on the local backend only. Otherwise run the tests only on the remote backend.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip the remote tests if local backend and local tests if remote backend.
    By default, run only on the remote backend.
    """
    local = config.getoption("--local")
    if local:
        skip_marker = pytest.mark.skip(reason="remove the --local option to run")
        keyword = "remote_only"
    else:
        skip_marker = pytest.mark.skip(reason="need the --local option to run")
        keyword = "local_only"
    for item in items:
        if keyword in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def client_debug_local(request):
    local = request.config.getoption("--local")
    if local:
        return True
    return False


class _DataEnv:
    """Test assets.

    Represents all the assets that have been added before the tests.
    """

    def __init__(self, datasets=None, metrics=None):
        self._datasets = datasets or []
        self._metrics = metrics or []

    @property
    def datasets(self):
        return self._datasets

    @property
    def metrics(self):
        return self._metrics

    def filter_by(self, node_id):
        datasets = [d for d in self._datasets if d.owner == node_id]
        metrics = [o for o in self._metrics if o.owner == node_id]

        return _DataEnv(metrics=metrics, datasets=datasets)


@dataclasses.dataclass
class Network:
    options: settings.Options
    clients: typing.List[sbt.Client] = dataclasses.field(default_factory=list)


@pytest.fixture
def factory(request, client_debug_local):
    """Factory fixture.

    Provide class methods to simply create asset specification in order to add them
    to the substra framework.
    """
    name = f"{TESTS_RUN_UUID}_{request.node.name}"
    with sbt.AssetsFactory(name=name, client_debug_local=client_debug_local) as f:
        yield f


@pytest.fixture
def debug_factory(request):
    """Factory fixture.

    Provide class methods to simply create asset specification in order to add them
    to the substra framework.
    """
    name = f"{TESTS_RUN_UUID}_{request.node.name}"
    with sbt.AssetsFactory(name=name, client_debug_local=True) as f:
        yield f


@pytest.fixture(scope="session")
def network(client_debug_local):
    """Network fixture.

    Network must be started outside of the tests environment and the network is kept
    alive while running all tests.

    Create network instance from settings.

    Returns an instance of the `Network` class.
    """
    if not client_debug_local:
        cfg = settings.load()
    else:
        # TODO check what enable_intermediate_model_removal does
        cfg = settings.load_local_backend()
    clients = [
        sbt.Client(
            debug=client_debug_local,
            node_id=n.msp_id,
            address=n.address,
            user=n.user,
            password=n.password,
        )
        for n in cfg.nodes
    ]
    return Network(
        options=cfg.options,
        clients=clients,
    )


@pytest.fixture(scope="session")
def default_data_env(network, client_debug_local):
    """Fixture with pre-existing assets in all nodes.

    The following assets will be created for each node:
    - 4 train data samples
    - 1 test data sample
    - 1 dataset
    - 1 metric

    Network must started outside of the tests environment and the network is kept
    alive while running all tests.

    Returns the assets created.
    """
    factory_name = f"{TESTS_RUN_UUID}_global"

    with sbt.AssetsFactory(name=factory_name, client_debug_local=client_debug_local) as f:
        datasets = []
        metrics = []
        for index, client in enumerate(network.clients):

            # create dataset
            spec = f.create_dataset()
            dataset = client.add_dataset(spec)

            # create train data samples
            for i in range(4):
                spec = f.create_data_sample(datasets=[dataset], test_only=False)
                client.add_data_sample(spec)

            # create test data sample
            spec = f.create_data_sample(datasets=[dataset], test_only=True)
            _ = client.add_data_sample(spec)

            # reload datasets (to ensure they are properly linked with the created data samples)
            dataset = client.get_dataset(dataset.key)
            datasets.append(dataset)

            # create metric
            spec = f.create_metric(offset=index)
            metric = client.add_metric(spec)
            metrics.append(metric)

        assets = _DataEnv(datasets=datasets, metrics=metrics)
        yield assets


@pytest.fixture
def data_env_1(default_data_env, client_1):
    """Fixture with pre-existing assets in first node."""
    return default_data_env.filter_by(client_1.node_id)


@pytest.fixture
def data_env_2(default_data_env, client_2):
    """Fixture with pre-existing assets in second node."""
    return default_data_env.filter_by(client_2.node_id)


@pytest.fixture
def default_dataset_1(data_env_1):
    """Fixture with pre-existing dataset in first node."""
    return data_env_1.datasets[0]


@pytest.fixture
def default_metric_1(data_env_1):
    """Fixture with pre-existing metric in first node."""
    return data_env_1.metrics[0]


@pytest.fixture
def default_dataset_2(data_env_2):
    """Fixture with pre-existing dataset in second node."""
    return data_env_2.datasets[0]


@pytest.fixture
def default_metric_2(data_env_2):
    """Fixture with pre-existing metric in second node."""
    return data_env_2.metrics[0]


@pytest.fixture
def default_dataset(default_dataset_1):
    """Fixture with pre-existing dataset in first node."""
    return default_dataset_1


@pytest.fixture
def default_metric(default_metric_1):
    """Fixture with pre-existing metric in first node."""
    return default_metric_1


@pytest.fixture
def default_datasets(default_data_env):
    """Fixture with pre-existing datasets."""
    return default_data_env.datasets


@pytest.fixture
def default_metrics(default_data_env):
    """Fixture with pre-existing metrics."""
    return default_data_env.metrics


@pytest.fixture
def client_1(network):
    """Client fixture (first node)."""
    return network.clients[0]


@pytest.fixture
def client_2(network):
    """Client fixture (second node)."""
    if len(network.clients) < 2:
        pytest.skip("Not enough nodes to run this test")

    return network.clients[1]


@pytest.fixture
def node_cfg():
    """Node configuration (first node)."""
    cfg = settings.load()
    return cfg.nodes[0]


@pytest.fixture(scope="session")
def client(network):
    """Client fixture (first node)."""
    return network.clients[0]


@pytest.fixture
def clients(network):
    """Client fixture (first node)."""
    return network.clients


@pytest.fixture(scope="session")
def channel(network):
    """Channel fixture (first node)."""
    return sbt.Channel(clients=network.clients)


@pytest.fixture(scope="session")
def debug_client(client):
    """
    Client fixture in debug mode (first node).
    Use it with @pytest.mark.remote_only
    """
    cfg = settings.load()
    node = cfg.nodes[0]
    # Debug client and client share the same
    # token, otherwise when one connects the other
    # is disconnected.
    return sbt.Client(
        debug=True,
        node_id=node.msp_id,
        address=node.address,
        user=node.user,
        password=node.password,
        token=client.token,
    )
