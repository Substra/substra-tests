import dataclasses
import typing
import uuid

import pytest
import substra

import substratest as sbt
from substratest import settings
from substratest.factory import AugmentedDataset
from substratest.fl_interface import AlgoCategory

TESTS_RUN_UUID = uuid.uuid4().hex  # unique uuid identifying the tests run

pytest_plugins = ["pytest_skipuntil"]


def pytest_report_header(config):
    """Print network configuration in pytest header to help configuration debugging."""
    cfg = settings.Settings.load()
    messages = [
        f"tests run uuid: {TESTS_RUN_UUID}",
        f"substra network configuration loaded from: '{cfg.path}'",
        "substra network setup:",
    ]
    for n in cfg.organizations:
        messages.append(f"  - organization: name={n.name} msp_id={n.msp_id} address={n.address}")
    messages.append(f"substra tools images: {cfg.substra_tools}")
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
    config.addinivalue_line(
        "markers",
        "subprocess_skip: marks that needs to be skip only in subprocess mode "
        "(deselect with '-m \"not subprocess_skip\"')",
    )


def pytest_addoption(parser):
    """Command line arguments to configure the network to be local or remote"""
    parser.addoption(
        "--mode",
        choices=["subprocess", "docker", "remote"],
        default="remote",
        help="Choose the mode on which to run the tests",
    )
    parser.addoption(
        "--nb-train-datasamples",
        default=500,
        type=int,
        help="number of train datasamples to use for the MNIST benchmark",
    )
    parser.addoption(
        "--nb-test-datasamples",
        default=200,
        type=int,
        help="number of test datasamples to use for the MNIST benchmark",
    )


def pytest_collection_modifyitems(config, items):
    """Skip the remote tests if local backend and local tests if remote backend.
    By default, run only on the remote backend.
    """
    mode = substra.BackendType(config.getoption("--mode"))
    if mode != substra.BackendType.REMOTE:
        skip_marker = pytest.mark.skip(reason="remove the --local option to run")
        keyword = "remote_only"
    else:
        skip_marker = pytest.mark.skip(reason="need the --local option to run")
        keyword = "local_only"
    for item in items:
        if keyword in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def client_mode(request):
    mode = request.config.getoption("--mode")
    return substra.BackendType(mode)


class _DataEnv:
    """Test assets.

    Represents all the assets that have been added before the tests.
    """

    def __init__(self, datasets, metrics) -> None:
        self._datasets = [AugmentedDataset(dataset) for dataset in datasets] or []
        self._metrics = metrics or []

    @property
    def datasets(self):
        return self._datasets

    @property
    def metrics(self):
        return self._metrics

    def filter_by(self, organization_id):
        datasets = [d for d in self._datasets if d.owner == organization_id]
        metrics = [o for o in self._metrics if o.owner == organization_id]

        return _DataEnv(metrics=metrics, datasets=datasets)


@dataclasses.dataclass
class Network:
    options: settings.Options
    clients: typing.List[sbt.Client] = dataclasses.field(default_factory=list)


@pytest.fixture
def factory(request, cfg, client_mode):
    """Factory fixture.

    Provide class methods to simply create asset specification in order to add them
    to the substra framework.
    """
    name = f"{TESTS_RUN_UUID}_{request.node.name}"
    with sbt.AssetsFactory(
        name=name,
        cfg=cfg,
        client_debug_local=(client_mode in [substra.BackendType.LOCAL_SUBPROCESS, substra.BackendType.LOCAL_DOCKER]),
    ) as f:
        yield f


@pytest.fixture(scope="session")
def cfg(client_mode):
    if client_mode == substra.BackendType.REMOTE:
        return settings.Settings.load()
    else:
        return settings.Settings.load_local_backend()


@pytest.fixture
def debug_factory(request, cfg):
    """Factory fixture.

    Provide class methods to simply create asset specification in order to add them
    to the substra framework.
    """
    name = f"{TESTS_RUN_UUID}_{request.node.name}"
    with sbt.AssetsFactory(name=name, cfg=cfg, client_debug_local=True) as f:
        yield f


@pytest.fixture(scope="session")
def network(cfg, client_mode):
    """Network fixture.

    Network must be started outside of the tests environment and the network is kept
    alive while running all tests.

    Create network instance from settings.

    Returns an instance of the `Network` class.
    """
    clients = [
        sbt.Client(
            backend_type=client_mode,
            organization_id=n.msp_id,
            address=n.address,
            user=n.user,
            password=n.password,
            future_timeout=cfg.options.future_timeout,
            future_polling_period=cfg.options.future_polling_period,
        )
        for n in cfg.organizations
    ]
    return Network(
        options=cfg.options,
        clients=clients,
    )


@pytest.fixture(scope="session")
def default_data_env(cfg, network, client_mode):
    """Fixture with pre-existing assets in all organizations.

    The following assets will be created for each organization:
    - 5 data samples
    - 1 dataset
    - 1 metric

    Network must started outside of the tests environment and the network is kept
    alive while running all tests.

    Returns the assets created.
    """
    factory_name = f"{TESTS_RUN_UUID}_global"

    with sbt.AssetsFactory(
        name=factory_name,
        cfg=cfg,
        client_debug_local=(client_mode in [substra.BackendType.LOCAL_SUBPROCESS, substra.BackendType.LOCAL_DOCKER]),
    ) as f:
        datasets = []
        metrics = []
        for index, client in enumerate(network.clients):

            # create dataset
            spec = f.create_dataset()
            dataset = client.add_dataset(spec)

            # create data samples
            for _ in range(5):
                spec = f.create_data_sample(datasets=[dataset])
                client.add_data_sample(spec)

            # reload datasets (to ensure they are properly linked with the created data samples)
            dataset = client.get_dataset(dataset.key)
            datasets.append(dataset)

            # create metric
            spec = f.create_algo(category=AlgoCategory.metric, offset=index)
            metric = client.add_algo(spec)
            metrics.append(metric)

        assets = _DataEnv(datasets=datasets, metrics=metrics)
        yield assets


@pytest.fixture
def data_env_1(default_data_env, client_1):
    """Fixture with pre-existing assets in first organization."""
    return default_data_env.filter_by(client_1.organization_id)


@pytest.fixture
def data_env_2(default_data_env, client_2):
    """Fixture with pre-existing assets in second organization."""
    return default_data_env.filter_by(client_2.organization_id)


@pytest.fixture
def default_dataset_1(data_env_1):
    """Fixture with pre-existing dataset in first organization."""
    return AugmentedDataset(data_env_1.datasets[0])


@pytest.fixture
def default_metric_1(data_env_1):
    """Fixture with pre-existing metric in first organization."""
    return data_env_1.metrics[0]


@pytest.fixture
def default_dataset_2(data_env_2):
    """Fixture with pre-existing dataset in second organization."""
    return AugmentedDataset(data_env_2.datasets[0])


@pytest.fixture
def default_metric_2(data_env_2):
    """Fixture with pre-existing metric in second organization."""
    return data_env_2.metrics[0]


@pytest.fixture
def default_dataset(default_dataset_1):
    """Fixture with pre-existing dataset in first organization."""
    return default_dataset_1


@pytest.fixture
def default_metric(default_metric_1):
    """Fixture with pre-existing metric in first organization."""
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
    """Client fixture (first organization)."""
    return network.clients[0]


@pytest.fixture
def client_2(network):
    """Client fixture (second organization)."""
    if len(network.clients) < 2:
        pytest.skip("Not enough organizations to run this test")

    return network.clients[1]


@pytest.fixture
def organization_cfg(cfg):
    """Organization configuration (first organization)."""
    return cfg.organizations[0]


@pytest.fixture(scope="session")
def client(network):
    """Client fixture (first organization)."""
    return network.clients[0]


@pytest.fixture(scope="session")
def clients(network):
    """Clients fixture (all organizations)."""
    return network.clients


@pytest.fixture(scope="session")
def worker(client):
    """Clients fixture (all organizations)."""
    return client.organization_info().organization_id


@pytest.fixture(scope="session")
def workers(clients):
    """Clients fixture (all organizations)."""
    return [client.organization_info().organization_id for client in clients]


@pytest.fixture(scope="session")
def channel(cfg, network):
    """Channel fixture (first organization)."""
    return sbt.Channel(network.clients, cfg.options.organization_sync_timeout)


@pytest.fixture(scope="session")
def hybrid_client(cfg, client):
    """
    Client fixture in hybrid mode (first organization).
    Use it with @pytest.mark.remote_only
    """
    organization = cfg.organizations[0]
    # Hybrid client and client share the same
    # token, otherwise when one connects the other
    # is disconnected.
    return sbt.Client(
        backend_type=substra.BackendType.LOCAL_DOCKER,
        organization_id=organization.msp_id,
        address=organization.address,
        user=organization.user,
        password=organization.password,
        future_timeout=cfg.options.future_timeout,
        future_polling_period=cfg.options.future_polling_period,
        token=client.token,
    )
