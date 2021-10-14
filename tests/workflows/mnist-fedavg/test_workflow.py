import csv
import pathlib
import typing

import pydantic

import pytest

import substra as sb
import substratest as sbt
from substratest.factory import AlgoCategory

# extra requirements located in requirements-workflows.txt
try:
    import h5py
    import numpy as np
    from sklearn.datasets import fetch_openml as sk_fetch_openml
except ImportError as e:
    _EXTRA_IMPORT_ERROR = e
else:
    _EXTRA_IMPORT_ERROR = None


_MNIST_DATASET_NAME = "mnist_784"

_PARENT_DIR = pathlib.Path(__file__).parent.absolute()

_CACHE_PATH = _PARENT_DIR / ".cache"

_OPENER = _PARENT_DIR / "assets" / "opener.py"
_METRICS = _PARENT_DIR / "assets" / "metrics.py"
_AGGREGATE_ALGO = _PARENT_DIR / "assets" / "aggregate_algo.py"
_COMPOSITE_ALGO = _PARENT_DIR / "assets" / "composite_algo.py"

_SEED = 42

_ML_BATCH_SIZE = 32
_ML_NUM_UPDATES = 50

_INPUT_SIZE = 28

# represents the number of orgs: it will be used to split the data into NB_ORGS parts
_NB_ORGS = 2

# this image is built and pushed by the connect-tools repository
_IMAGE = (
    f"{sbt.factory.DEFAULT_TOOLS_BASE_IMAGE}:"
    f"{sbt.factory.DEFAULT_TOOLS_VERSION}-workflows"
)

_ALGO_DOCKERFILE = f"""
FROM {_IMAGE}
COPY algo.py .
ENTRYPOINT ["python3", "algo.py"]
"""

_METRICS_DOCKERFILE = f"""
FROM {_IMAGE}
COPY metrics.py .
ENTRYPOINT ["python3", "metrics.py"]
"""


@pytest.fixture(autouse=True)
def check_imports():
    """Check imports at runtime as all external dependencies may not be installed.

    If this fixture fails, install requirements in file requirements-workflows.txt
    """
    if _EXTRA_IMPORT_ERROR:
        message = "Install requirements-workflows.txt should solve the issue."
        raise ImportError(message) from _EXTRA_IMPORT_ERROR


@pytest.fixture
def clients(client_debug_local, clients):
    """Override clients fixture to return one client per org.

    The compute plan shape does not depend depend on the network topology (that is to
    say the number of orgs). The compute plan shape will be exactly the same with /
    without the local debug mode.
    """
    if client_debug_local:
        assert len(clients) == 1
        return [clients[0]] * _NB_ORGS
    else:
        assert len(clients) <= _NB_ORGS
        return clients[:_NB_ORGS]


@pytest.fixture
def mnist_train_test():
    """Download MNIST data using sklearn and store it to disk.

    This will check if MNIST is present in a cache data folder. If not it will download
    it. Then it will be reshape to N, C, H, W format and stored into a train and a test
    h5 files.

    This fixture returns the train and the test filepaths.
    """
    _CACHE_PATH.mkdir(parents=True, exist_ok=True)

    train_data_filepath = _CACHE_PATH / "train.h5"
    test_data_filepath = _CACHE_PATH / "test.h5"

    if train_data_filepath.exists() and test_data_filepath.exists():
        return train_data_filepath, test_data_filepath

    mnist = sk_fetch_openml(_MNIST_DATASET_NAME)

    # MNIST has 70k samples, 60k are used for trained, 10k for testing
    nb_train = 60000
    X_train, y_train = mnist.data[:nb_train], mnist.target[:nb_train]
    X_test, y_test = mnist.data[nb_train:], mnist.target[nb_train:]

    # Preprocess training data and save it in HDF5 files
    X_train = X_train.values.reshape(
        (-1, 1, _INPUT_SIZE, _INPUT_SIZE)).astype(np.float32) / 255.0
    y_train = y_train.astype(int)
    with h5py.File(train_data_filepath, "w") as fp:
        fp["X"] = X_train
        fp["y"] = y_train

    # Repeat procedure for test data
    X_test = X_test.values.reshape(
        (-1, 1, _INPUT_SIZE, _INPUT_SIZE)).astype(np.float32) / 255.0
    y_test = y_test.astype(int)
    with h5py.File(test_data_filepath, "w") as fp:
        fp["X"] = X_test
        fp["y"] = y_test

    return train_data_filepath, test_data_filepath


def _split_into_datasamples(
        path: pathlib.Path,
        type_: str,
        destination: pathlib.Path,
        nb_samples: int,
        nb_orgs: int,
):
    """Split h5 file into datasample folders.

    Each datasample contains a single row to create a workflow similar to what is done
    in connectlib.

    Returns a list of folders for each org.
    """
    with h5py.File(path, "r") as fp:
        data_X = np.array(fp["X"])
        data_y = np.array(fp["y"])

    # Shuffle data
    np.random.seed(_SEED)
    indices = np.arange(data_X.shape[0])
    np.random.shuffle(indices)
    data_X = data_X[indices]
    data_y = data_y[indices]
    assert nb_samples < data_X.shape[0]

    orgs = [f"org{org_idx}" for org_idx in range(nb_orgs)]
    folders_per_org = [list() for _ in range(nb_orgs)]

    for i in range(nb_samples):
        sample_idx = (i // 2) + 1
        pseudoid = i + 1
        org_idx = i % nb_orgs
        foldername = f"sample_{sample_idx:08d}"
        folder = destination / orgs[org_idx] / f"{type_}_data_samples" / foldername
        folder.mkdir(parents=True, exist_ok=True)

        folders_per_org[org_idx].append(folder)

        filename_npy = folder / f"{type_}_{pseudoid:08d}.npy"
        np.save(str(filename_npy), data_X[i])

        filename_csv = folder / f"{type_}_{pseudoid:08d}.csv"
        with open(filename_csv, "w") as fp:
            writer = csv.writer(fp, delimiter=";")
            writer.writerow(["pseudoid", "label"])
            writer.writerow([f"{type_}_{pseudoid:08d}", data_y[i]])

    return folders_per_org


class _DatasampleFolders(pydantic.BaseModel):
    """Datasample folders for a single org."""
    train: typing.List[str] = []
    test: typing.List[str] = []


@pytest.fixture
def datasamples_folders(tmpdir, mnist_train_test):
    """Split input dataset into datasamples for each orgs."""
    train_path, test_path = mnist_train_test
    tmpdir = pathlib.Path(tmpdir)

    folders = [_DatasampleFolders() for org_idx in range(_NB_ORGS)]

    # XXX this example is using 700 samples (out of 70k) as it is sufficient to have
    #     a good performance.
    train_folders = _split_into_datasamples(
        train_path, "train", tmpdir, nb_samples=500, nb_orgs=_NB_ORGS)
    test_folders = _split_into_datasamples(
        test_path, "test", tmpdir, nb_samples=200, nb_orgs=_NB_ORGS)

    for train, test, folder in zip(train_folders, test_folders, folders):
        folder.train = train
        folder.test = test
    return folders


class _InputsSubset(pydantic.BaseModel):
    """Inputs objects required to launch a FL pipeline on a Connect org.

    One subset per org.
    """
    dataset: sb.sdk.models.Dataset = None
    metric: sb.sdk.models.Metric = None
    train_data_sample_keys: typing.List[str] = []


class _Inputs(pydantic.BaseModel):
    """Inputs objects required to launch a FL pipeline on a Connect Network."""
    # XXX datasets must have the same order as the clients fixture
    datasets: typing.List[_InputsSubset]
    composite_algo: sb.sdk.models.Algo = None
    aggregate_algo: sb.sdk.models.Algo = None


@pytest.fixture
def inputs(datasamples_folders, factory, clients):
    """Register for each orgs substra inputs (dataset, datasamples and metric)."""
    results = _Inputs(datasets=[_InputsSubset() for _ in range(_NB_ORGS)])

    batch_size = 100

    def _split_into_chunks(items, size):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(items), size):
            yield items[i:i + size]

    for client, folders, res in zip(clients, datasamples_folders, results.datasets):
        spec = factory.create_dataset(py_script=_OPENER.open().read())
        res.dataset = client.add_dataset(spec)

        train_keys = []
        for paths in _split_into_chunks(folders.train, batch_size):
            keys_per_batch = client.add_data_samples(sbt.factory.DataSampleBatchSpec(
                paths=[str(p) for p in paths],
                test_only=False,
                data_manager_keys=[res.dataset.key],
            ))
            train_keys.extend(keys_per_batch)

        test_keys = []
        for paths in _split_into_chunks(folders.test, batch_size):
            keys_per_batch = client.add_data_samples(sbt.factory.DataSampleBatchSpec(
                paths=[str(p) for p in paths],
                test_only=True,
                data_manager_keys=[res.dataset.key],
            ))
            test_keys.extend(keys_per_batch)

        # XXX is it required to link dataset with datasamples ?

        spec = factory.create_metric(
            dockerfile=_METRICS_DOCKERFILE,
            py_script=_METRICS.open().read(),
        )
        res.metric = client.add_metric(spec)

        # refresh dataset (to be up-to-date with added samples)
        res.dataset = client.get_dataset(res.dataset.key)
        # store also the train keys as the order might not be the same in the
        # dataset.train_data_sample_keys field
        res.train_data_sample_keys = train_keys

    client = clients[0]
    spec = factory.create_algo(
        AlgoCategory.composite,
        py_script=_COMPOSITE_ALGO.open().read(),
        dockerfile=_ALGO_DOCKERFILE,
    )
    results.composite_algo = client.add_algo(spec)

    spec = factory.create_algo(
        AlgoCategory.aggregate,
        py_script=_AGGREGATE_ALGO.open().read(),
        dockerfile=_ALGO_DOCKERFILE,
    )
    results.aggregate_algo = client.add_algo(spec)

    return results


@pytest.mark.slow
@pytest.mark.workflows
def test_mnist(factory, inputs, clients):
    client = clients[0]
    nb_rounds = 20
    testing_rounds = (1, 5, 10, 15, 20)
    cp_spec = factory.create_compute_plan()

    aggregate_worker = client.node_id

    trunk_model_perms = [sbt.factory.Permissions(
        public=False,
        authorized_ids=[aggregate_worker, c.node_id],
    ) for c in clients]

    # emtpy initialization for first round
    composite_specs = [None] * len(inputs.datasets)
    aggregate_spec = None

    # At each round all node samples are used by the composite traintuple. This is due
    # to the fact that the algo processes 32 (batch_size) * 50 (num_updates) samples,
    # and that the total amount of samples per node is smaller than this value.
    assert all(
        [len(org.dataset.train_data_sample_keys) < _ML_BATCH_SIZE * _ML_NUM_UPDATES
         for org in inputs.datasets]
    )

    # next rounds
    for round_idx in range(nb_rounds):

        composite_specs = [cp_spec.add_composite_traintuple(
            composite_algo=inputs.composite_algo,
            dataset=org_inputs.dataset,
            data_samples=org_inputs.train_data_sample_keys,
            in_head_model=composite_specs[idx],
            in_trunk_model=aggregate_spec,
            out_trunk_model_permissions=trunk_model_perms[idx],
        ) for idx, org_inputs in enumerate(inputs.datasets)]

        aggregate_spec = cp_spec.add_aggregatetuple(
            aggregate_algo=inputs.aggregate_algo,
            worker=aggregate_worker,
            in_models=composite_specs,
        )

        # add testtuples for specified rounds
        if round_idx + 1 in testing_rounds:
            for idx, org_inputs in enumerate(inputs.datasets):
                cp_spec.add_testtuple(
                    traintuple_spec=composite_specs[idx],
                    metrics=[org_inputs.metric],
                    dataset=org_inputs.dataset,
                    data_samples=org_inputs.dataset.test_data_sample_keys,
                )

    cp = client.add_compute_plan(cp_spec)
    cp = client.wait(cp, timeout=30 * 60 * 60)

    # display all testtuples performances
    testtuples = client.list_compute_plan_testtuples(cp.key)
    testtuples = sorted(testtuples, key=lambda x: (x.rank, x.worker))
    for testtuple in testtuples:
        print(
            f"testtuple({testtuple.worker}) - {testtuple.rank} "
            f"perf: {list(testtuple.test.perfs.values())[0]}"
        )
    # check perf is as good as expected: after 20 rounds we expect a performance of
    # around 0.86. To avoid a flaky test a lower performance is expected.
    mininum_expected_perf = 0.85
    assert all([
        list(testtuple.test.perfs.values())[0] > mininum_expected_perf
        for testtuple in testtuples[-_NB_ORGS:]
    ])
