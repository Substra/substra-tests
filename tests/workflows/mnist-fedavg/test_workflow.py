import pathlib
import typing

import pydantic
import pytest
import substra as sb

import substratest as sbt
from substratest.factory import DEFAULT_FUNCTION_NAME
from substratest.factory import AugmentedDataset
from substratest.factory import FunctionCategory
from substratest.fl_interface import FLTaskInputGenerator
from substratest.fl_interface import FLTaskOutputGenerator
from substratest.fl_interface import OutputIdentifiers
from substratest.settings import Settings

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
_AGGREGATE_FUNCTION = _PARENT_DIR / "assets" / "aggregate_function.py"
_COMPOSITE_FUNCTION = _PARENT_DIR / "assets" / "composite_function.py"
_PREDICT_FUNCTION = _PARENT_DIR / "assets" / "composite_function.py"

_SEED = 1

_INPUT_SIZE = 28

# represents the number of orgs: it will be used to split the data into NB_ORGS parts
_NB_ORGS = 2


_EXPECTED_RESULTS = {
    (500, 200): [0.8, 0.8, 0.8, 0.85, 0.78, 0.84, 0.78, 0.85, 0.79, 0.85],  # E2E
    (60000, 10000): [0.8118, 0.7792, 0.9042, 0.8814, 0.9176, 0.9348, 0.9434, 0.9456, 0.9564, 0.9556],  # benchmark
}


@pytest.fixture
def function_dockerfile(cfg: Settings) -> str:
    return (
        f"FROM {cfg.substra_tools.image_workflows}\n"
        f"COPY function.py .\n"
        f'ENTRYPOINT ["python3", "function.py", "--function-name", "{{method_name}}"]\n'
    )


@pytest.fixture(autouse=True)
def check_imports():
    """Check imports at runtime as all external dependencies may not be installed.

    If this fixture fails, install requirements in file requirements-workflows.txt
    """
    if _EXTRA_IMPORT_ERROR:
        message = "Install requirements-workflows.txt should solve the issue."
        raise ImportError(message) from _EXTRA_IMPORT_ERROR


@pytest.fixture
def clients(clients):
    """Override clients fixture to return one client per org.

    The compute plan shape does not depend depend on the network topology (that is to
    say the number of orgs). The compute plan shape will be exactly the same with /
    without the local debug mode.
    """
    assert len(clients) >= _NB_ORGS
    return clients[:_NB_ORGS]


@pytest.fixture
def workers(clients):
    """Override workers fixture to return one worker per org.

    The compute plan shape does not depend depend on the network topology (that is to
    say the number of orgs). The compute plan shape will be exactly the same with /
    without the local debug mode.
    """
    return [client.organization_info().organization_id for client in clients]


@pytest.fixture
def mnist_train_test(cfg: Settings):
    """Download MNIST data using sklearn and store it to disk.

    This will check if MNIST is present in a cache data folder. If not it will download
    it. Then it will be reshape to N, C, H, W format and stored into a train and a test
    h5 files.

    This fixture returns the train and the test filepaths.
    """
    _CACHE_PATH.mkdir(parents=True, exist_ok=True)

    train_data_filepath = _CACHE_PATH / "train.h5"
    test_data_filepath = _CACHE_PATH / "test.h5"

    mnist = sk_fetch_openml(_MNIST_DATASET_NAME)

    nb_train = cfg.mnist_workflow.train_samples
    nb_test = cfg.mnist_workflow.test_samples

    assert nb_train + nb_test <= 70000, "MNIST has 70k samples"

    print(f"nb of train datasamples: {nb_train}, test datasamples: {nb_test}")

    indices = np.arange(mnist.data.shape[0])
    rng = np.random.default_rng(seed=_SEED)
    rng.shuffle(indices)

    train_indices = indices[:nb_train]
    test_indices = indices[nb_train : nb_train + nb_test]

    X_train, y_train = mnist.data.iloc[train_indices], mnist.target.iloc[train_indices]
    X_test, y_test = mnist.data.iloc[test_indices], mnist.target.iloc[test_indices]
    assert len(X_train) == len(y_train) == nb_train
    assert len(X_test) == len(y_test) == nb_test

    # Preprocess training data and save it in HDF5 files
    X_train = X_train.values.reshape((-1, 1, _INPUT_SIZE, _INPUT_SIZE)).astype(np.float32) / 255.0
    y_train = y_train.astype(int)
    with h5py.File(train_data_filepath, "w") as fp:
        fp["X"] = X_train
        fp["y"] = y_train

    # Repeat procedure for test data
    X_test = X_test.values.reshape((-1, 1, _INPUT_SIZE, _INPUT_SIZE)).astype(np.float32) / 255.0
    y_test = y_test.astype(int)
    with h5py.File(test_data_filepath, "w") as fp:
        fp["X"] = X_test
        fp["y"] = y_test

    return train_data_filepath, test_data_filepath


def _split_into_datasamples(
    path: pathlib.Path,
    type_: str,
    destination: pathlib.Path,
    nb_orgs: int,
):
    """Split h5 file into datasample folders.

    Each datasample contains a single row to create a workflow similar to what is done
    in substrafl.

    Returns a list of folders for each org.
    """
    with h5py.File(path, "r") as fp:
        data_X = np.array(fp["X"])
        data_y = np.array(fp["y"])

    orgs = [f"org{org_idx}" for org_idx in range(nb_orgs)]
    folders_per_org = list()

    len_X = len(data_X) // nb_orgs
    len_y = len(data_y) // nb_orgs

    for org_idx in range(nb_orgs):
        # split datasamples into chunks for each org
        org_data_X = data_X[org_idx * len_X : (org_idx + 1) * len_X]
        org_data_y = data_y[org_idx * len_y : (org_idx + 1) * len_y]

        assert len(org_data_X) == len_X
        assert len(org_data_y) == len_y

        folder = destination / orgs[org_idx] / f"{type_}_data_samples"
        folder.mkdir(parents=True, exist_ok=True)

        folders_per_org.append(folder)

        filename_X_npy = folder / "x.npy"
        np.save(str(filename_X_npy), org_data_X)

        filename_y_npy = folder / "y.npy"
        np.save(str(filename_y_npy), org_data_y)

    return folders_per_org


class _DatasampleFolders(pydantic.BaseModel):
    """Datasample folders for a single org."""

    train: str = None
    test: str = None


@pytest.fixture
def datasamples_folders(tmpdir, mnist_train_test):
    """Split input dataset into datasamples for each orgs.
    Total available datasamples: 70k
    """
    train_path, test_path = mnist_train_test
    tmpdir = pathlib.Path(tmpdir)

    folders = [_DatasampleFolders() for _ in range(_NB_ORGS)]

    train_folders = _split_into_datasamples(train_path, "train", tmpdir, nb_orgs=_NB_ORGS)
    test_folders = _split_into_datasamples(test_path, "test", tmpdir, nb_orgs=_NB_ORGS)

    for train, test, folder in zip(train_folders, test_folders, folders):
        folder.train = train
        folder.test = test
    return folders


class _InputsSubset(pydantic.BaseModel):
    """Inputs objects required to launch a FL pipeline on a Substra org.
    One subset per org.
    """

    dataset: AugmentedDataset = None
    metric: sb.sdk.models.Function = None
    train_data_sample_keys: typing.List[str] = []

    class Config:
        arbitrary_types_allowed = True


class _Inputs(pydantic.BaseModel):
    """Inputs objects required to launch a FL pipeline on a Substra Network."""

    # XXX datasets must have the same order as the clients fixture
    datasets: typing.List[_InputsSubset]
    composite_function: sb.sdk.models.Function = None
    aggregate_function: sb.sdk.models.Function = None
    predict_function: sb.sdk.models.Function = None


@pytest.fixture
def inputs(datasamples_folders, factory, clients, channel, function_dockerfile):
    """Register for each orgs substra inputs (dataset, datasamples and metric)."""
    results = _Inputs(datasets=[_InputsSubset() for _ in range(_NB_ORGS)])

    for client, folders, res in zip(clients, datasamples_folders, results.datasets):
        spec = factory.create_dataset(py_script=_OPENER.open().read())
        res.dataset = client.add_dataset(spec)

        train_keys = client.add_data_samples(
            sbt.factory.DataSampleBatchSpec(
                paths=[str(folders.train)],
                data_manager_keys=[res.dataset.key],
            )
        )
        test_keys = client.add_data_samples(
            sbt.factory.DataSampleBatchSpec(
                paths=[str(folders.test)],
                data_manager_keys=[res.dataset.key],
            )
        )

        metric_dockerfile = function_dockerfile.format(method_name=DEFAULT_FUNCTION_NAME[FunctionCategory.metric])

        spec = factory.create_function(
            category=FunctionCategory.metric, dockerfile=metric_dockerfile, py_script=_METRICS.open().read()
        )

        res.metric = client.add_function(spec)

        # refresh dataset (to be up-to-date with added samples)
        res.dataset = AugmentedDataset(client.get_dataset(res.dataset.key))
        res.dataset.set_train_test_dasamples(
            train_data_sample_keys=train_keys,
            test_data_sample_keys=test_keys,
        )
        # store also the train keys as the order might not be the same in the
        # dataset.train_data_sample_keys field
        res.train_data_sample_keys = train_keys

    client = clients[0]
    spec = factory.create_function(
        FunctionCategory.composite,
        py_script=_COMPOSITE_FUNCTION.open().read(),
        dockerfile=function_dockerfile.format(method_name=DEFAULT_FUNCTION_NAME[FunctionCategory.composite]),
    )
    results.composite_function = client.add_function(spec)

    spec = factory.create_function(
        FunctionCategory.aggregate,
        py_script=_AGGREGATE_FUNCTION.open().read(),
        dockerfile=function_dockerfile.format(method_name=DEFAULT_FUNCTION_NAME[FunctionCategory.aggregate]),
    )
    results.aggregate_function = client.add_function(spec)

    spec = factory.create_function(
        FunctionCategory.predict_composite,
        py_script=_PREDICT_FUNCTION.open().read(),
        dockerfile=function_dockerfile.format(method_name=DEFAULT_FUNCTION_NAME[FunctionCategory.predict_composite]),
    )
    results.predict_function = client.add_function(spec)
    # ensure last registered asset is synchronized on all organizations
    channel.wait_for_asset_synchronized(results.aggregate_function)

    return results


@pytest.mark.slow
@pytest.mark.workflows
def test_mnist(factory, inputs, clients, cfg: Settings, workers: typing.List[str]):
    client = clients[0]
    nb_rounds = 20
    testing_rounds = (1, 5, 10, 15, 20)
    cp_spec = factory.create_compute_plan()

    aggregate_worker = client.organization_id

    # emtpy initialization for first round
    composite_specs = [None] * len(inputs.datasets)
    aggregate_spec = None

    # At each round, the whole dataset is passed to the task
    # For the e2e test, the train task uses 50 updates * 32 batch size
    # samples and there are 500 train samples so it goes over the whole dataset
    # several times.
    # For the benchmark, there are 60'000 samples so it does not use the whole dataset
    # but it does not make much difference in the final result and simplifies the code
    # to pass the whole dataset.

    # next rounds
    for round_idx in range(nb_rounds):
        for idx, org_inputs in enumerate(inputs.datasets):
            if aggregate_spec:
                input_models = FLTaskInputGenerator.composite_to_local(
                    composite_specs[idx].task_id
                ) + FLTaskInputGenerator.aggregate_to_shared(aggregate_spec.task_id)
            else:
                input_models = []

            composite_specs[idx] = cp_spec.create_composite_traintask(
                composite_function=inputs.composite_function,
                inputs=org_inputs.dataset.train_data_inputs + input_models,
                outputs=FLTaskOutputGenerator.composite_traintask(
                    shared_authorized_ids=[aggregate_worker, clients[idx].organization_id],
                    local_authorized_ids=[clients[idx].organization_id],
                ),
                metadata={
                    "round_idx": round_idx,
                },
                worker=workers[idx],
            )

        aggregate_spec = cp_spec.create_aggregatetask(
            aggregate_function=inputs.aggregate_function,
            worker=aggregate_worker,
            inputs=FLTaskInputGenerator.composites_to_aggregate(
                [composite_spec.task_id for composite_spec in composite_specs]
            ),
            metadata={
                "round_idx": round_idx,
            },
        )

        # add testtasks for specified rounds
        if round_idx + 1 in testing_rounds:
            for idx, org_inputs in enumerate(inputs.datasets):
                predicttask_spec = cp_spec.create_predicttask(
                    function=inputs.predict_function,
                    inputs=org_inputs.dataset.test_data_inputs
                    + FLTaskInputGenerator.composite_to_predict(composite_specs[idx].task_id),
                    metadata={
                        "round_idx": round_idx,
                    },
                    worker=workers[idx],
                )
                cp_spec.create_testtask(
                    function=org_inputs.metric,
                    inputs=org_inputs.dataset.test_data_inputs
                    + FLTaskInputGenerator.predict_to_test(predicttask_spec.task_id),
                    metadata={
                        "round_idx": round_idx,
                    },
                    worker=workers[idx],
                )

    cp = client.add_compute_plan(cp_spec)
    cp = client.wait(cp, timeout=30 * 60 * 60)

    # display all testtasks performances
    tasks = client.list_compute_plan_tasks(cp.key)
    testtasks = [t for t in tasks if OutputIdentifiers.performance in t.outputs]
    testtasks = sorted(testtasks, key=lambda x: (x.rank, x.worker))
    for testtask in testtasks:
        print(
            f"testtask({testtask.worker}) - rank {testtask.rank} "
            f"- round {testtask.metadata['round_idx']} "
            f"perf: {testtask.outputs[OutputIdentifiers.performance]}"
        )

    nb_samples = (cfg.mnist_workflow.train_samples, cfg.mnist_workflow.test_samples)

    # performance should be deterministic a fixed number of samples:
    if nb_samples in _EXPECTED_RESULTS.keys():
        expected_perf = _EXPECTED_RESULTS[nb_samples]
        assert all(
            [
                testtask.outputs[OutputIdentifiers.performance].value == pytest.approx(perf)
                for (perf, testtask) in zip(expected_perf, testtasks)
            ]
        )
    else:
        # check perf is as good as expected: after 20 rounds we expect a performance of
        # around 0.86. To avoid a flaky test a lower performance is expected.
        mininum_expected_perf = 0.85
        assert all([testtask.outputs[OutputIdentifiers.performance] > mininum_expected_perf for testtask in testtasks])
