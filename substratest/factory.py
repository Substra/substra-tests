import abc
import dataclasses
import pathlib
import random
import shutil
import tempfile
import typing
import uuid

from . import utils, assets


DEFAULT_DATA_SAMPLE_FILENAME = 'data.csv'

DEFAULT_SUBSTRATOOLS_VERSION = '0.3.0'

# TODO improve opener get_X/get_y methods
# TODO improve metrics score method

DEFAULT_OPENER_SCRIPT = """
import json
import substratools as tools
class TestOpener(tools.Opener):
    def get_X(self, folders):
        return folders
    def get_y(self, folders):
        return folders
    def fake_X(self):
        return 'fakeX'
    def fake_y(self):
        return 'fakey'
    def get_predictions(self, path):
        with open(path) as f:
            return json.load(f)
    def save_predictions(self, y_pred, path):
        with open(path, 'w') as f:
            return json.dump(y_pred, f)
"""

DEFAULT_METRICS_SCRIPT = """
import json
import substratools as tools
class TestMetrics(tools.Metrics):
    def score(self, y_true, y_pred):
        return 101
if __name__ == '__main__':
    tools.metrics.execute(TestMetrics())
"""

DEFAULT_ALGO_SCRIPT = f"""
import json
import substratools as tools
class TestAlgo(tools.Algo):
    def train(self, X, y, models, rank):
        return [0, 42], [0, 1]
    def predict(self, X, model):
        return [0, 99]
    def load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)
if __name__ == '__main__':
    tools.algo.execute(TestAlgo())
"""

DEFAULT_AGGREGATE_ALGO_SCRIPT = f"""
import json
import substratools as tools
class TestAggregateAlgo(tools.AggregateAlgo):
    def aggregate(self, models, rank):
        return [0, 66]
    def load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)
if __name__ == '__main__':
    tools.algo.execute(TestAggregateAlgo())
"""

DEFAULT_COMPOSITE_ALGO_SCRIPT = f"""
import json
import substratools as tools
class TestCompositeAlgo(tools.CompositeAlgo):
    def train(self, X, y, head_model, trunk_model, rank):
        return [0, 42], [0, 1], [0, 2]
    def predict(self, X, head_model, trunk_model):
        return [0, 99]
    def load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)
if __name__ == '__main__':
    tools.algo.execute(TestCompositeAlgo())
"""

INVALID_ALGO_SCRIPT = DEFAULT_ALGO_SCRIPT.replace('train', 'naitr')

DEFAULT_METRICS_DOCKERFILE = f"""
FROM substrafoundation/substra-tools:{DEFAULT_SUBSTRATOOLS_VERSION}
COPY metrics.py .
ENTRYPOINT ["python3", "metrics.py"]
"""

DEFAULT_ALGO_DOCKERFILE = f"""
FROM substrafoundation/substra-tools:{DEFAULT_SUBSTRATOOLS_VERSION}
COPY algo.py .
ENTRYPOINT ["python3", "algo.py"]
"""


def random_uuid():
    return uuid.uuid4().hex


class Counter:
    def __init__(self):
        self._idx = -1

    def inc(self):
        self._idx += 1
        return self._idx


@dataclasses.dataclass
class _Spec(abc.ABC):
    """Asset specification base class."""

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Permissions:
    public: bool
    authorized_ids: typing.List[str]


DEFAULT_PERMISSIONS = Permissions(public=True, authorized_ids=[])


@dataclasses.dataclass
class DataSampleSpec(_Spec):
    path: str
    test_only: bool
    data_manager_keys: typing.List[str]


@dataclasses.dataclass
class DatasetSpec(_Spec):
    name: str
    data_opener: str
    type: str
    description: str
    permissions: Permissions = None

    def read_opener(self):
        with open(self.data_opener, 'rb') as f:
            return f.read()


@dataclasses.dataclass
class ObjectiveSpec(_Spec):
    name: str
    description: str
    metrics_name: str
    metrics: str
    test_data_sample_keys: typing.List[str]
    test_data_manager_key: str
    permissions: Permissions = None


@dataclasses.dataclass
class _AlgoSpec(_Spec):
    name: str
    description: str
    file: str
    permissions: Permissions = None


@dataclasses.dataclass
class AlgoSpec(_AlgoSpec):
    pass


@dataclasses.dataclass
class AggregateAlgoSpec(_AlgoSpec):
    pass


@dataclasses.dataclass
class CompositeAlgoSpec(_AlgoSpec):
    pass


@dataclasses.dataclass
class TraintupleSpec(_Spec):
    algo_key: str
    objective_key: str
    data_manager_key: str
    train_data_sample_keys: str
    in_models_keys: str
    tag: str
    compute_plan_id: str
    rank: int = None


@dataclasses.dataclass
class AggregatetupleSpec(_Spec):
    algo_key: str
    objective_key: str
    worker: str
    in_models_keys: str
    tag: str
    compute_plan_id: str
    rank: int = None


@dataclasses.dataclass
class CompositeTraintupleSpec(_Spec):
    algo_key: str
    objective_key: str
    data_manager_key: str
    train_data_sample_keys: str
    in_head_model_key: str
    in_trunk_model_key: str
    tag: str
    compute_plan_id: str
    out_trunk_model_permissions: typing.Dict
    rank: int = None


@dataclasses.dataclass
class TesttupleSpec(_Spec):
    traintuple_key: str
    tag: str


@dataclasses.dataclass
class ComputePlanTraintupleSpec:
    data_manager_key: str
    train_data_sample_keys: str
    traintuple_id: str
    in_models_ids: typing.List[str]
    tag: str


@dataclasses.dataclass
class ComputePlanTesttupleSpec:
    traintuple_id: str
    tag: str


def _get_key(obj, field='key'):
    """Get key from asset/spec or key."""
    if isinstance(obj, str):
        return obj
    return getattr(obj, field)


def _get_keys(obj, field='key'):
    """Get keys from asset/spec or key.

    This is particularly useful for data samples to accept as input args a list of keys
    and a list of data samples.
    """
    if not obj:
        return []
    return [_get_key(x, field=field) for x in obj]


@dataclasses.dataclass
class ComputePlanSpec(_Spec):
    algo_key: str
    objective_key: str
    traintuples: typing.List[ComputePlanTraintupleSpec]
    testtuples: typing.List[ComputePlanTesttupleSpec]

    def add_traintuple(self, dataset, data_samples, traintuple_specs=None, tag=None):
        traintuple_specs = traintuple_specs or []
        spec = ComputePlanTraintupleSpec(
            traintuple_id=random_uuid(),
            data_manager_key=dataset.key,
            train_data_sample_keys=_get_keys(data_samples),
            in_models_ids=[t.traintuple_id for t in traintuple_specs],
            tag=tag or '',
        )
        self.traintuples.append(spec)
        return spec

    def add_testtuple(self, traintuple_spec, tag=None):
        spec = ComputePlanTesttupleSpec(
            traintuple_id=traintuple_spec.traintuple_id,
            tag=tag or '',
        )
        self.testtuples.append(spec)
        return spec


class AssetsFactory:

    def __init__(self, name):
        self._data_sample_counter = Counter()
        self._dataset_counter = Counter()
        self._objective_counter = Counter()
        self._algo_counter = Counter()
        self._workdir = pathlib.Path(tempfile.mkdtemp())
        self._uuid = name

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        shutil.rmtree(str(self._workdir), ignore_errors=True)

    def create_data_sample(self, datasets=None, test_only=False):
        rdm = random.random()
        idx = self._data_sample_counter.inc()
        tmpdir = self._workdir / f'data-{idx}'
        tmpdir.mkdir()

        content = '0,{idx}'
        content = f'# random={rdm} \n' + content

        data_filepath = tmpdir / DEFAULT_DATA_SAMPLE_FILENAME
        with open(data_filepath, 'w') as f:
            f.write(content)

        datasets = datasets or []

        return DataSampleSpec(
            path=str(tmpdir),
            test_only=test_only,
            data_manager_keys=[d.key for d in datasets],
        )

    def create_dataset(self, permissions=None):
        rdm = random.random()
        idx = self._dataset_counter.inc()
        tmpdir = self._workdir / f'dataset-{idx}'
        tmpdir.mkdir()
        name = f'{self._uuid} - Dataset {idx}'

        description_path = tmpdir / 'description.md'
        description_content = name
        with open(description_path, 'w') as f:
            f.write(description_content)

        opener_path = tmpdir / 'opener.py'
        opener_content = f'# random={rdm} \n' + DEFAULT_OPENER_SCRIPT
        with open(opener_path, 'w') as f:
            f.write(opener_content)

        return DatasetSpec(
            name=name,
            data_opener=str(opener_path),
            type='Test',
            description=str(description_path),
            permissions=permissions or DEFAULT_PERMISSIONS,
        )

    def create_objective(self, dataset=None, data_samples=None, permissions=None):
        rdm = random.random()
        idx = self._objective_counter.inc()
        tmpdir = self._workdir / f'objective-{idx}'
        tmpdir.mkdir()
        name = f'{self._uuid} - Objective {idx}'

        description_path = tmpdir / 'description.md'
        description_content = f'# random={rdm} {name}'
        with open(description_path, 'w') as f:
            f.write(description_content)

        metrics_content = f'# random={rdm} \n' + DEFAULT_METRICS_SCRIPT

        metrics_zip = utils.create_archive(
            tmpdir / 'metrics',
            ('metrics.py', metrics_content),
            ('Dockerfile', DEFAULT_METRICS_DOCKERFILE),
        )

        data_samples = data_samples or []

        return ObjectiveSpec(
            name=name,
            description=str(description_path),
            metrics_name='test metrics',
            metrics=str(metrics_zip),
            permissions=permissions or DEFAULT_PERMISSIONS,
            test_data_sample_keys=_get_keys(data_samples),
            test_data_manager_key=dataset.key if dataset else None,
        )

    def _create_algo(self, py_script, permissions=None):
        rdm = random.random()
        idx = self._algo_counter.inc()
        tmpdir = self._workdir / f'algo-{idx}'
        tmpdir.mkdir()
        name = f'{self._uuid} - Algo {idx}'

        description_path = tmpdir / 'description.md'
        description_content = name
        with open(description_path, 'w') as f:
            f.write(description_content)

        algo_content = f'# random={rdm} \n' + py_script

        algo_zip = utils.create_archive(
            tmpdir / 'algo',
            ('algo.py', algo_content),
            ('Dockerfile', DEFAULT_ALGO_DOCKERFILE),
        )

        return AlgoSpec(
            name=name,
            description=str(description_path),
            file=str(algo_zip),
            permissions=permissions or DEFAULT_PERMISSIONS,
        )

    def create_algo(self, py_script=None, permissions=None):
        return self._create_algo(
            py_script or DEFAULT_ALGO_SCRIPT,
            permissions=permissions,
        )

    def create_aggregate_algo(self, py_script=None, permissions=None):
        return self._create_algo(
            py_script or DEFAULT_AGGREGATE_ALGO_SCRIPT,
            permissions=permissions,
        )

    def create_composite_algo(self, py_script=None, permissions=None):
        return self._create_algo(
            py_script or DEFAULT_COMPOSITE_ALGO_SCRIPT,
            permissions=permissions,
        )

    def create_traintuple(self, algo=None, objective=None, dataset=None,
                          data_samples=None, traintuples=None, tag=None,
                          compute_plan_id=None, rank=None):
        data_samples = data_samples or []
        traintuples = traintuples or []

        for t in traintuples:
            assert isinstance(t, assets.Traintuple)

        return TraintupleSpec(
            algo_key=algo.key if algo else None,
            objective_key=objective.key if objective else None,
            data_manager_key=dataset.key if dataset else None,
            train_data_sample_keys=_get_keys(data_samples),
            in_models_keys=[t.key for t in traintuples],
            tag=tag,
            compute_plan_id=compute_plan_id,
            rank=rank,
        )

    def create_aggregatetuple(self, algo=None, objective=None, worker=None,
                              traintuples=None, tag=None, compute_plan_id=None,
                              rank=None):
        traintuples = traintuples or []

        for t in traintuples:
            assert isinstance(t, (assets.Traintuple, assets.CompositeTraintuple))

        return AggregatetupleSpec(
            algo_key=algo.key if algo else None,
            objective_key=objective.key if objective else None,
            worker=worker,
            in_models_keys=[t.key for t in traintuples],
            tag=tag,
            compute_plan_id=compute_plan_id,
            rank=rank,
        )

    def create_composite_traintuple(self, algo=None, objective=None, dataset=None,
                                    data_samples=None, head_traintuple=None,
                                    trunk_traintuple=None, tag=None,
                                    compute_plan_id=None, rank=None,
                                    permissions=None):
        data_samples = data_samples or []

        kwargs = {}

        if head_traintuple and trunk_traintuple:
            assert isinstance(head_traintuple, assets.CompositeTraintuple)
            assert isinstance(trunk_traintuple, assets.CompositeTraintuple)
            in_head_model_key = head_traintuple.key
            in_trunk_model_key = trunk_traintuple.key
        else:
            in_head_model_key = None
            in_trunk_model_key = None

        return CompositeTraintupleSpec(
            algo_key=algo.key if algo else None,
            objective_key=objective.key if objective else None,
            data_manager_key=dataset.key if dataset else None,
            train_data_sample_keys=_get_keys(data_samples),
            in_head_model_key=in_head_model_key,
            in_trunk_model_key=in_trunk_model_key,
            tag=tag,
            compute_plan_id=compute_plan_id,
            rank=rank,
            out_trunk_model_permissions=permissions or DEFAULT_PERMISSIONS,
            **kwargs,
        )

    def create_testtuple(self, traintuple=None, tag=None):
        return TesttupleSpec(
            traintuple_key=traintuple.key if traintuple else None,
            tag=tag,
        )

    def create_compute_plan(self, algo=None, objective=None):
        return ComputePlanSpec(
            algo_key=algo.key if algo else None,
            objective_key=objective.key if objective else None,
            traintuples=[],
            testtuples=[],
        )
