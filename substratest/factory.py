import abc
import os
import pathlib
import random
import shutil
import tempfile
import typing
import uuid

import pydantic

from . import utils, assets


DEFAULT_DATA_SAMPLE_FILENAME = 'data.csv'

DEFAULT_SUBSTRATOOLS_VERSION = '0.5.0'

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
        return [0, 1]
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

# TODO we should have a different serializer for head and trunk models

DEFAULT_COMPOSITE_ALGO_SCRIPT = f"""
import json
import substratools as tools
class TestCompositeAlgo(tools.CompositeAlgo):
    def train(self, X, y, head_model, trunk_model, rank):
        return [0, 1], [0, 2]
    def predict(self, X, head_model, trunk_model):
        return [0, 99]
    def load_head_model(self, path):
        return self._load_model(path)
    def save_head_model(self, model, path):
        return self._save_model(model, path)
    def load_trunk_model(self, path):
        return self._load_model(path)
    def save_trunk_model(self, model, path):
        return self._save_model(model, path)
    def _load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def _save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)
if __name__ == '__main__':
    tools.algo.execute(TestCompositeAlgo())
"""

INVALID_ALGO_SCRIPT = DEFAULT_ALGO_SCRIPT.replace('train', 'naitr')
INVALID_COMPOSITE_ALGO_SCRIPT = DEFAULT_COMPOSITE_ALGO_SCRIPT.replace('train', 'naitr')
INVALID_AGGREGATE_ALGO_SCRIPT = DEFAULT_AGGREGATE_ALGO_SCRIPT.replace('aggregate', 'etagergga')

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


def _shorten_name(name):
    """Format asset name to ensure they match the backend requirements."""
    if len(name) < 100:
        return name
    return name[:75] + '...' + name[:20]


class Counter:
    def __init__(self):
        self._idx = -1

    def inc(self):
        self._idx += 1
        return self._idx


class _Spec(pydantic.BaseModel, abc.ABC):
    """Asset specification base class."""
    pass


class Permissions(pydantic.BaseModel):
    public: bool
    authorized_ids: typing.List[str]


class PrivatePermissions(pydantic.BaseModel):
    authorized_ids: typing.List[str]


DEFAULT_PERMISSIONS = Permissions(public=True, authorized_ids=[])
DEFAULT_OUT_TRUNK_MODEL_PERMISSIONS = PrivatePermissions(authorized_ids=[])


class DataSampleSpec(_Spec):
    path: str
    test_only: bool
    data_manager_keys: typing.List[str]

    def move_data_to(self, destination):
        destination = destination if destination.endswith('/') else destination + '/'
        destination = tempfile.mkdtemp(dir=destination)
        shutil.move(self.path, destination)
        self.path = os.path.join(destination, os.path.basename(self.path))


class DatasetSpec(_Spec):
    name: str
    data_opener: str
    type: str
    description: str
    permissions: Permissions = None
    objective_key: str = None

    def read_opener(self):
        with open(self.data_opener, 'rb') as f:
            return f.read()

    def read_description(self):
        with open(self.description, 'r') as f:
            return f.read()


class ObjectiveSpec(_Spec):
    name: str
    description: str
    metrics_name: str
    metrics: str
    test_data_sample_keys: typing.List[str]
    test_data_manager_key: str = None
    permissions: Permissions = None


class _AlgoSpec(_Spec):
    name: str
    description: str
    file: str
    permissions: Permissions = None


class AlgoSpec(_AlgoSpec):
    pass


class AggregateAlgoSpec(_AlgoSpec):
    pass


class CompositeAlgoSpec(_AlgoSpec):
    pass


class TraintupleSpec(_Spec):
    algo_key: str
    data_manager_key: str
    train_data_sample_keys: typing.List[str]
    in_models_keys: typing.List[str] = None
    tag: str = None
    compute_plan_id: str = None
    rank: int = None


class AggregatetupleSpec(_Spec):
    algo_key: str
    worker: str
    in_models_keys: typing.List[str]
    tag: str = None
    compute_plan_id: str = None
    rank: int = None


class CompositeTraintupleSpec(_Spec):
    algo_key: str
    data_manager_key: str
    train_data_sample_keys: typing.List[str]
    in_head_model_key: str = None
    in_trunk_model_key: str = None
    tag: str = None
    compute_plan_id: str = None
    out_trunk_model_permissions: PrivatePermissions
    rank: int = None


class TesttupleSpec(_Spec):
    objective_key: str
    traintuple_key: str
    tag: str = None
    data_manager_key: str = None
    test_data_sample_keys: typing.List[str] = None


class ComputePlanTraintupleSpec(_Spec):
    algo_key: str
    data_manager_key: str
    train_data_sample_keys: typing.List[str]
    traintuple_id: str
    in_models_ids: typing.List[str] = None
    tag: str = None

    @property
    def id(self):
        return self.traintuple_id


class ComputePlanAggregatetupleSpec(_Spec):
    aggregatetuple_id: str
    algo_key: str
    worker: str
    in_models_ids: typing.List[str] = None
    tag: str = None

    @property
    def id(self):
        return self.aggregatetuple_id


class ComputePlanCompositeTraintupleSpec(_Spec):
    composite_traintuple_id: str
    algo_key: str
    data_manager_key: str
    train_data_sample_keys: typing.List[str]
    in_head_model_id: str = None
    in_trunk_model_id: str = None
    tag: str = None
    out_trunk_model_permissions: Permissions

    @property
    def id(self):
        return self.composite_traintuple_id


class ComputePlanTesttupleSpec(_Spec):
    objective_key: str
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


class _BaseComputePlanSpec(_Spec, abc.ABC):
    traintuples: typing.List[ComputePlanTraintupleSpec]
    composite_traintuples: typing.List[ComputePlanCompositeTraintupleSpec]
    aggregatetuples: typing.List[ComputePlanAggregatetupleSpec]
    testtuples: typing.List[ComputePlanTesttupleSpec]

    def add_traintuple(self, algo, dataset, data_samples, in_models=None, tag=''):
        in_models = in_models or []
        spec = ComputePlanTraintupleSpec(
            algo_key=algo.key,
            traintuple_id=random_uuid(),
            data_manager_key=dataset.key,
            train_data_sample_keys=_get_keys(data_samples),
            in_models_ids=[t.id for t in in_models],
            tag=tag,
        )
        self.traintuples.append(spec)
        return spec

    def add_aggregatetuple(self, aggregate_algo, worker, in_models=None, tag=''):
        in_models = in_models or []

        for t in in_models:
            assert isinstance(t, (ComputePlanTraintupleSpec, ComputePlanCompositeTraintupleSpec))

        spec = ComputePlanAggregatetupleSpec(
            aggregatetuple_id=random_uuid(),
            algo_key=aggregate_algo.key,
            worker=worker,
            in_models_ids=[t.id for t in in_models],
            tag=tag,
        )
        self.aggregatetuples.append(spec)
        return spec

    def add_composite_traintuple(self, composite_algo, dataset=None, data_samples=None,
                                 in_head_model=None, in_trunk_model=None,
                                 out_trunk_model_permissions=None, tag=''):
        data_samples = data_samples or []

        if in_head_model and in_trunk_model:
            assert isinstance(in_head_model, ComputePlanCompositeTraintupleSpec)
            assert isinstance(
                in_trunk_model,
                (ComputePlanCompositeTraintupleSpec, ComputePlanAggregatetupleSpec)
            )

        spec = ComputePlanCompositeTraintupleSpec(
            composite_traintuple_id=random_uuid(),
            algo_key=composite_algo.key,
            data_manager_key=dataset.key if dataset else None,
            train_data_sample_keys=_get_keys(data_samples),
            in_head_model_id=in_head_model.id if in_head_model else None,
            in_trunk_model_id=in_trunk_model.id if in_trunk_model else None,
            out_trunk_model_permissions=out_trunk_model_permissions or DEFAULT_OUT_TRUNK_MODEL_PERMISSIONS,
            tag=tag,
        )
        self.composite_traintuples.append(spec)
        return spec

    def add_testtuple(self, objective, traintuple_spec, tag=None):
        spec = ComputePlanTesttupleSpec(
            objective_key=objective.key,
            traintuple_id=traintuple_spec.id,
            tag=tag or '',
        )
        self.testtuples.append(spec)
        return spec


class ComputePlanSpec(_BaseComputePlanSpec):
    tag: str
    clean_models: bool


class UpdateComputePlanSpec(_BaseComputePlanSpec):
    compute_plan_id: str


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

    def create_data_sample(self, content=None, datasets=None, test_only=False):
        rdm = random.random()
        idx = self._data_sample_counter.inc()
        tmpdir = self._workdir / f'data-{idx}'
        tmpdir.mkdir()

        encoding = 'utf-8'
        content = content or f'0,{idx}'.encode(encoding)
        content = f'# random={rdm} \n'.encode(encoding) + content

        data_filepath = tmpdir / DEFAULT_DATA_SAMPLE_FILENAME
        with open(data_filepath, 'wb') as f:
            f.write(content)

        datasets = datasets or []

        return DataSampleSpec(
            path=str(tmpdir),
            test_only=test_only,
            data_manager_keys=[d.key for d in datasets],
        )

    def create_dataset(self, objective=None, permissions=None):
        rdm = random.random()
        idx = self._dataset_counter.inc()
        tmpdir = self._workdir / f'dataset-{idx}'
        tmpdir.mkdir()
        name = _shorten_name(f'{self._uuid} - Dataset {idx}')

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
            objective_key=objective.key if objective else None,
            permissions=permissions or DEFAULT_PERMISSIONS,
        )

    def create_objective(self, dataset=None, data_samples=None, permissions=None):
        rdm = random.random()
        idx = self._objective_counter.inc()
        tmpdir = self._workdir / f'objective-{idx}'
        tmpdir.mkdir()
        name = _shorten_name(f'{self._uuid} - Objective {idx}')

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
        name = _shorten_name(f'{self._uuid} - Algo {idx}')

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

    def create_traintuple(self, algo=None, dataset=None,
                          data_samples=None, traintuples=None, tag=None,
                          compute_plan_id=None, rank=None):
        data_samples = data_samples or []
        traintuples = traintuples or []

        for t in traintuples:
            assert isinstance(t, assets.Traintuple)

        return TraintupleSpec(
            algo_key=algo.key if algo else None,
            data_manager_key=dataset.key if dataset else None,
            train_data_sample_keys=_get_keys(data_samples),
            in_models_keys=[t.key for t in traintuples],
            tag=tag,
            compute_plan_id=compute_plan_id,
            rank=rank,
        )

    def create_aggregatetuple(self, algo=None, worker=None,
                              traintuples=None, tag=None, compute_plan_id=None,
                              rank=None):
        traintuples = traintuples or []

        for t in traintuples:
            assert isinstance(t, (assets.Traintuple, assets.CompositeTraintuple))

        return AggregatetupleSpec(
            algo_key=algo.key if algo else None,
            worker=worker,
            in_models_keys=[t.key for t in traintuples],
            tag=tag,
            compute_plan_id=compute_plan_id,
            rank=rank,
        )

    def create_composite_traintuple(self, algo=None, dataset=None,
                                    data_samples=None, head_traintuple=None,
                                    trunk_traintuple=None, tag=None,
                                    compute_plan_id=None, rank=None,
                                    permissions=None):
        data_samples = data_samples or []

        if head_traintuple and trunk_traintuple:
            assert isinstance(head_traintuple, assets.CompositeTraintuple)
            assert isinstance(
                trunk_traintuple,
                (assets.CompositeTraintuple, assets.Aggregatetuple)
            )
            in_head_model_key = head_traintuple.key
            in_trunk_model_key = trunk_traintuple.key
        else:
            in_head_model_key = None
            in_trunk_model_key = None

        return CompositeTraintupleSpec(
            algo_key=algo.key if algo else None,
            data_manager_key=dataset.key if dataset else None,
            train_data_sample_keys=_get_keys(data_samples),
            in_head_model_key=in_head_model_key,
            in_trunk_model_key=in_trunk_model_key,
            tag=tag,
            compute_plan_id=compute_plan_id,
            rank=rank,
            out_trunk_model_permissions=permissions or DEFAULT_OUT_TRUNK_MODEL_PERMISSIONS,
        )

    def create_testtuple(self, objective=None, traintuple=None, tag=None, dataset=None, data_samples=None):
        return TesttupleSpec(
            objective_key=objective.key if objective else None,
            traintuple_key=traintuple.key if traintuple else None,
            data_manager_key=dataset.key if dataset else None,
            test_data_sample_keys=_get_keys(data_samples),
            tag=tag,
        )

    def create_compute_plan(self, tag='', clean_models=False):
        return ComputePlanSpec(
            traintuples=[],
            composite_traintuples=[],
            aggregatetuples=[],
            testtuples=[],
            tag=tag,
            clean_models=clean_models
        )

    def update_compute_plan(self, compute_plan):
        return UpdateComputePlanSpec(
            traintuples=[],
            composite_traintuples=[],
            aggregatetuples=[],
            testtuples=[],
            compute_plan_id=compute_plan.compute_plan_id,
        )
