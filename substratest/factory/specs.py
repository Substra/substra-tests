"""Specification of assets metadata to add to the platform."""
import abc
import dataclasses
import os
import shutil
import tempfile
import typing
import uuid


def random_uuid():
    return uuid.uuid4().hex


def _get_key(obj, field='key'):
    """Get key from asset/spec or key."""
    if isinstance(obj, str):
        return obj
    return getattr(obj, field)


def get_keys(obj, field='key'):
    """Get keys from asset/spec or key.

    This is particularly useful for data samples to accept as input args a list of keys
    and a list of data samples.
    """
    if not obj:
        return []
    return [_get_key(x, field=field) for x in obj]


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
DEFAULT_OUT_MODEL_PERMISSIONS = Permissions(public=False, authorized_ids=[])


@dataclasses.dataclass
class DataSample(_Spec):
    path: str
    test_only: bool
    data_manager_keys: typing.List[str]

    def move_data_to(self, destination):
        destination = destination if destination.endswith('/') else destination + '/'
        destination = tempfile.mkdtemp(dir=destination)
        shutil.move(self.path, destination)
        self.path = os.path.join(destination, os.path.basename(self.path))


@dataclasses.dataclass
class Dataset(_Spec):
    name: str
    data_opener: str
    type: str
    description: str
    permissions: Permissions = None

    def read_opener(self):
        with open(self.data_opener, 'rb') as f:
            return f.read()


@dataclasses.dataclass
class Objective(_Spec):
    name: str
    description: str
    metrics_name: str
    metrics: str
    test_data_sample_keys: typing.List[str]
    test_data_manager_key: str
    permissions: Permissions = None


@dataclasses.dataclass
class _Algo(_Spec):
    name: str
    description: str
    file: str
    permissions: Permissions = None


@dataclasses.dataclass
class Algo(_Algo):
    pass


@dataclasses.dataclass
class AggregateAlgo(_Algo):
    pass


@dataclasses.dataclass
class CompositeAlgo(_Algo):
    pass


@dataclasses.dataclass
class Traintuple(_Spec):
    algo_key: str
    data_manager_key: str
    train_data_sample_keys: str
    in_models_keys: typing.List[str]
    tag: str
    compute_plan_id: str
    rank: int = None


@dataclasses.dataclass
class Aggregatetuple(_Spec):
    algo_key: str
    worker: str
    in_models_keys: typing.List[str]
    tag: str
    compute_plan_id: str
    rank: int = None


@dataclasses.dataclass
class CompositeTraintuple(_Spec):
    algo_key: str
    data_manager_key: str
    train_data_sample_keys: str
    in_head_model_key: str
    in_trunk_model_key: str
    tag: str
    compute_plan_id: str
    out_trunk_model_permissions: typing.Dict
    rank: int = None


@dataclasses.dataclass
class Testtuple(_Spec):
    objective_key: str
    traintuple_key: str
    tag: str


@dataclasses.dataclass
class ComputePlanTraintuple:
    algo_key: str
    data_manager_key: str
    train_data_sample_keys: str
    traintuple_id: str
    in_models_ids: typing.List[str]
    tag: str

    @property
    def id(self):
        return self.traintuple_id


@dataclasses.dataclass
class ComputePlanAggregatetuple:
    aggregatetuple_id: str
    algo_key: str
    worker: str
    in_models_ids: typing.List[str]
    tag: str

    @property
    def id(self):
        return self.aggregatetuple_id


@dataclasses.dataclass
class ComputePlanCompositeTraintuple:
    composite_traintuple_id: str
    algo_key: str
    data_manager_key: str
    train_data_sample_keys: str
    in_head_model_id: str
    in_trunk_model_id: str
    tag: str
    out_trunk_model_permissions: typing.Dict

    @property
    def id(self):
        return self.composite_traintuple_id


@dataclasses.dataclass
class ComputePlanTesttuple:
    objective_key: str
    traintuple_id: str
    tag: str


@dataclasses.dataclass
class ComputePlan(_Spec):
    traintuples: typing.List[ComputePlanTraintuple]
    composite_traintuples: typing.List[ComputePlanCompositeTraintuple]
    aggregatetuples: typing.List[ComputePlanAggregatetuple]
    testtuples: typing.List[ComputePlanTesttuple]

    def add_traintuple(self, algo, dataset, data_samples, in_models=None, tag=''):
        in_models = in_models or []
        spec = ComputePlanTraintuple(
            algo_key=algo.key,
            traintuple_id=random_uuid(),
            data_manager_key=dataset.key,
            train_data_sample_keys=get_keys(data_samples),
            in_models_ids=[t.id for t in in_models],
            tag=tag,
        )
        self.traintuples.append(spec)
        return spec

    def add_aggregatetuple(self, aggregate_algo, worker, in_models=None, tag=''):
        in_models = in_models or []

        for t in in_models:
            assert isinstance(t, (ComputePlanTraintuple, ComputePlanCompositeTraintuple))

        spec = ComputePlanAggregatetuple(
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
            assert isinstance(in_head_model, ComputePlanCompositeTraintuple)
            assert isinstance(
                in_trunk_model,
                (ComputePlanCompositeTraintuple, ComputePlanAggregatetuple)
            )

        spec = ComputePlanCompositeTraintuple(
            composite_traintuple_id=random_uuid(),
            algo_key=composite_algo.key,
            data_manager_key=dataset.key if dataset else None,
            train_data_sample_keys=get_keys(data_samples),
            in_head_model_id=in_head_model.id if in_head_model else None,
            in_trunk_model_id=in_trunk_model.id if in_trunk_model else None,
            out_trunk_model_permissions=out_trunk_model_permissions or DEFAULT_OUT_MODEL_PERMISSIONS,
            tag=tag,
        )
        self.composite_traintuples.append(spec)
        return spec

    def add_testtuple(self, objective, traintuple_spec, tag=None):
        spec = ComputePlanTesttuple(
            objective_key=objective.key,
            traintuple_id=traintuple_spec.id,
            tag=tag or '',
        )
        self.testtuples.append(spec)
        return spec
