import abc
import enum
import re
import time
import typing
from inspect import isclass

import pydantic

from . import errors, cfg


class BaseFuture(abc.ABC):
    @abc.abstractmethod
    def wait(self, timeout=cfg.FUTURE_TIMEOUT, raises=True):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self):
        raise NotImplementedError


class Status:
    doing = 'doing'
    done = 'done'
    failed = 'failed'
    todo = 'todo'
    waiting = 'waiting'
    canceled = 'canceled'


class Future(BaseFuture):
    """Future asset."""
    # mapper from asset class name to client getter method
    _methods = {
        'Traintuple': 'get_traintuple',
        'Testtuple': 'get_testtuple',
        'Aggregatetuple': 'get_aggregatetuple',
        'CompositeTraintuple': 'get_composite_traintuple',
    }

    def __init__(self, asset, session):
        self._asset = asset
        try:
            m = self._methods[asset.__class__.__name__]
        except KeyError:
            assert False, 'Future not supported'
        self._getter = getattr(session, m)

    def wait(self, timeout=cfg.FUTURE_TIMEOUT, raises=True):
        """Wait until completed (done or failed)."""
        tstart = time.time()
        key = self._asset.key
        while self._asset.status not in [Status.done, Status.failed, Status.canceled]:
            if time.time() - tstart > timeout:
                raise errors.FutureTimeoutError(f'Future timeout on {self._asset}')

            time.sleep(cfg.FUTURE_POLLING_PERIOD)
            self._asset = self._getter(key)

        if raises and self._asset.status == Status.failed:
            raise errors.FutureFailureError(f'Future execution failed on {self._asset}')

        if raises and self._asset.status == Status.canceled:
            raise errors.FutureFailureError(f'Future execution canceled on {self._asset}')

        return self.get()

    def get(self):
        """Get asset."""
        return self._asset


class ComputePlanFuture(BaseFuture):
    def __init__(self, compute_plan, session):
        self._compute_plan = compute_plan
        self._session = session

    def wait(self, timeout=cfg.FUTURE_TIMEOUT):
        """wait until all tuples are completed (done or failed)."""
        tuples = (self._compute_plan.list_traintuple() +
                  self._compute_plan.list_composite_traintuple() +
                  self._compute_plan.list_aggregatetuple())
        # order tuples by rank to wait on the tuples that should be executed first
        tuples = sorted(tuples, key=lambda t: t.rank)
        # testtuples do not have a rank attribute
        tuples += self._compute_plan.list_testtuple()

        for tuple_ in tuples:
            tuple_.future().wait(timeout, raises=False)

        return self.get()

    def get(self):
        return self._session.get_compute_plan(self._compute_plan.compute_plan_id)


class _BaseFutureMixin(abc.ABC):
    _future_cls = None

    def attach(self, session):
        """Attach session to asset."""
        # XXX because Pydantic doesn't support private fields, we have to use
        # __getattribute__ and __setattr__ (https://github.com/samuelcolvin/pydantic/issues/655)
        object.__setattr__(self, '__session', session)
        return self

    @property
    def _session(self):
        try:
            return object.__getattribute__(self, '__session')
        except AttributeError:
            raise errors.TError(f'No session attached with {self}')

    def future(self):
        """Returns future from asset."""
        return self._future_cls(self, self._session)


class _FutureMixin(_BaseFutureMixin):
    """Represents a single task that is executed on the platform."""
    _future_cls = Future

    def future(self):
        assert hasattr(self, 'status')
        assert hasattr(self, 'key')
        return super().future()


class _ComputePlanFutureMixin(_BaseFutureMixin):
    _future_cls = ComputePlanFuture


def _convert(name):
    """Convert camel case to snake case."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


class _DataclassLoader(abc.ABC):
    """Base model structure defining assets.

    Provides a load method to create the object from a dictionary.
    Converts automatically camel case fields to snake case fields and provides a mapper
    to define custom fields mapping.
    """

    class Meta:
        mapper = {}

    @classmethod
    def load(cls, d):
        """Create asset from dictionary."""
        if isinstance(d, cls):
            return d

        mapper = cls.Meta.mapper
        kwargs = {}
        for k, v in d.items():
            attr_name = mapper[k] if k in mapper else _convert(k)
            if attr_name not in cls.__annotations__:
                continue
            # handle nested structures;
            attr_type = cls.__annotations__[attr_name]

            # handle optional arguments

            if hasattr(attr_type, '__origin__') and attr_type.__origin__ is typing.Union \
                    and len(attr_type.__args__) == 2 and type(None) in attr_type.__args__:
                attr_type = [arg for arg in attr_type.__args__
                             if arg is not type(None)][0]  # noqa: E721

            # because typing.List doesn't work the same way as the other types, we have to check
            # if attr_type is a class before using issubclass()
            if isclass(attr_type) and issubclass(attr_type, _DataclassLoader) \
                    and isinstance(v, dict):
                v = attr_type.load(v)

            # handle list of internal structures;
            elif hasattr(attr_type, '__origin__') and attr_type.__origin__ is list \
                    and isinstance(v, list):
                list_value_type = attr_type.__args__
                first_value_type = list_value_type[0]
                if issubclass(first_value_type, _DataclassLoader):
                    v = [first_value_type.load(e) for e in v]

            kwargs[attr_name] = v

        try:
            return cls(**kwargs)
        except TypeError as e:
            raise errors.TError(f"cannot parse asset `{d}`") from e


class _InternalStruct(pydantic.BaseModel, _DataclassLoader, abc.ABC):
    """Internal nested structure"""


class _Asset(_InternalStruct, abc.ABC):
    """Represents assets stored in the Substra framework.

    Convert a dict with camel case fields to a Dataclass.
    """


class Permission(_InternalStruct):
    public: bool
    authorized_ids: typing.List[str]

    class Meta:
        mapper = {
            'authorizedIDs': 'authorized_ids',
        }


class Permissions(_InternalStruct):
    """Permissions structure stored in various asset types."""
    process: Permission


class DataSampleCreated(_Asset):
    key: str
    validated: bool
    path: str

    class Meta:
        mapper = {
            'pkhash': 'key',
        }


class DataSample(_Asset):
    key: str
    owner: str
    data_manager_keys: typing.List[str]


class ObjectiveDataset(_InternalStruct):
    dataset_key: str
    data_sample_keys: typing.List[str]

    class Meta:
        mapper = {
            'dataManagerKey': 'dataset_key',
        }


class Dataset(_Asset):
    key: str
    name: str
    owner: str
    objective_key: str
    permissions: Permissions
    # the JSON data returned by list_dataset doesn't include the following keys at all
    # they are only included in the result of get_dataset
    train_data_sample_keys: typing.List[str] = None
    test_data_sample_keys: typing.List[str] = None


class _Algo(_Asset):
    key: str
    name: str
    owner: str
    permissions: Permissions


class Algo(_Algo):
    pass


class AggregateAlgo(_Algo):
    pass


class CompositeAlgo(_Algo):
    pass


class Objective(_Asset):
    key: str
    name: str
    owner: str
    permissions: Permissions
    test_dataset: typing.Optional[ObjectiveDataset]


class TesttupleDataset(_InternalStruct):
    key: str
    perf: float
    keys: typing.List[str]
    worker: str

    class Meta:
        mapper = {
            'openerHash': 'key',
        }


class TraintupleDataset(_InternalStruct):
    key: str
    keys: typing.List[str]
    worker: str

    class Meta:
        mapper = {
            'openerHash': 'key',
        }


class InModel(_InternalStruct):
    key: str
    storage_address: str

    class Meta:
        mapper = {
            'hash': 'key',
        }


class OutModel(_InternalStruct):
    key: str
    storage_address: str

    class Meta:
        mapper = {
            'hash': 'key',
        }


class Traintuple(_Asset, _FutureMixin):
    key: str
    creator: str
    status: str
    dataset: TraintupleDataset
    permissions: Permissions
    compute_plan_id: str
    rank: int
    tag: str
    log: str
    in_models: typing.Optional[typing.List[InModel]]
    out_model: typing.Optional[OutModel]

    class Meta:
        mapper = {
            'pkhash': 'key',
        }


class Aggregatetuple(_Asset, _FutureMixin):
    key: str
    creator: str
    status: str
    worker: str
    permissions: Permissions
    compute_plan_id: str
    rank: int
    tag: str
    log: str
    in_models: typing.List[InModel]
    out_model: typing.Optional[OutModel]

    class Meta:
        mapper = {
            'pkhash': 'key',
        }


class OutCompositeModel(_Asset):
    permissions: Permissions
    out_model: typing.Optional[OutModel]


class CompositeTraintuple(_Asset, _FutureMixin):
    key: str
    creator: str
    status: str
    dataset: TraintupleDataset
    compute_plan_id: str
    rank: int
    tag: str
    log: str
    in_head_model: typing.Optional[InModel]
    in_trunk_model: typing.Optional[InModel]
    out_head_model: OutCompositeModel
    out_trunk_model: OutCompositeModel

    class Meta:
        mapper = {
            'pkhash': 'key',
        }


class Testtuple(_Asset, _FutureMixin):
    key: str
    creator: str
    status: str
    dataset: TesttupleDataset
    certified: bool
    rank: int
    tag: str
    log: str

    class Meta:
        mapper = {
            'pkhash': 'key',
        }


class ComputePlan(_Asset, _ComputePlanFutureMixin):
    compute_plan_id: str
    status: str
    traintuple_keys: typing.List[str]
    composite_traintuple_keys: typing.List[str]
    aggregatetuple_keys: typing.List[str]
    testtuple_keys: typing.List[str]
    tag: str

    def __init__(self, *args, **kwargs):
        kwargs['traintuple_keys'] = kwargs.get('traintuple_keys') or []
        kwargs['composite_traintuple_keys'] = kwargs.get('composite_traintuple_keys') or []
        kwargs['aggregatetuple_keys'] = kwargs.get('aggregatetuple_keys') or []
        kwargs['testtuple_keys'] = kwargs.get('testtuple_keys') or []
        super().__init__(*args, **kwargs)

    def list_traintuple(self):
        filters = [
            f'traintuple:computePlanID:{self.compute_plan_id}',
        ]
        tuples = self._session.list_traintuple(filters=filters)
        assert len(tuples) == len(self.traintuple_keys)
        assert set(self.traintuple_keys) == set([t.key for t in tuples])
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_composite_traintuple(self):
        filters = [
            f'composite_traintuple:computePlanID:{self.compute_plan_id}',
        ]
        tuples = self._session.list_composite_traintuple(filters=filters)
        assert len(tuples) == len(self.composite_traintuple_keys)
        assert set(self.composite_traintuple_keys) == set([t.key for t in tuples])
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_aggregatetuple(self):
        filters = [
            f'aggregatetuple:computePlanID:{self.compute_plan_id}',
        ]
        tuples = self._session.list_aggregatetuple(filters=filters)
        assert len(tuples) == len(self.aggregatetuple_keys)
        assert set(self.aggregatetuple_keys) == set([t.key for t in tuples])
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_testtuple(self):
        filters = [
            f'testtuple:computePlanID:{self.compute_plan_id}',
        ]
        tuples = self._session.list_testtuple(filters=filters)
        assert len(tuples) == len(self.testtuple_keys)
        assert set(self.testtuple_keys) == set([t.key for t in tuples])
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples


class Node(_Asset):
    id: str
    is_current: bool


class AssetType(enum.Enum):
    algo = enum.auto()
    aggregate_algo = enum.auto()
    aggregatetuple = enum.auto()
    composite_algo = enum.auto()
    composite_traintuple = enum.auto()
    data_sample = enum.auto()
    dataset = enum.auto()
    objective = enum.auto()
    node = enum.auto()
    testtuple = enum.auto()
    traintuple = enum.auto()
    compute_plan = enum.auto()

    @classmethod
    def all(cls):
        return [e for e in cls]

    @classmethod
    def can_be_get(cls):
        gettable = cls.all()
        gettable.remove(cls.data_sample)
        gettable.remove(cls.node)
        return gettable

    @classmethod
    def can_be_listed(cls):
        return cls.all()
