import abc
import enum
import re
import time
import typing
import pydantic

from inspect import isclass
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
            # handle nested dataclasses;
            # FIXME does not work for list of nested dataclasses
            attr_type = cls.__annotations__[attr_name]
            # because typing.List doesn't work the same way as the other types, we have to check
            # if attr_type is a class before using issubclass()
            if isclass(attr_type) and issubclass(attr_type, _DataclassLoader) \
                    and isinstance(v, dict):
                v = attr_type.load(v)
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


class _BaseFutureAsset(_Asset):
    __session: typing.Any = None
    _future_cls = None

    def attach(self, session):
        """Attach session to asset."""
        object.__setattr__(self, '__session', session)
        return self

    @property
    def _session(self):
        # because self.__session doesn't work properly with Pydantic, we have to use
        # __getattribute__ and __setattr__ (https://docs.python.org/3/reference/datamodel.html)
        if not object.__getattribute__(self, '__session'):
            raise errors.TError(f'No session attached with {self}')
        return object.__getattribute__(self, '__session')

    def future(self):
        """Returns future from asset."""
        return self._future_cls(self, object.__getattribute__(self, '__session'))


class _FutureAsset(_BaseFutureAsset):
    """Represents a single task that is executed on the platform."""
    _future_cls = Future

    def future(self):
        assert hasattr(self, 'status')
        assert hasattr(self, 'key')
        return super().future()


class _ComputePlanFutureAsset(_BaseFutureAsset):
    _future_cls = ComputePlanFuture


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
    dataset_key: str = None
    data_sample_keys: typing.List[str] = None

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
    test_dataset: ObjectiveDataset


class TesttupleDataset(_InternalStruct):
    key: str = None
    perf: float
    keys: typing.List[str]
    worker: str

    class Meta:
        mapper = {
            'openerHash': 'key',
        }


class TraintupleDataset(_InternalStruct):
    key: str = None
    keys: typing.List[str]
    worker: str

    class Meta:
        mapper = {
            'openerHash': 'key',
        }


class InModel(_InternalStruct):
    key: str = None
    storage_address: str = None

    class Meta:
        mapper = {
            'hash': 'key',
        }


class OutModel(_InternalStruct):
    key: str = None
    storage_address: str = None

    class Meta:
        mapper = {
            'hash': 'key',
        }


class Traintuple(_FutureAsset):
    key: str = None
    creator: str
    status: str
    dataset: TraintupleDataset
    permissions: Permissions
    compute_plan_id: str
    rank: int
    tag: str
    log: str
    in_models: typing.List[InModel] = None
    out_model: OutModel = None

    class Meta:
        mapper = {
            'pkhash': 'key',
        }


class Aggregatetuple(_FutureAsset):
    key: str = None
    creator: str
    status: str
    worker: str
    permissions: Permissions
    compute_plan_id: str
    rank: int
    tag: str
    log: str
    in_models: typing.List[InModel]
    out_model: OutModel = None

    class Meta:
        mapper = {
            'pkhash': 'key',
        }


class OutCompositeModel(_InternalStruct):
    permissions: Permissions
    out_model: OutModel = None


class CompositeTraintuple(_FutureAsset):
    key: str
    creator: str
    status: str
    dataset: TraintupleDataset
    compute_plan_id: str
    rank: int
    tag: str
    log: str
    in_head_model: InModel = None
    in_trunk_model: InModel = None
    out_head_model: OutCompositeModel = None
    out_trunk_model: OutCompositeModel = None

    class Meta:
        mapper = {
            'pkhash': 'key',
        }


class Testtuple(_FutureAsset):
    key: str = None
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


class ComputePlan(_ComputePlanFutureAsset):
    compute_plan_id: str = None
    status: str
    traintuple_keys: typing.List[str] = None
    composite_traintuple_keys: typing.List[str] = None
    aggregatetuple_keys: typing.List[str] = None
    testtuple_keys: typing.List[str] = None
    tag: str

    def __post_init__(self):
        if self.traintuple_keys is None:
            self.traintuple_keys = []

        if self.composite_traintuple_keys is None:
            self.composite_traintuple_keys = []

        if self.aggregatetuple_keys is None:
            self.aggregatetuple_keys = []

        if self.testtuple_keys is None:
            self.testtuple_keys = []

    def list_traintuple(self):
        self.__post_init__()
        filters = [
            f'traintuple:computePlanID:{self.compute_plan_id}',
        ]
        tuples = self._session.list_traintuple(filters=filters)
        assert len(tuples) == len(self.traintuple_keys)
        assert set(self.traintuple_keys) == set([t.key for t in tuples])
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_composite_traintuple(self):
        self.__post_init__()
        filters = [
            f'composite_traintuple:computePlanID:{self.compute_plan_id}',
        ]
        tuples = self._session.list_composite_traintuple(filters=filters)
        assert len(tuples) == len(self.composite_traintuple_keys)
        assert set(self.composite_traintuple_keys) == set([t.key for t in tuples])
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_aggregatetuple(self):
        self.__post_init__()
        filters = [
            f'aggregatetuple:computePlanID:{self.compute_plan_id}',
        ]
        tuples = self._session.list_aggregatetuple(filters=filters)
        assert len(tuples) == len(self.aggregatetuple_keys)
        assert set(self.aggregatetuple_keys) == set([t.key for t in tuples])
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_testtuple(self):
        self.__post_init__()
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
