import abc
import enum
import dataclasses
import re
import time
import typing

from . import errors


FUTURE_TIMEOUT = 120  # seconds


class BaseFuture(abc.ABC):
    @abc.abstractmethod
    def wait(self, timeout=FUTURE_TIMEOUT, raises=True):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self):
        raise NotImplementedError


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

    def wait(self, timeout=FUTURE_TIMEOUT, raises=True):
        """Wait until completed (done or failed)."""
        tstart = time.time()
        key = self._asset.key
        completed_statuses = ['done', 'failed']
        while self._asset.status not in completed_statuses:
            if time.time() - tstart > timeout:
                raise errors.FutureTimeoutError(f'Future timeout on {self._asset}')

            time.sleep(3)
            self._asset = self._getter(key)

        if raises and self._asset.status == 'failed':
            raise errors.FutureFailureError(f'Future execution failed on {self._asset}')
        return self.get()

    def get(self):
        """Get asset."""
        return self._asset


class ComputePlanFuture(BaseFuture):
    def __init__(self, compute_plan, session):
        self._compute_plan = compute_plan
        self._session = session

    def wait(self, timeout=FUTURE_TIMEOUT):
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
        self.__session = session
        return self

    @property
    def _session(self):
        if not self.__session:
            raise errors.TError(f'No session attached with {self}')
        return self.__session

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
            # handle nested dataclasses;
            # FIXME does not work for list of nested dataclasses
            attr_type = cls.__annotations__[attr_name]
            if dataclasses.is_dataclass(attr_type) and isinstance(v, dict):
                v = attr_type.load(v)
            kwargs[attr_name] = v

        try:
            return cls(**kwargs)
        except TypeError as e:
            raise errors.TError(f"cannot parse asset `{d}`") from e


class _Asset(_DataclassLoader, abc.ABC):
    """Represents assets stored in the Substra framework.

    Convert a dict with camel case fields to a Dataclass.
    """


@dataclasses.dataclass(frozen=True)
class Permission(_DataclassLoader):
    public: bool
    authorized_ids: typing.List[str]

    class Meta:
        mapper = {
            'authorizedIDs': 'authorized_ids',
        }


@dataclasses.dataclass(frozen=True)
class Permissions(_DataclassLoader):
    """Permissions structure stored in various asset types."""
    process: Permission


@dataclasses.dataclass(frozen=True)
class DataSampleCreated(_Asset):
    key: str
    validated: bool
    path: str

    class Meta:
        mapper = {
            'pkhash': 'key',
        }


@dataclasses.dataclass(frozen=True)
class DataSample(_Asset):
    key: str
    owner: str
    data_manager_keys: typing.List[str]


@dataclasses.dataclass(frozen=True)
class ObjectiveDataset(_DataclassLoader):
    dataset_key: str
    data_sample_keys: typing.List[str]

    class Meta:
        mapper = {
            'dataManagerKey': 'dataset_key',
        }


@dataclasses.dataclass(frozen=True)
class Dataset(_Asset):
    key: str
    name: str
    owner: str
    objective_key: str
    permissions: Permissions
    train_data_sample_keys: typing.List[str] = None
    test_data_sample_keys: typing.List[str] = None


@dataclasses.dataclass(frozen=True)
class _Algo(_Asset):
    key: str
    name: str
    owner: str
    permissions: Permissions


@dataclasses.dataclass(frozen=True)
class Algo(_Algo):
    pass


@dataclasses.dataclass(frozen=True)
class AggregateAlgo(_Algo):
    pass


@dataclasses.dataclass(frozen=True)
class CompositeAlgo(_Algo):
    pass


@dataclasses.dataclass(frozen=True)
class Objective(_Asset):
    key: str
    name: str
    owner: str
    permissions: Permissions
    test_dataset: ObjectiveDataset


@dataclasses.dataclass(frozen=True)
class TupleDataset(_DataclassLoader):
    key: str
    perf: float
    keys: typing.List[str]
    worker: str

    class Meta:
        mapper = {
            'openerHash': 'key',
        }


@dataclasses.dataclass(frozen=True)
class InModel(_DataclassLoader):
    key: str
    storage_address: str

    class Meta:
        mapper = {
            'hash': 'key',
        }


@dataclasses.dataclass(frozen=True)
class OutModel(_DataclassLoader):
    key: str
    storage_address: str

    class Meta:
        mapper = {
            'hash': 'key',
        }


@dataclasses.dataclass
class Traintuple(_Asset, _FutureMixin):
    key: str
    creator: str
    status: str
    dataset: TupleDataset
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


@dataclasses.dataclass
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
    out_model: OutModel = None

    class Meta:
        mapper = {
            'pkhash': 'key',
        }


@dataclasses.dataclass(frozen=True)
class OutCompositeModel(_DataclassLoader):
    permissions: Permissions
    out_model: OutModel = None


@dataclasses.dataclass
class CompositeTraintuple(_Asset, _FutureMixin):
    key: str
    creator: str
    status: str
    dataset: TupleDataset
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


@dataclasses.dataclass
class Testtuple(_Asset, _FutureMixin):
    key: str
    creator: str
    status: str
    dataset: TupleDataset
    certified: bool
    tag: str
    log: str

    class Meta:
        mapper = {
            'pkhash': 'key',
        }


@dataclasses.dataclass
class ComputePlan(_Asset, _ComputePlanFutureMixin):
    compute_plan_id: str
    objective_key: str
    traintuple_keys: typing.List[str]
    composite_traintuple_keys: typing.List[str]
    aggregatetuple_keys: typing.List[str]
    testtuple_keys: typing.List[str]

    def __post_init__(self):
        if self.composite_traintuple_keys is None:
            self.composite_traintuple_keys = []

        if self.aggregatetuple_keys is None:
            self.aggregatetuple_keys = []

        if self.testtuple_keys is None:
            self.testtuple_keys = []

    def list_traintuple(self):
        filters = [
            f'traintuple:computePlanID:{self.compute_plan_id}',
        ]
        tuples = self._session.list_traintuple(filters=filters)
        assert len(tuples) == len(self.traintuple_keys)
        return tuples

    def list_composite_traintuple(self):
        filters = [
            f'composite_traintuple:computePlanID:{self.compute_plan_id}',
        ]
        tuples = self._session.list_composite_traintuple(filters=filters)
        assert len(tuples) == len(self.composite_traintuple_keys)
        return tuples

    def list_aggregatetuple(self):
        filters = [
            f'aggregatetuple:computePlanID:{self.compute_plan_id}',
        ]
        tuples = self._session.list_aggregatetuple(filters=filters)
        assert len(tuples) == len(self.aggregatetuple_keys)
        return tuples

    def list_testtuple(self):
        if not self.testtuple_keys:
            return []
        filters = [f'testtuple:key:{k}' for k in self.testtuple_keys]
        tuples = self._session.list_testtuple(filters=filters)
        assert len(tuples) == len(self.testtuple_keys)
        return tuples


@dataclasses.dataclass(frozen=True)
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
    def can_be_listed(cls):
        return cls.all()
