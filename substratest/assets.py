import abc
import enum
import dataclasses
import re
import time
import typing

from . import errors


class Future:
    """Future asset."""
    # mapper from asset class name to client getter method
    _methods = {
        'Traintuple': 'get_traintuple',
        'Testtuple': 'get_testtuple',
    }

    def __init__(self, asset, session):
        self._asset = asset
        try:
            m = self._methods[asset.__class__.__name__]
        except KeyError:
            assert False, 'Future not supported'
        self._getter = getattr(session, m)

    def wait(self, timeout=120):
        """Wait until completed (done or failed)."""
        tstart = time.time()
        key = self._asset.key
        return_statuses = ['done', 'failed']
        while self._asset.status not in return_statuses:
            if time.time() - tstart > timeout:
                raise errors.TError(f'Future timeout on {self._asset}')

            time.sleep(3)
            self._asset = self._getter(key)
        return self.get()

    def get(self):
        """Get asset."""
        return self._asset


class _FutureMixin(abc.ABC):
    def attach(self, session):
        """Attach session to asset."""
        self._session = session
        return self

    def future(self):
        """Returns future from asset."""
        assert hasattr(self, 'status')
        assert hasattr(self, 'key')
        return Future(self, self._session)


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
class Algo(_Asset):
    key: str
    name: str
    owner: str
    permissions: Permissions


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
class ComputePlan(_Asset):
    compute_plan_id: str
    traintuple_keys: typing.List[str]
    testtuple_keys: typing.List[str]


@dataclasses.dataclass(frozen=True)
class Node(_Asset):
    id: str
    is_current: bool


class AssetType(enum.Enum):
    algo = enum.auto()
    data_sample = enum.auto()
    dataset = enum.auto()
    objective = enum.auto()
    node = enum.auto()
    testtuple = enum.auto()
    traintuple = enum.auto()

    @classmethod
    def all(cls):
        return [e for e in cls]

    @classmethod
    def can_be_listed(cls):
        return cls.all()
