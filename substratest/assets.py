import abc
import enum
import time
import typing

import pydantic

from substra.sdk import models
from . import errors, cfg


class BaseFuture(abc.ABC):
    @abc.abstractmethod
    def wait(self, timeout=cfg.FUTURE_TIMEOUT, raises=True):
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
        'ComputePlan': 'get_compute_plan'
    }

    def __init__(self, asset, client):
        self._asset = asset
        try:
            m = self._methods[asset.__class__.__name__]
        except KeyError:
            assert False, 'Future not supported'
        self._getter = getattr(client, m)
        if asset.__class__.__name__ == "ComputePlan":
            self._key = asset.compute_plan_id
        else:
            self._key = asset.key

    def wait(self, timeout=cfg.FUTURE_TIMEOUT, raises=True):
        """Wait until completed (done or failed)."""
        tstart = time.time()
        while self._asset.status not in [models.Status.done, models.Status.failed, models.Status.canceled]:
            if time.time() - tstart > timeout:
                raise errors.FutureTimeoutError(f'Future timeout on {self._asset}')

            time.sleep(cfg.FUTURE_POLLING_PERIOD)
            self._asset = self._getter(self._key)

        if raises and self._asset.status == models.Status.failed:
            raise errors.FutureFailureError(f'Future execution failed on {self._asset}')

        if raises and self._asset.status == models.Status.canceled:
            raise errors.FutureFailureError(f'Future execution canceled on {self._asset}')

        return self.get()

    def get(self):
        """Get asset."""
        return self._asset


class ComputePlan(pydantic.BaseModel):
    compute_plan_id: str
    status: str
    traintuple_keys: typing.List[str]
    composite_traintuple_keys: typing.List[str]
    aggregatetuple_keys: typing.List[str]
    testtuple_keys: typing.List[str]
    id_to_key: typing.Dict[str, str]
    tag: str
    metadata: typing.Dict[str, str]

    def __init__(self, *args, **kwargs):
        kwargs['traintuple_keys'] = kwargs.get('traintuple_keys') or []
        kwargs['composite_traintuple_keys'] = kwargs.get('composite_traintuple_keys') or []
        kwargs['aggregatetuple_keys'] = kwargs.get('aggregatetuple_keys') or []
        kwargs['testtuple_keys'] = kwargs.get('testtuple_keys') or []
        super().__init__(*args, **kwargs)

    def list_traintuple(self):
        filters = [
            f'traintuple:compute_plan_id:{self.compute_plan_id}',
        ]
        tuples = self._client.list_traintuple(filters=filters)
        assert len(tuples) == len(self.traintuple_keys)
        assert set(self.traintuple_keys) == set([t.key for t in tuples])
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_composite_traintuple(self):
        filters = [
            f'composite_traintuple:compute_plan_id:{self.compute_plan_id}',
        ]
        tuples = self._client.list_composite_traintuple(filters=filters)
        assert len(tuples) == len(self.composite_traintuple_keys)
        assert set(self.composite_traintuple_keys) == set([t.key for t in tuples])
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_aggregatetuple(self):
        filters = [
            f'aggregatetuple:compute_plan_id:{self.compute_plan_id}',
        ]
        tuples = self._client.list_aggregatetuple(filters=filters)
        assert len(tuples) == len(self.aggregatetuple_keys)
        assert set(self.aggregatetuple_keys) == set([t.key for t in tuples])
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_testtuple(self):
        filters = [
            f'testtuple:compute_plan_id:{self.compute_plan_id}',
        ]
        tuples = self._client.list_testtuple(filters=filters)
        assert len(tuples) == len(self.testtuple_keys)
        assert set(self.testtuple_keys) == set([t.key for t in tuples])
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples


class Node(pydantic.BaseModel):
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
