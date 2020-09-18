import abc
import enum
import time
import typing

from substra.sdk.models import Status
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
        while self._asset.status not in [Status.done, Status.failed, Status.canceled]:
            if time.time() - tstart > timeout:
                raise errors.FutureTimeoutError(f'Future timeout on {self._asset}')

            time.sleep(cfg.FUTURE_POLLING_PERIOD)
            self._asset = self._getter(self._key)

        if raises and self._asset.status == Status.failed:
            raise errors.FutureFailureError(f'Future execution failed on {self._asset}')

        if raises and self._asset.status == Status.canceled:
            raise errors.FutureFailureError(f'Future execution canceled on {self._asset}')

        return self.get()

    def get(self):
        """Get asset."""
        return self._asset


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
