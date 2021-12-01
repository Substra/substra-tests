import os
import tempfile
import time

import substra
from substra.sdk.models import Status, ComputePlanStatus, ModelType

from . import errors, cfg

DATASET_DOWNLOAD_FILENAME = 'opener.py'
ALGO_DOWNLOAD_FILENAME = 'algo.tar.gz'

_get_methods = {
    'Traintuple': 'get_traintuple',
    'Testtuple': 'get_testtuple',
    'Aggregatetuple': 'get_aggregatetuple',
    'CompositeTraintuple': 'get_composite_traintuple',
    'ComputePlan': 'get_compute_plan'
}


class Client:
    """Client to interact with a Node of Substra.

    Parses responses from server to return Asset instances.
    """

    def __init__(self, debug, node_id=None, address=None, token=None, user=None, password=None):
        super().__init__()

        self.node_id = node_id
        self.debug = debug
        self._client = substra.Client(debug=debug, url=address, insecure=False, token=token)
        if not token:
            token = self._client.login(user, password)
        self.token = token

    def add_data_sample(self, spec, *args, **kwargs):
        key = self._client.add_data_sample(spec.dict(), *args, **kwargs)
        return key

    def add_data_samples(self, spec, *args, **kwargs):
        keys = self._client.add_data_samples(spec.dict(), *args, **kwargs)
        return keys

    def add_dataset(self, spec, *args, **kwargs):
        key = self._client.add_dataset(spec.dict(), *args, **kwargs)
        return self._client.get_dataset(key)

    def add_metric(self, spec, *args, **kwargs):
        key = self._client.add_metric(spec.dict(), *args, **kwargs)
        return self._client.get_metric(key)

    def add_algo(self, spec, *args, **kwargs):
        key = self._client.add_algo(spec.dict(), *args, **kwargs)
        return self._client.get_algo(key)

    def add_traintuple(self, spec, *args, **kwargs):
        key = self._client.add_traintuple(spec.dict(), *args, **kwargs)
        return self._client.get_traintuple(key)

    def add_aggregatetuple(self, spec, *args, **kwargs):
        key = self._client.add_aggregatetuple(spec.dict(), *args, **kwargs)
        return self._client.get_aggregatetuple(key)

    def add_composite_traintuple(self, spec, *args, **kwargs):
        key = self._client.add_composite_traintuple(spec.dict(), *args, **kwargs)
        return self._client.get_composite_traintuple(key)

    def add_testtuple(self, spec, *args, **kwargs):
        key = self._client.add_testtuple(spec.dict(), *args, **kwargs)
        return self._client.get_testtuple(key)

    def add_compute_plan(self, spec, *args, **kwargs):
        return self._client.add_compute_plan(spec.dict(), *args, **kwargs)

    def update_compute_plan(self, spec, *args, **kwargs):
        spec_dict = spec.dict()
        # Remove extra field from data
        spec_dict.pop("key")
        return self._client.update_compute_plan(spec.key, spec_dict, *args, **kwargs)

    def list_compute_plan(self, *args, **kwargs):
        return self._client.list_compute_plan(*args, **kwargs)

    def get_compute_plan(self, *args, **kwargs):
        return self._client.get_compute_plan(*args, **kwargs)

    def list_data_sample(self, *args, **kwargs):
        return self._client.list_data_sample(*args, **kwargs)

    def get_algo(self, *args, **kwargs):
        return self._client.get_algo(*args, **kwargs)

    def list_algo(self, *args, **kwargs):
        return self._client.list_algo(*args, **kwargs)

    def get_dataset(self, *args, **kwargs):
        return self._client.get_dataset(*args, **kwargs)

    def list_dataset(self, *args, **kwargs):
        return self._client.list_dataset(*args, **kwargs)

    def get_metric(self, *args, **kwargs):
        return self._client.get_metric(*args, **kwargs)

    def list_metric(self, *args, **kwargs):
        return self._client.list_metric(*args, **kwargs)

    def get_traintuple(self, *args, **kwargs):
        return self._client.get_traintuple(*args, **kwargs)

    def list_traintuple(self, *args, **kwargs):
        return self._client.list_traintuple(*args, **kwargs)

    def get_aggregatetuple(self, *args, **kwargs):
        return self._client.get_aggregatetuple(*args, **kwargs)

    def list_aggregatetuple(self, *args, **kwargs):
        return self._client.list_aggregatetuple(*args, **kwargs)

    def get_composite_traintuple(self, *args, **kwargs):
        return self._client.get_composite_traintuple(*args, **kwargs)

    def list_composite_traintuple(self, *args, **kwargs):
        return self._client.list_composite_traintuple(*args, **kwargs)

    def get_testtuple(self, *args, **kwargs):
        return self._client.get_testtuple(*args, **kwargs)

    def list_testtuple(self, *args, **kwargs):
        return self._client.list_testtuple(*args, **kwargs)

    def list_node(self, *args, **kwargs):
        return self._client.list_node(*args, **kwargs)

    def download_opener(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            self._client.download_dataset(key, tmp)
            path = os.path.join(tmp, DATASET_DOWNLOAD_FILENAME)
            with open(path, 'rb') as f:
                return f.read()

    def download_algo(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            self._client.download_algo(key, tmp)
            path = os.path.join(tmp, ALGO_DOWNLOAD_FILENAME)
            with open(path, 'rb') as f:
                return f.read()

    def download_model(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            self._client.download_model(key, tmp)
            path = os.path.join(tmp, f'model_{key}')
            with open(path, 'rb') as f:
                return f.read()

    def download_trunk_model_from_composite_traintuple(self, composite_traintuple_key):
        with tempfile.TemporaryDirectory() as tmp:
            self._client.download_trunk_model_from_composite_traintuple(composite_traintuple_key, tmp)
            tuple = self.get_composite_traintuple(composite_traintuple_key)
            for model in tuple.composite.models:
                if model.category == ModelType.simple:
                    model_key = model.key
            path = os.path.join(tmp, f'model_{model_key}')
            with open(path, 'rb') as f:
                return f.read()

    def describe_dataset(self, key):
        return self._client.describe_dataset(key)

    def cancel_compute_plan(self, *args, **kwargs):
        return self._client.cancel_compute_plan(*args, **kwargs)

    def link_dataset_with_data_samples(self, dataset, data_samples):
        self._client.link_dataset_with_data_samples(dataset.key, data_samples)

    def list_compute_plan_traintuples(self, compute_plan_key):
        filters = [
            f'traintuple:compute_plan_key:{compute_plan_key}',
        ]
        tuples = self.list_traintuple(filters=filters)
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_compute_plan_composite_traintuples(self, compute_plan_key):
        filters = [
            f'composite_traintuple:compute_plan_key:{compute_plan_key}',
        ]
        tuples = self.list_composite_traintuple(filters=filters)
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_compute_plan_aggregatetuples(self, compute_plan_key):
        filters = [
            f'aggregatetuple:compute_plan_key:{compute_plan_key}',
        ]
        tuples = self.list_aggregatetuple(filters=filters)
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_compute_plan_testtuples(self, compute_plan_key):
        filters = [
            f'testtuple:compute_plan_key:{compute_plan_key}',
        ]
        tuples = self.list_testtuple(filters=filters)
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def wait(self, asset, timeout=cfg.FUTURE_TIMEOUT, raises=True):
        try:
            m = _get_methods[asset.__class__.__name__]
        except KeyError:
            assert False, 'Future not supported'
        getter = getattr(self, m)

        key = asset.key

        tstart = time.time()
        while asset.status not in [
                Status.done.value,
                Status.failed.value,
                Status.canceled.value,
                ComputePlanStatus.done.value,
                ComputePlanStatus.failed.value,
                ComputePlanStatus.canceled.value]:
            if time.time() - tstart > timeout:
                raise errors.FutureTimeoutError(f'Future timeout on {asset}')

            time.sleep(cfg.FUTURE_POLLING_PERIOD)
            asset = getter(key)

        if raises and asset.status in (Status.failed.value, ComputePlanStatus.failed.value):
            raise errors.FutureFailureError(f'Future execution failed on {asset}')

        if raises and asset.status in (Status.canceled.value, ComputePlanStatus.canceled.value):
            raise errors.FutureFailureError(f'Future execution canceled on {asset}')

        return asset
