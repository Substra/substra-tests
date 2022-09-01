import tempfile
import time
from typing import Optional

import requests
import substra
from substra.sdk import models
from substra.sdk.models import ComputePlanStatus
from substra.sdk.models import Status

from . import errors


class _APIClient:
    def __init__(self, url: str, auth_token: str) -> None:
        self._base_url = url
        self._headers = {
            "Authorization": f"Token {auth_token}",
            "Accept": "application/json;version=0.0",
        }

    def _get(self, path: str) -> requests.Response:
        url = self._base_url + path
        response = requests.get(url, headers=self._headers)
        response.raise_for_status()
        return response

    def get_compute_task_profiling(self, task_key: str):
        return self._get(f"/task_profiling/{task_key}/").json()


class Client:
    """Client to interact with a Organization of Substra.

    Parses responses from server to return Asset instances.
    """

    def __init__(
        self,
        debug: bool,
        organization_id: str,
        address: str,
        user: str,
        password: str,
        future_timeout: int,
        future_polling_period: int,
        token: Optional[str] = None,
    ):

        super().__init__()

        self.organization_id = organization_id
        self._client = substra.Client(debug=debug, url=address, insecure=False, token=token)
        if not token:
            token = self._client.login(user, password)
        self._api_client = _APIClient(address, token)
        self.backend_mode = self._client.backend_mode
        self.token = token
        self.future_timeout = future_timeout
        self.future_polling_period = future_polling_period

    def add_data_sample(self, spec, *args, **kwargs):
        key = self._client.add_data_sample(spec.dict(), *args, **kwargs)
        return key

    def add_data_samples(self, spec, *args, **kwargs):
        keys = self._client.add_data_samples(spec.dict(), *args, **kwargs)
        return keys

    def add_dataset(self, spec, *args, **kwargs):
        key = self._client.add_dataset(spec.dict(), *args, **kwargs)
        return self._client.get_dataset(key)

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

    def add_predicttuple(self, spec, *args, **kwargs):
        key = self._client.add_predicttuple(spec.dict(), *args, **kwargs)
        return self._client.get_predicttuple(key)

    def add_testtuple(self, spec, *args, **kwargs):
        key = self._client.add_testtuple(spec.dict(), *args, **kwargs)
        return self._client.get_testtuple(key)

    def add_compute_plan(self, spec, *args, **kwargs):
        return self._client.add_compute_plan(spec.dict(), *args, **kwargs)

    def add_compute_plan_tuples(self, spec, *args, **kwargs):
        spec_dict = spec.dict()
        # Remove extra field from data
        spec_dict.pop("key")
        return self._client.add_compute_plan_tuples(spec.key, spec_dict, *args, **kwargs)

    def list_compute_plan(self, *args, **kwargs):
        return self._client.list_compute_plan(*args, **kwargs)

    def get_compute_plan(self, *args, **kwargs):
        return self._client.get_compute_plan(*args, **kwargs)

    def get_performances(self, *args, **kwargs):
        return self._client.get_performances(*args, **kwargs)

    def list_data_sample(self, *args, **kwargs):
        return self._client.list_data_sample(*args, **kwargs)

    def get_algo(self, *args, **kwargs):
        return self._client.get_algo(*args, **kwargs)

    def list_algo(self, *args, **kwargs):
        return self._client.list_algo(*args, **kwargs)

    def get_dataset(self, *args, **kwargs):
        return self._client.get_dataset(*args, **kwargs)

    def get_data_sample(self, *args, **kwargs):
        return self._client.get_data_sample(*args, **kwargs)

    def list_dataset(self, *args, **kwargs):
        return self._client.list_dataset(*args, **kwargs)

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

    def get_predicttuple(self, *args, **kwargs):
        return self._client.get_predicttuple(*args, **kwargs)

    def list_predicttuple(self, *args, **kwargs):
        return self._client.list_predicttuple(*args, **kwargs)

    def get_testtuple(self, *args, **kwargs):
        return self._client.get_testtuple(*args, **kwargs)

    def list_testtuple(self, *args, **kwargs):
        return self._client.list_testtuple(*args, **kwargs)

    def list_organization(self, *args, **kwargs):
        return self._client.list_organization(*args, **kwargs)

    def download_opener(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._client.download_dataset(key, tmp)
            with open(path, "rb") as f:
                return f.read()

    def download_algo(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._client.download_algo(key, tmp)
            with open(path, "rb") as f:
                return f.read()

    def download_model(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._client.download_model(key, tmp)
            with open(path, "rb") as f:
                return f.read()

    def download_trunk_model_from_composite_traintuple(self, composite_traintuple_key):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._client.download_trunk_model_from_composite_traintuple(composite_traintuple_key, tmp)
            with open(path, "rb") as f:
                return f.read()

    def get_logs(self, tuple_key):
        return self._client.get_logs(tuple_key)

    def download_logs(self, tuple_key):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._client.download_logs(tuple_key, tmp)
            with open(path, "r") as f:
                return f.read()

    def describe_dataset(self, key):
        return self._client.describe_dataset(key)

    def cancel_compute_plan(self, *args, **kwargs) -> None:
        self._client.cancel_compute_plan(*args, **kwargs)

    def link_dataset_with_data_samples(self, dataset, data_samples):
        self._client.link_dataset_with_data_samples(dataset.key, data_samples)

    def list_compute_plan_traintuples(self, compute_plan_key):
        filters = {"compute_plan_key": [compute_plan_key]}
        tuples = self.list_traintuple(filters=filters)
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_compute_plan_composite_traintuples(self, compute_plan_key):
        filters = {"compute_plan_key": [compute_plan_key]}
        tuples = self.list_composite_traintuple(filters=filters)
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_compute_plan_aggregatetuples(self, compute_plan_key):
        filters = {"compute_plan_key": [compute_plan_key]}
        tuples = self.list_aggregatetuple(filters=filters)
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_compute_plan_predicttuples(self, compute_plan_key):
        filters = {"compute_plan_key": [compute_plan_key]}
        tuples = self.list_predicttuple(filters=filters)
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def list_compute_plan_testtuples(self, compute_plan_key):
        filters = {"compute_plan_key": [compute_plan_key]}
        tuples = self.list_testtuple(filters=filters)
        tuples = sorted(tuples, key=lambda t: t.rank)
        return tuples

    def get(self, asset):
        """Asset getter (valid only for first class asset)."""
        getters = {
            models.Dataset: self.get_dataset,
            models.Algo: self.get_algo,
            models.Traintuple: self.get_traintuple,
            models.Predicttuple: self.get_predicttuple,
            models.Testtuple: self.get_testtuple,
            models.Aggregatetuple: self.get_aggregatetuple,
            models.CompositeTraintuple: self.get_composite_traintuple,
            models.ComputePlan: self.get_compute_plan,
            models.DataSample: self.get_data_sample,
        }

        try:
            getter = getters[asset.__class__]
        except KeyError:
            raise AssertionError(f"Cannot find getter for asset {asset}")

        return getter(asset.key)

    def wait(self, asset, raises=True, timeout=None):

        if timeout is None:
            timeout = self.future_timeout

        tstart = time.time()
        while True:
            if asset.status in (
                Status.done.value,
                Status.canceled.value,
                ComputePlanStatus.done.value,
                ComputePlanStatus.failed.value,
                ComputePlanStatus.canceled.value,
            ):
                break

            if asset.status == Status.failed.value and asset.error_type is not None:
                # when dealing with a failed tuple, wait for the error_type field of the tuple to be set
                # i.e. wait for the registration of the failure report
                break

            if time.time() - tstart > timeout:
                raise errors.FutureTimeoutError(f"Future timeout on {asset}")

            time.sleep(self.future_polling_period)
            asset = self.get(asset)

        if raises and asset.status in (Status.failed.value, ComputePlanStatus.failed.value):
            raise errors.FutureFailureError(f"Future execution failed on {asset}")

        if raises and asset.status in (Status.canceled.value, ComputePlanStatus.canceled.value):
            raise errors.FutureFailureError(f"Future execution canceled on {asset}")

        return asset

    def wait_model_deletion(self, model_key):
        """Wait for the model to be deleted (address unset)"""
        tstart = time.time()
        model = self._client.get_model(model_key)
        while model.address:
            if time.time() - tstart > self.future_timeout:
                raise errors.FutureTimeoutError(f"Future timeout waiting on model deletion for {model_key}")

            time.sleep(self.future_polling_period)
            model = self._client.get_model(model_key)

    def update_algo(self, algo, name, *args, **kwargs):
        return self._client.update_algo(algo.key, name, *args, **kwargs)

    def update_compute_plan(self, compute_plan, name, *args, **kwargs):
        return self._client.update_compute_plan(compute_plan.key, name, *args, **kwargs)

    def update_dataset(self, dataset, name, *args, **kwargs):
        return self._client.update_dataset(dataset.key, name, *args, **kwargs)

    def get_compute_task_profiling(self, task_key: str):
        return self._api_client.get_compute_task_profiling(task_key)
