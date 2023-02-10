import tempfile
import time
import typing
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
        backend_type: substra.BackendType,
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
        self._client = substra.Client(backend_type=backend_type, url=address, insecure=False, token=token)
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

    def add_function(self, spec, *args, **kwargs):
        key = self._client.add_function(spec.dict(), *args, **kwargs)
        return self._client.get_function(key)

    def add_task(self, spec, *args, **kwargs):
        key = self._client.add_task(spec.dict(), *args, **kwargs)
        return self._client.get_task(key)

    def add_compute_plan(self, spec, *args, **kwargs):
        return self._client.add_compute_plan(spec.dict(), *args, **kwargs)

    def add_compute_plan_tasks(self, spec, *args, **kwargs):
        spec_dict = spec.dict()
        # Remove extra field from data
        spec_dict.pop("key")
        return self._client.add_compute_plan_tasks(spec.key, spec_dict, *args, **kwargs)

    def list_compute_plan(self, *args, **kwargs):
        return self._client.list_compute_plan(*args, **kwargs)

    def get_compute_plan(self, *args, **kwargs):
        return self._client.get_compute_plan(*args, **kwargs)

    def get_performances(self, *args, **kwargs):
        return self._client.get_performances(*args, **kwargs)

    def list_data_sample(self, *args, **kwargs):
        return self._client.list_data_sample(*args, **kwargs)

    def get_function(self, *args, **kwargs):
        return self._client.get_function(*args, **kwargs)

    def list_function(self, *args, **kwargs):
        return self._client.list_function(*args, **kwargs)

    def get_dataset(self, *args, **kwargs):
        return self._client.get_dataset(*args, **kwargs)

    def get_data_sample(self, *args, **kwargs):
        return self._client.get_data_sample(*args, **kwargs)

    def list_dataset(self, *args, **kwargs):
        return self._client.list_dataset(*args, **kwargs)

    def get_task(self, *args, **kwargs):
        return self._client.get_task(*args, **kwargs)

    def list_task(self, *args, **kwargs):
        return self._client.list_task(*args, **kwargs)

    def list_organization(self, *args, **kwargs):
        return self._client.list_organization(*args, **kwargs)

    def download_opener(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._client.download_dataset(key, tmp)
            with open(path, "rb") as f:
                return f.read()

    def download_function(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._client.download_function(key, tmp)
            with open(path, "rb") as f:
                return f.read()

    def download_model(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._client.download_model(key, tmp)
            with open(path, "rb") as f:
                return f.read()

    def download_model_from_task(self, task_key, identifier):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._client.download_model_from_task(task_key, identifier=identifier, folder=tmp)
            with open(path, "rb") as f:
                return f.read()

    def get_task_models(self, compute_task_key: str) -> typing.List[substra.models.OutModel]:
        return self._client.list_model(filters={"compute_task_key": [compute_task_key]})

    def get_logs(self, task_key):
        return self._client.get_logs(task_key)

    def download_logs(self, task_key):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._client.download_logs(task_key, tmp)
            with open(path, "r") as f:
                return f.read()

    def describe_dataset(self, key):
        return self._client.describe_dataset(key)

    def cancel_compute_plan(self, *args, **kwargs) -> None:
        self._client.cancel_compute_plan(*args, **kwargs)

    def link_dataset_with_data_samples(self, dataset, data_samples):
        self._client.link_dataset_with_data_samples(dataset.key, data_samples)

    def list_compute_plan_tasks(self, compute_plan_key):
        return self.list_task(filters={"compute_plan_key": [compute_plan_key]})

    def get(self, asset):
        """Asset getter (valid only for first class asset)."""
        getters = {
            models.Dataset: self.get_dataset,
            models.Function: self.get_function,
            models.Task: self.get_task,
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
                # when dealing with a failed task, wait for the error_type field of the task to be set
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

    def update_function(self, function, name, *args, **kwargs):
        return self._client.update_function(function.key, name, *args, **kwargs)

    def update_compute_plan(self, compute_plan, name, *args, **kwargs):
        return self._client.update_compute_plan(compute_plan.key, name, *args, **kwargs)

    def update_dataset(self, dataset, name, *args, **kwargs):
        return self._client.update_dataset(dataset.key, name, *args, **kwargs)

    def get_compute_task_profiling(self, task_key: str):
        return self._api_client.get_compute_task_profiling(task_key)

    def organization_info(self):
        return self._client.organization_info()
