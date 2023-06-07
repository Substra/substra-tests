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


class Client(substra.Client):
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
        super().__init__(
            backend_type=backend_type, url=address, insecure=False, token=token, username=user, password=password
        )

        self.organization_id = organization_id

        self._api_client = _APIClient(address, self._token)
        self.future_timeout = future_timeout
        self.future_polling_period = future_polling_period

    def add_data_sample(self, spec, *args, **kwargs):
        key = super().add_data_sample(spec.dict(), *args, **kwargs)
        return key

    def add_data_samples(self, spec, *args, **kwargs):
        keys = super().add_data_samples(spec.dict(), *args, **kwargs)
        return keys

    def add_dataset(self, spec, *args, **kwargs):
        key = super().add_dataset(spec.dict(), *args, **kwargs)
        return super().get_dataset(key)

    def add_function(self, spec, *args, **kwargs):
        key = super().add_function(spec.dict(), *args, **kwargs)
        return super().get_function(key)

    def add_task(self, spec, *args, **kwargs):
        key = super().add_task(spec.dict(), *args, **kwargs)
        return super().get_task(key)

    def add_compute_plan(self, spec, *args, **kwargs):
        return super().add_compute_plan(spec.dict(), *args, **kwargs)

    def add_compute_plan_tasks(self, spec, *args, **kwargs):
        spec_dict = spec.dict()
        # Remove extra field from data
        spec_dict.pop("key")
        return super().add_compute_plan_tasks(spec.key, spec_dict, *args, **kwargs)

    def download_opener(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            path = super().download_dataset(key, tmp)
            with open(path, "rb") as f:
                return f.read()

    def download_function(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            path = super().download_function(key, tmp)
            with open(path, "rb") as f:
                return f.read()

    def get_model_content(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            path = super().download_model(key, tmp)
            with open(path, "rb") as f:
                return f.read()

    def get_model_content_from_task(self, task_key, identifier):
        with tempfile.TemporaryDirectory() as tmp:
            path = super().download_model_from_task(task_key, identifier=identifier, folder=tmp)
            with open(path, "rb") as f:
                return f.read()

    def get_task_models(self, compute_task_key: str) -> typing.List[substra.models.OutModel]:
        return super().list_model(filters={"compute_task_key": [compute_task_key]})

    def download_logs(self, task_key):
        with tempfile.TemporaryDirectory() as tmp:
            path = super().download_logs(task_key, tmp)
            with open(path, "r") as f:
                return f.read()

    def link_dataset_with_data_samples(self, dataset, data_samples):
        super().link_dataset_with_data_samples(dataset.key, data_samples)

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

    def wait_compute_plan(self, key: str, **kwargs):
        return self._wait(key=key, asset_type=models.ComputePlan, **kwargs)

    def wait_task(self, key: str, **kwargs):
        return self._wait(key=key, asset_type=models.Task, **kwargs)

    def _wait(self, *, key: str, asset_type, raises=True, timeout=None):
        if timeout is None:
            timeout = self.future_timeout

        tstart = time.time()
        while True:
            if asset_type == models.ComputePlan:
                asset = self.get_compute_plan(key)
                asset_failed = ComputePlanStatus.failed.value
                asset_canceled = ComputePlanStatus.canceled.value
                asset_stopped = (
                    ComputePlanStatus.done.value,
                    ComputePlanStatus.failed.value,
                    ComputePlanStatus.canceled.value,
                )
            elif asset_type == models.Task:
                asset = self.get_task(key)
                asset_canceled = Status.canceled.value
                asset_failed = Status.failed.value
                asset_stopped = (Status.done.value, Status.canceled.value)
            else:
                raise ValueError(f"{asset_type} cannot be waited")

            if asset.status in asset_stopped:
                break

            if asset.status == Status.failed.value and asset.error_type is not None:
                # when dealing with a failed task, wait for the error_type field of the task to be set
                # i.e. wait for the registration of the failure report
                break

            if time.time() - tstart > timeout:
                raise errors.FutureTimeoutError(f"Future timeout on {asset}")

            time.sleep(self.future_polling_period)

        if raises and asset.status == asset_failed:
            raise errors.FutureFailureError(f"Future execution failed on {asset}")

        if raises and asset.status == asset_canceled:
            raise errors.FutureFailureError(f"Future execution canceled on {asset}")

        return asset

    def wait_model_deletion(self, model_key):
        """Wait for the model to be deleted (address unset)"""
        tstart = time.time()
        model = super().get_model(model_key)
        while model.address:
            if time.time() - tstart > self.future_timeout:
                raise errors.FutureTimeoutError(f"Future timeout waiting on model deletion for {model_key}")

            time.sleep(self.future_polling_period)
            super().get_model(model_key)

    def update_function(self, function, name, *args, **kwargs):
        return super().update_function(function.key, name, *args, **kwargs)

    def update_compute_plan(self, compute_plan, name, *args, **kwargs):
        return super().update_compute_plan(compute_plan.key, name, *args, **kwargs)

    def update_dataset(self, dataset, name, *args, **kwargs):
        return super().update_dataset(dataset.key, name, *args, **kwargs)

    def get_compute_task_profiling(self, task_key: str):
        return self._api_client.get_compute_task_profiling(task_key)
