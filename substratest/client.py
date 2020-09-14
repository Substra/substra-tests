import os
import tempfile

import substra

from . import assets

DATASET_DOWNLOAD_FILENAME = 'opener.py'


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

    def add_dataset(self, spec, *args, **kwargs):
        key = self._client.add_dataset(spec.dict(), *args, **kwargs)
        return self._client.get_dataset(key)

    def add_objective(self, spec, *args, **kwargs):
        key = self._client.add_objective(spec.dict(), *args, **kwargs)
        return self._client.get_objective(key)

    def add_algo(self, spec, *args, **kwargs):
        key = self._client.add_algo(spec.dict(), *args, **kwargs)
        return self._client.get_algo(key)

    def add_aggregate_algo(self, spec, *args, **kwargs):
        key = self._client.add_aggregate_algo(spec.dict(), *args, **kwargs)
        return self._client.get_aggregate_algo(key)

    def add_composite_algo(self, spec, *args, **kwargs):
        key = self._client.add_composite_algo(spec.dict(), *args, **kwargs)
        return self._client.get_composite_algo(key)

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
        spec_dict.pop("compute_plan_id")
        return self._client.update_compute_plan(spec.compute_plan_id, spec_dict, *args, **kwargs)

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

    def get_aggregate_algo(self, *args, **kwargs):
        return self._client.get_aggregate_algo(*args, **kwargs)

    def list_aggregate_algo(self, *args, **kwargs):
        return self._client.list_aggregate_algo(*args, **kwargs)

    def get_composite_algo(self, *args, **kwargs):
        return self._client.get_composite_algo(*args, **kwargs)

    def list_composite_algo(self, *args, **kwargs):
        return self._client.list_composite_algo(*args, **kwargs)

    def get_dataset(self, *args, **kwargs):
        return self._client.get_dataset(*args, **kwargs)

    def list_dataset(self, *args, **kwargs):
        return self._client.list_dataset(*args, **kwargs)

    def get_objective(self, *args, **kwargs):
        return self._client.get_objective(*args, **kwargs)

    def list_objective(self, *args, **kwargs):
        return self._client.list_objective(*args, **kwargs)

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

    def describe_dataset(self, key):
        return self._client.describe_dataset(key)

    def cancel_compute_plan(self, *args, **kwargs):
        return self._client.cancel_compute_plan(*args, **kwargs)

    def link_dataset_with_objective(self, dataset, objective):
        self._client.link_dataset_with_objective(dataset.key, objective.key)
        # XXX do not return anything as currently the chaincode simply returns the
        #     updated dataset key

    def link_dataset_with_data_samples(self, dataset, data_samples):
        self._client.link_dataset_with_data_samples(dataset.key, data_samples)
