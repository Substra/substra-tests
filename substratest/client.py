import os
import tempfile

import substra

from . import assets

DATASET_DOWNLOAD_FILENAME = 'opener.py'


class Client:
    """Client to interact with a Node of Substra.

    Parses responses from server to return Asset instances.
    """

    def __init__(self, backend, node_id=None, address=None, user=None, password=None):
        super().__init__()

        self.node_id = node_id
        self._client = substra.Client(backend=backend, url=address, version="0.0", insecure=False)
        self._client.login(user, password)

    def add_data_sample(self, spec, *args, **kwargs):
        res = self._client.add_data_sample(spec.dict(), *args, **kwargs)
        data_sample = assets.DataSampleCreated.load(res)
        return data_sample

    def add_dataset(self, spec, *args, **kwargs):
        res = self._client.add_dataset(spec.dict(), *args, **kwargs)
        dataset = assets.Dataset.load(res)
        return dataset

    def add_objective(self, spec, *args, **kwargs):
        res = self._client.add_objective(spec.dict(), *args, **kwargs)
        objective = assets.Objective.load(res)
        return objective

    def add_algo(self, spec, *args, **kwargs):
        res = self._client.add_algo(spec.dict(), *args, **kwargs)
        algo = assets.Algo.load(res)
        return algo

    def add_aggregate_algo(self, spec, *args, **kwargs):
        res = self._client.add_aggregate_algo(spec.dict(), *args, **kwargs)
        aggregate_algo = assets.AggregateAlgo.load(res)
        return aggregate_algo

    def add_composite_algo(self, spec, *args, **kwargs):
        res = self._client.add_composite_algo(spec.dict(), *args, **kwargs)
        composite_algo = assets.CompositeAlgo.load(res)
        return composite_algo

    def add_traintuple(self, spec, *args, **kwargs):
        res = self._client.add_traintuple(spec.dict(), *args, **kwargs)
        traintuple = assets.Traintuple.load(res).attach(self)
        return traintuple

    def add_aggregatetuple(self, spec, *args, **kwargs):
        res = self._client.add_aggregatetuple(spec.dict(), *args, **kwargs)
        aggregatetuple = assets.Aggregatetuple.load(res).attach(self)
        return aggregatetuple

    def add_composite_traintuple(self, spec, *args, **kwargs):
        res = self._client.add_composite_traintuple(spec.dict(), *args, **kwargs)
        composite_traintuple = assets.CompositeTraintuple.load(res).attach(self)
        return composite_traintuple

    def add_testtuple(self, spec, *args, **kwargs):
        res = self._client.add_testtuple(spec.dict(), *args, **kwargs)
        testtuple = assets.Testtuple.load(res).attach(self)
        return testtuple

    def add_compute_plan(self, spec, *args, **kwargs):
        res = self._client.add_compute_plan(spec.dict(), *args, **kwargs)
        compute_plan = assets.ComputePlan.load(res).attach(self)
        return compute_plan

    def update_compute_plan(self, spec, *args, **kwargs):
        spec_dict = spec.dict()
        # Remove extra field from data
        spec_dict.pop("compute_plan_id")
        res = self._client.update_compute_plan(spec.compute_plan_id, spec_dict, *args, **kwargs)
        compute_plan = assets.ComputePlan.load(res).attach(self)
        return compute_plan

    def list_compute_plan(self, *args, **kwargs):
        res = self._client.list_compute_plan(*args, **kwargs)
        return [assets.ComputePlan.load(x) for x in res]

    def get_compute_plan(self, *args, **kwargs):
        res = self._client.get_compute_plan(*args, **kwargs)
        compute_plan = assets.ComputePlan.load(res).attach(self)
        return compute_plan

    def list_data_sample(self, *args, **kwargs):
        res = self._client.list_data_sample(*args, **kwargs)
        return [assets.DataSample.load(x) for x in res]

    def get_algo(self, *args, **kwargs):
        res = self._client.get_algo(*args, **kwargs)
        return assets.Algo.load(res)

    def list_algo(self, *args, **kwargs):
        res = self._client.list_algo(*args, **kwargs)
        return [assets.Algo.load(x) for x in res]

    def get_aggregate_algo(self, *args, **kwargs):
        res = self._client.get_aggregate_algo(*args, **kwargs)
        return assets.AggregateAlgo.load(res)

    def list_aggregate_algo(self, *args, **kwargs):
        res = self._client.list_aggregate_algo(*args, **kwargs)
        return [assets.AggregateAlgo.load(x) for x in res]

    def get_composite_algo(self, *args, **kwargs):
        res = self._client.get_composite_algo(*args, **kwargs)
        return assets.CompositeAlgo.load(res)

    def list_composite_algo(self, *args, **kwargs):
        res = self._client.list_composite_algo(*args, **kwargs)
        return [assets.CompositeAlgo.load(x) for x in res]

    def get_dataset(self, *args, **kwargs):
        res = self._client.get_dataset(*args, **kwargs)
        dataset = assets.Dataset.load(res)
        return dataset

    def list_dataset(self, *args, **kwargs):
        res = self._client.list_dataset(*args, **kwargs)
        return [assets.Dataset.load(x) for x in res]

    def get_objective(self, *args, **kwargs):
        res = self._client.get_objective(*args, **kwargs)
        return assets.Objective.load(res)

    def list_objective(self, *args, **kwargs):
        res = self._client.list_objective(*args, **kwargs)
        return [assets.Objective.load(x) for x in res]

    def get_traintuple(self, *args, **kwargs):
        res = self._client.get_traintuple(*args, **kwargs)
        traintuple = assets.Traintuple.load(res).attach(self)
        return traintuple

    def list_traintuple(self, *args, **kwargs):
        res = self._client.list_traintuple(*args, **kwargs)
        return [assets.Traintuple.load(x).attach(self) for x in res]

    def get_aggregatetuple(self, *args, **kwargs):
        res = self._client.get_aggregatetuple(*args, **kwargs)
        aggregatetuple = assets.Aggregatetuple.load(res).attach(self)
        return aggregatetuple

    def list_aggregatetuple(self, *args, **kwargs):
        res = self._client.list_aggregatetuple(*args, **kwargs)
        return [assets.Aggregatetuple.load(x).attach(self) for x in res]

    def get_composite_traintuple(self, *args, **kwargs):
        res = self._client.get_composite_traintuple(*args, **kwargs)
        composite_traintuple = assets.CompositeTraintuple.load(res).attach(self)
        return composite_traintuple

    def list_composite_traintuple(self, *args, **kwargs):
        res = self._client.list_composite_traintuple(*args, **kwargs)
        return [assets.CompositeTraintuple.load(x).attach(self) for x in res]

    def get_testtuple(self, *args, **kwargs):
        res = self._client.get_testtuple(*args, **kwargs)
        testtuple = assets.Testtuple.load(res).attach(self)
        return testtuple

    def list_testtuple(self, *args, **kwargs):
        res = self._client.list_testtuple(*args, **kwargs)
        return [assets.Testtuple.load(x).attach(self) for x in res]

    def list_node(self, *args, **kwargs):
        res = self._client.list_node(*args, **kwargs)
        return [assets.Node.load(x) for x in res]

    def download_opener(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            self._client.download_dataset(key, tmp)
            path = os.path.join(tmp, DATASET_DOWNLOAD_FILENAME)
            with open(path, 'rb') as f:
                return f.read()

    def describe_dataset(self, key):
        return self._client.describe_dataset(key)

    def cancel_compute_plan(self, *args, **kwargs):
        res = self._client.cancel_compute_plan(*args, **kwargs)
        compute_plan = assets.ComputePlan.load(res).attach(self)
        return compute_plan

    def link_dataset_with_objective(self, dataset, objective):
        self._client.link_dataset_with_objective(dataset.key, objective.key)
        # XXX do not return anything as currently the chaincode simply returns the
        #     updated dataset key

    def link_dataset_with_data_samples(self, dataset, data_samples):
        data_sample_keys = [d.key for d in data_samples]
        self._client.link_dataset_with_data_samples(dataset.key, data_sample_keys)
