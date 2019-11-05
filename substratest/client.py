import os
import tempfile

import substra

from . import assets

DATASET_DOWNLOAD_FILENAME = 'opener.py'


class _State:
    """Session state."""

    # TODO handle state updates when getting asset for instance; that would be
    #      particularly useful for dataset update when adding a traintuple for instance

    def __init__(self):
        self.datasets = []
        self.test_data_samples = []
        self.train_data_samples = []
        self.objectives = []
        self.algos = []
        self.traintuples = []
        self.testtuples = []


class Session:
    """Client to interact with a Node of Substra.

    Stores asset(s) added during the session.
    Parses responses from server to return Asset instances.
    """

    def __init__(self, node_name, node_id, address, user, password):
        super().__init__()
        # session added/modified assets during the session lifetime
        self.state = _State()

        # node / client
        self.node_id = node_id
        self._client = substra.Client()
        self._client.add_profile(node_name, user, password, address, '0.0')
        self._client.login()

    def add_data_sample(self, spec):
        res = self._client.add_data_sample(spec.to_dict())
        data_sample = assets.DataSample.load(res)

        if spec.test_only:
            self.state.test_data_samples.append(data_sample)
        else:
            self.state.train_data_samples.append(data_sample)

        return data_sample

    def add_dataset(self, spec, *args, **kwargs):
        res = self._client.add_dataset(spec.to_dict(), *args, **kwargs)
        dataset = assets.Dataset.load(res)
        self.state.datasets.append(dataset)
        return dataset

    def add_objective(self, spec):
        res = self._client.add_objective(spec.to_dict())
        objective = assets.Objective.load(res)
        self.state.objectives.append(objective)
        return objective

    def add_algo(self, spec):
        res = self._client.add_algo(spec.to_dict())
        algo = assets.Algo.load(res)
        self.state.algos.append(algo)
        return algo

    def add_traintuple(self, spec, *args, **kwargs):
        res = self._client.add_traintuple(spec.to_dict(), *args, **kwargs)
        traintuple = assets.Traintuple.load(res).attach(self)
        self.state.traintuples.append(traintuple)
        return traintuple

    def add_testtuple(self, spec):
        res = self._client.add_testtuple(spec.to_dict())
        testtuple = assets.Testtuple.load(res).attach(self)
        self.state.testtuples.append(testtuple)
        return testtuple

    def add_compute_plan(self, spec):
        res = self._client.add_compute_plan(spec.to_dict())
        cp = assets.ComputePlan.load(res)
        return cp

    def get_algo(self, *args, **kwargs):
        res = self._client.get_algo(*args, **kwargs)
        return assets.Algo.load(res)

    def list_algo(self, *args, **kwargs):
        res = self._client.list_algo(*args, **kwargs)
        return [assets.Algo.load(x) for x in res]

    def get_dataset(self, *args, **kwargs):
        res = self._client.get_dataset(*args, **kwargs)
        return assets.Dataset.load(res)

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
        return assets.Traintuple.load(res).attach(self)

    def list_traintuple(self, *args, **kwargs):
        res = self._client.list_traintuple(*args, **kwargs)
        return [assets.Traintuple.load(x) for x in res]

    def get_testtuple(self, *args, **kwargs):
        res = self._client.get_testtuple(*args, **kwargs)
        return assets.Testtuple.load(res).attach(self)

    def list_testtuple(self, *args, **kwargs):
        res = self._client.list_testtuple(*args, **kwargs)
        return [assets.Testtuple.load(x) for x in res]

    def list_node(self, *args, **kwargs):
        res = self._client.list_node(*args, **kwargs)
        return [assets.Node.load(x) for x in res]

    def download_opener(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            self._client.download_dataset(key, tmp)
            path = os.path.join(tmp, DATASET_DOWNLOAD_FILENAME)
            with open(path, 'rb') as f:
                return f.read()
