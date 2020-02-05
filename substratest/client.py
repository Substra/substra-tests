import os
import tempfile
import copy

import substra

from . import assets

DATASET_DOWNLOAD_FILENAME = 'opener.py'


class _State:
    """Session state.

    Represents all the assets that have been added during the life of the session.
    """

    def __init__(self):
        self.aggregate_algos = []
        self.aggregatetuples = []
        self.algos = []
        self.composite_algos = []
        self.composite_traintuples = []
        self.compute_plans = []
        self.datasets = []
        self.objectives = []
        self.test_data_samples = []
        self.testtuples = []
        self.train_data_samples = []
        self.traintuples = []

    def _update_assets(self, asset_list_name, asset):
        """Replace or insert asset in self.[asset_list_name]"""
        asset_list = getattr(self, asset_list_name)
        new_asset_list = []
        found = False
        for a in asset_list:
            if asset.key == a.key:
                found = True
                new_asset_list.append(asset)
            else:
                new_asset_list.append(a)
        if not found:
            new_asset_list.append(asset)
        setattr(self, asset_list_name, new_asset_list)

    def update_aggregate_algo(self, aggregate_algo):
        self._update_assets('aggregate_algos', aggregate_algo)

    def update_aggregatetuple(self, aggregatetuple):
        self._update_assets('aggregatetuples', aggregatetuple)

    def update_algo(self, algo):
        self._update_assets('algos', algo)

    def update_composite_algo(self, composite_algo):
        self._update_assets('composite_algos', composite_algo)

    def update_composite_traintuple(self, composite_traintuple):
        self._update_assets('composite_traintuples', composite_traintuple)

    def update_compute_plan(self, compute_plan):
        self.compute_plans = [
            compute_plan if cp.compute_plan_id == compute_plan.compute_plan_id else cp
            for cp in self.compute_plans
        ]

    def update_dataset(self, dataset):
        self._update_assets('datasets', dataset)

    def update_objective(self, objective):
        self._update_assets('objectives', objective)

    def update_testtuple(self, testtuple):
        self._update_assets('testtuples', testtuple)

    def update_traintuple(self, traintuple):
        self._update_assets('traintuples', traintuple)


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

    def copy(self):
        return copy.deepcopy(self)

    def add_data_sample(self, spec, *args, **kwargs):
        res = self._client.add_data_sample(spec.to_dict(), *args, **kwargs)
        data_sample = assets.DataSampleCreated.load(res)

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

    def add_objective(self, spec, *args, **kwargs):
        res = self._client.add_objective(spec.to_dict(), *args, **kwargs)
        objective = assets.Objective.load(res)
        self.state.objectives.append(objective)
        return objective

    def add_algo(self, spec, *args, **kwargs):
        res = self._client.add_algo(spec.to_dict(), *args, **kwargs)
        algo = assets.Algo.load(res)
        self.state.algos.append(algo)
        return algo

    def add_aggregate_algo(self, spec, *args, **kwargs):
        res = self._client.add_aggregate_algo(spec.to_dict(), *args, **kwargs)
        aggregate_algo = assets.AggregateAlgo.load(res)
        self.state.aggregate_algos.append(aggregate_algo)
        return aggregate_algo

    def add_composite_algo(self, spec, *args, **kwargs):
        res = self._client.add_composite_algo(spec.to_dict(), *args, **kwargs)
        composite_algo = assets.CompositeAlgo.load(res)
        self.state.composite_algos.append(composite_algo)
        return composite_algo

    def add_traintuple(self, spec, *args, **kwargs):
        res = self._client.add_traintuple(spec.to_dict(), *args, **kwargs)
        traintuple = assets.Traintuple.load(res).attach(self)
        self.state.traintuples.append(traintuple)
        return traintuple

    def add_aggregatetuple(self, spec, *args, **kwargs):
        res = self._client.add_aggregatetuple(spec.to_dict(), *args, **kwargs)
        aggregatetuple = assets.Aggregatetuple.load(res).attach(self)
        self.state.aggregatetuples.append(aggregatetuple)
        return aggregatetuple

    def add_composite_traintuple(self, spec, *args, **kwargs):
        res = self._client.add_composite_traintuple(spec.to_dict(), *args, **kwargs)
        composite_traintuple = assets.CompositeTraintuple.load(res).attach(self)
        self.state.composite_traintuples.append(composite_traintuple)
        return composite_traintuple

    def add_testtuple(self, spec, *args, **kwargs):
        res = self._client.add_testtuple(spec.to_dict(), *args, **kwargs)
        testtuple = assets.Testtuple.load(res).attach(self)
        self.state.testtuples.append(testtuple)
        return testtuple

    def add_compute_plan(self, spec, *args, **kwargs):
        res = self._client.add_compute_plan(spec.to_dict(), *args, **kwargs)
        compute_plan = assets.ComputePlan.load(res).attach(self)
        self.state.compute_plans.append(compute_plan)
        return compute_plan

    def list_compute_plan(self, *args, **kwargs):
        res = self._client.list_compute_plan(*args, **kwargs)
        return [assets.ComputePlan.load(x) for x in res]

    def get_compute_plan(self, *args, **kwargs):
        res = self._client.get_compute_plan(*args, **kwargs)
        compute_plan = assets.ComputePlan.load(res).attach(self)
        self.state.update_compute_plan(compute_plan)
        return compute_plan

    def list_data_sample(self, *args, **kwargs):
        res = self._client.list_data_sample(*args, **kwargs)
        return [assets.DataSample.load(x) for x in res]

    def get_algo(self, *args, **kwargs):
        res = self._client.get_algo(*args, **kwargs)
        algo = assets.Algo.load(res)
        self.state.update_algo(algo)
        return algo

    def list_algo(self, *args, **kwargs):
        res = self._client.list_algo(*args, **kwargs)
        return [assets.Algo.load(x) for x in res]

    def get_aggregate_algo(self, *args, **kwargs):
        res = self._client.get_aggregate_algo(*args, **kwargs)
        aggregate_algo = assets.AggregateAlgo.load(res)
        self.state.update_aggregate_algo(aggregate_algo)
        return aggregate_algo

    def list_aggregate_algo(self, *args, **kwargs):
        res = self._client.list_aggregate_algo(*args, **kwargs)
        return [assets.AggregateAlgo.load(x) for x in res]

    def get_composite_algo(self, *args, **kwargs):
        res = self._client.get_composite_algo(*args, **kwargs)
        composite_algo = assets.CompositeAlgo.load(res)
        self.state.update_composite_algo(composite_algo)
        return composite_algo

    def list_composite_algo(self, *args, **kwargs):
        res = self._client.list_composite_algo(*args, **kwargs)
        return [assets.CompositeAlgo.load(x) for x in res]

    def get_dataset(self, *args, **kwargs):
        res = self._client.get_dataset(*args, **kwargs)
        dataset = assets.Dataset.load(res)
        self.state.update_dataset(dataset)
        return dataset

    def list_dataset(self, *args, **kwargs):
        res = self._client.list_dataset(*args, **kwargs)
        return [assets.Dataset.load(x) for x in res]

    def get_objective(self, *args, **kwargs):
        res = self._client.get_objective(*args, **kwargs)
        objective = assets.Objective.load(res)
        self.state.update_objective(objective)
        return objective

    def list_objective(self, *args, **kwargs):
        res = self._client.list_objective(*args, **kwargs)
        return [assets.Objective.load(x) for x in res]

    def get_traintuple(self, *args, **kwargs):
        res = self._client.get_traintuple(*args, **kwargs)
        traintuple = assets.Traintuple.load(res).attach(self)
        self.state.update_traintuple(traintuple)
        return traintuple

    def list_traintuple(self, *args, **kwargs):
        res = self._client.list_traintuple(*args, **kwargs)
        return [assets.Traintuple.load(x).attach(self) for x in res]

    def get_aggregatetuple(self, *args, **kwargs):
        res = self._client.get_aggregatetuple(*args, **kwargs)
        aggregatetuple = assets.Aggregatetuple.load(res).attach(self)
        self.state.update_aggregatetuple(aggregatetuple)
        return aggregatetuple

    def list_aggregatetuple(self, *args, **kwargs):
        res = self._client.list_aggregatetuple(*args, **kwargs)
        return [assets.Aggregatetuple.load(x).attach(self) for x in res]

    def get_composite_traintuple(self, *args, **kwargs):
        res = self._client.get_composite_traintuple(*args, **kwargs)
        composite_traintuple = assets.CompositeTraintuple.load(res).attach(self)
        self.state.update_composite_traintuple(composite_traintuple)
        return composite_traintuple

    def list_composite_traintuple(self, *args, **kwargs):
        res = self._client.list_composite_traintuple(*args, **kwargs)
        return [assets.CompositeTraintuple.load(x).attach(self) for x in res]

    def get_testtuple(self, *args, **kwargs):
        res = self._client.get_testtuple(*args, **kwargs)
        testtuple = assets.Testtuple.load(res).attach(self)
        self.state.update_testtuple(testtuple)
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
        self.state.update_compute_plan(compute_plan)
        return compute_plan

    def link_dataset_with_objective(self, dataset, objective):
        self._client.link_dataset_with_objective(dataset.key, objective.key)
        # XXX do not return anything as currently the chaincode simply returns the
        #     updated dataset key
