import random
import pytest
import copy


ALGO_SCRIPT = """
import json
import substratools as tools
class TestAlgo(tools.Algo):
    def train(self, X, y, models, rank):
        data_sample_keys = [folder.split('/')[-1] for folder in X]
        assert data_sample_keys == {data_sample_keys}, data_sample_keys
        return [0, 1]
    def predict(self, X, model):
        return [0, 99]
    def load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)
if __name__ == '__main__':
    tools.algo.execute(TestAlgo())
"""

COMPOSITE_ALGO_SCRIPT = """
import json
import substratools as tools
class TestCompositeAlgo(tools.CompositeAlgo):
    def train(self, X, y, head_model, trunk_model, rank):
        data_sample_keys = [folder.split('/')[-1] for folder in X]
        assert data_sample_keys == {data_sample_keys}, data_sample_keys
        return [0, 1], [0, 2]
    def predict(self, X, head_model, trunk_model):
        return [0, 99]
    def load_head_model(self, path):
        return self._load_model(path)
    def save_head_model(self, model, path):
        return self._save_model(model, path)
    def load_trunk_model(self, path):
        return self._load_model(path)
    def save_trunk_model(self, model, path):
        return self._save_model(model, path)
    def _load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def _save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)
if __name__ == '__main__':
    tools.algo.execute(TestCompositeAlgo())
"""


def _shuffle(items):
    res = copy.copy(items)
    random.shuffle(res)
    # make sure the items are not sorted in alphabetical order
    while res == sorted(res):
        random.shuffle(res)
    return res


@pytest.mark.xfail(reason='data samples relative order is not preserved (yet)')
def test_traintuple_data_samples_relative_order(factory, client, default_dataset):
    data_sample_keys = _shuffle(default_dataset.train_data_sample_keys)

    algo_script = ALGO_SCRIPT.format(data_sample_keys=data_sample_keys, models=None)
    spec = factory.create_algo(py_script=algo_script)
    algo = client.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=data_sample_keys,
    )
    traintuple = client.add_traintuple(spec)
    assert traintuple.dataset.keys == data_sample_keys
    traintuple.future().wait()


@pytest.mark.xfail(reason='data samples relative order is not preserved (yet)')
def test_composite_traintuple_data_samples_relative_order(factory, client, default_dataset):
    data_sample_keys = _shuffle(default_dataset.train_data_sample_keys)

    composite_algo_script = COMPOSITE_ALGO_SCRIPT.format(data_sample_keys=data_sample_keys, models=None)
    spec = factory.create_composite_algo(py_script=composite_algo_script)
    composite_algo = client.add_composite_algo(spec)

    spec = factory.create_composite_traintuple(
        algo=composite_algo,
        dataset=default_dataset,
        data_samples=data_sample_keys,
    )
    composite_traintuple = client.add_composite_traintuple(spec)
    assert composite_traintuple.dataset.keys == data_sample_keys
    composite_traintuple.future().wait()
