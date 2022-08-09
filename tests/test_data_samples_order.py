import pytest
from substra.sdk.models import Status

import substratest as sbt
from substratest.factory import DEFAULT_DATA_SAMPLE_FILENAME
from substratest.factory import AlgoCategory
from substratest.fl_interface import FLTaskInputGenerator
from substratest.fl_interface import InputIdentifiers

OPENER_SCRIPT = """
import json
import substratools as tools
class TestOpener(tools.Opener):
    def get_X(self, folders):
        return folders
    def get_y(self, folders):
        return folders
    def fake_X(self, n_samples=None):
        pass
    def fake_y(self, n_samples=None):
        pass
    def get_predictions(self, path):
        with open(path) as f:
            return json.load(f)
    def save_predictions(self, y_pred, path):
        with open(path, 'w') as f:
            return json.dump(y_pred, f)
"""

TEMPLATE_ALGO_SCRIPT = """
import json
import substratools as tools
class TestAlgo(tools.Algo):
    def train(self, X, y, models, rank):
        # Check that the order of X is the same as the one passed to add_traintuple
        X_data_sample_keys = [folder.split('/')[-1] for folder in X]
        assert X_data_sample_keys == {data_sample_keys}, X_data_sample_keys

        # Check that the order of y is the same as the one passed to add_traintuple
        y_data_sample_keys = [folder.split('/')[-1] for folder in y]
        assert y_data_sample_keys == {data_sample_keys}, y_data_sample_keys

        # Check that the order of X is the same as the order of y
        assert X_data_sample_keys == y_data_sample_keys

        return [0, 1], [0, 2]

    def predict(self, X, model):
        # Check that the order of X is the same as the one passed to add_predicttuple
        test_data_sample_keys = [folder.split('/')[-1] for folder in X]
        assert test_data_sample_keys == {test_data_sample_keys}, test_data_sample_keys

        return X
    def load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)
if __name__ == '__main__':
    tools.algo.execute(TestAlgo())
"""

TEMPLATE_COMPOSITE_ALGO_SCRIPT = """
import json
import substratools as tools
class TestCompositeAlgo(tools.CompositeAlgo):
    def train(self, X, y, head_model, trunk_model, rank):
        # Check that the order of X is the same as the one passed to add_traintuple
        X_data_sample_keys = [folder.split('/')[-1] for folder in X]
        assert X_data_sample_keys == {data_sample_keys}, X_data_sample_keys

        # Check that the order of y is the same as the one passed to add_traintuple
        y_data_sample_keys = [folder.split('/')[-1] for folder in y]
        assert y_data_sample_keys == {data_sample_keys}, y_data_sample_keys

        # Check that the order of X is the same as the order of y
        assert X_data_sample_keys == y_data_sample_keys

        return [0, 1], [0, 2]

    def predict(self, X, head_model, trunk_model):
        # Check that the order of X is the same as the one passed to add_predicttuple
        test_data_sample_keys = [folder.split('/')[-1] for folder in X]
        assert test_data_sample_keys == {test_data_sample_keys}, test_data_sample_keys

        return X

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

TEMPLATE_METRIC_SCRIPT = """
import substratools as tools
class Metrics(tools.Metrics):
    def score(self, y_true, y_pred):
        y_pred_data_sample_keys = [folder.split('/')[-1] for folder in y_pred]
        assert y_pred_data_sample_keys == {data_sample_keys}

        y_true_data_sample_keys = [folder.split('/')[-1] for folder in y_true]
        assert y_true_data_sample_keys == {data_sample_keys}

        # y_true is a list of unordered data samples
        # since the Algo returns y==x, y_pred should respect the same order

        assert  y_true_data_sample_keys == y_pred_data_sample_keys, (y_true_data_sample_keys, y_pred_data_sample_keys)

        return 1.0
if __name__ == "__main__":
    tools.metrics.execute(Metrics())
"""


def _shuffle(items):
    """Makes sure items are not in alphabetical order or any other kind of order."""
    items = sorted(items)
    return [items[1], items[0]] + items[2:]


class Dataset:
    def __init__(self, factory, client) -> None:
        """Creates a pass through dataset only handling folder names, not actual data."""
        # create dataset
        spec = factory.create_dataset(py_script=OPENER_SCRIPT)
        dataset = client.add_dataset(spec)

        # create train data samples
        for _ in range(4):
            spec = factory.create_data_sample(datasets=[dataset], test_only=False)
            client.add_data_sample(spec)

        # create test data samples
        for _ in range(2):
            spec = factory.create_data_sample(datasets=[dataset], test_only=True)
            client.add_data_sample(spec)

        self.dataset = client.get_dataset(dataset.key)
        self.train_data_sample_keys = _shuffle(self.dataset.train_data_sample_keys)
        self.test_data_sample_keys = self.train_data_sample_keys[:2]
        self.train_data_inputs = FLTaskInputGenerator.tuple(
            opener_key=dataset.key,
            data_sample_keys=self.train_data_sample_keys,
        )
        self.test_data_inputs = FLTaskInputGenerator.tuple(
            opener_key=dataset.key,
            data_sample_keys=self.test_data_sample_keys,
        )


@pytest.fixture
def dataset(factory, client):
    return Dataset(factory, client)


def test_traintuple_data_samples_relative_order(factory, client, dataset):

    # Format TEMPLATE_ALGO_SCRIPT with current data_sample_keys
    algo_script = TEMPLATE_ALGO_SCRIPT.format(
        data_sample_keys=dataset.train_data_sample_keys,
        test_data_sample_keys=dataset.test_data_sample_keys,
        models=None,
    )
    algo_spec = factory.create_algo(category=AlgoCategory.simple, py_script=algo_script)
    algo = client.add_algo(algo_spec)

    predict_algo_spec = factory.create_algo(category=AlgoCategory.predict, py_script=algo_script)
    predict_algo = client.add_algo(predict_algo_spec)

    metric_script = TEMPLATE_METRIC_SCRIPT.format(data_sample_keys=dataset.test_data_sample_keys)
    metric_spec = factory.create_algo(category=AlgoCategory.metric, py_script=metric_script)
    metric = client.add_algo(metric_spec)

    traintuple_spec = factory.create_traintuple(algo=algo, inputs=dataset.train_data_inputs)
    traintuple = client.add_traintuple(traintuple_spec)

    # Ensure the order of the data sample keys is correct at 2 levels: :
    #  1. In the returned traintuple
    #  2. In the train method of the algo. If the order is incorrect, wait() will fail.
    assert [
        i.asset_key for i in traintuple.inputs if i.identifier == InputIdentifiers.datasamples
    ] == dataset.train_data_sample_keys
    client.wait(traintuple)

    predict_input_models = FLTaskInputGenerator.train_to_predict(traintuple.key)
    predicttuple_spec = factory.create_predicttuple(
        algo=predict_algo, inputs=dataset.test_data_inputs + predict_input_models
    )
    predicttuple = client.add_predicttuple(predicttuple_spec)

    test_input_models = FLTaskInputGenerator.predict_to_test(predicttuple.key)

    testtuple_spec = factory.create_testtuple(algo=metric, inputs=dataset.test_data_inputs + test_input_models)
    testtuple = client.add_testtuple(testtuple_spec)

    # Assert order is correct in the metric. If not, wait() will fail.
    client.wait(testtuple)


def test_composite_traintuple_data_samples_relative_order(factory, client, dataset):
    # Format TEMPLATE_COMPOSITE_ALGO_SCRIPT with current data_sample_keys
    composite_algo_script = TEMPLATE_COMPOSITE_ALGO_SCRIPT.format(
        data_sample_keys=dataset.train_data_sample_keys,
        test_data_sample_keys=dataset.test_data_sample_keys,
        models=None,
    )
    algo_spec = factory.create_algo(AlgoCategory.composite, py_script=composite_algo_script)
    composite_algo = client.add_algo(algo_spec)

    predict_algo_script = TEMPLATE_COMPOSITE_ALGO_SCRIPT.format(
        data_sample_keys=dataset.train_data_sample_keys,
        test_data_sample_keys=dataset.test_data_sample_keys,
        models=None,
    )
    predict_algo_spec = factory.create_algo(AlgoCategory.predict, py_script=predict_algo_script)
    predict_algo = client.add_algo(predict_algo_spec)

    metric_script = TEMPLATE_METRIC_SCRIPT.format(
        data_sample_keys=dataset.test_data_sample_keys,
    )
    metric_spec = factory.create_algo(category=AlgoCategory.metric, py_script=metric_script)
    metric = client.add_algo(metric_spec)

    traintuple_spec = factory.create_composite_traintuple(
        algo=composite_algo,
        inputs=dataset.train_data_inputs,
    )
    composite_traintuple = client.add_composite_traintuple(traintuple_spec)
    # Ensure the order of the data sample keys is correct at 2 levels: :
    #  1. In the returned composite traintuple
    #  2. In the train method of the algo. If the order is incorrect, wait() will fail.
    assert [
        i.asset_key for i in composite_traintuple.inputs if i.identifier == InputIdentifiers.datasamples
    ] == dataset.train_data_sample_keys
    client.wait(composite_traintuple)

    predict_input_models = FLTaskInputGenerator.composite_to_predict(composite_traintuple.key)

    predicttuple_spec = factory.create_predicttuple(
        algo=predict_algo, inputs=dataset.test_data_inputs + predict_input_models
    )
    predicttuple = client.add_predicttuple(predicttuple_spec)

    test_input_models = FLTaskInputGenerator.predict_to_test(predicttuple.key)

    testtuple_spec = factory.create_testtuple(algo=metric, inputs=dataset.test_data_inputs + test_input_models)
    testtuple = client.add_testtuple(testtuple_spec)

    # Assert order is correct in the metric. If not, wait() will fail.
    client.wait(testtuple)


@pytest.mark.slow
def test_execution_data_sample_values(factory, client):
    """Check data samples order is preserved when adding data samples by batch."""
    batch_size = 10
    spec = factory.create_dataset(
        py_script=f"""
import json
import os
import substratools as tools
class TestOpener(tools.Opener):
    def get_X(self, folders):
        res = []
        for folder in folders:
            path = os.path.join(folder, '{DEFAULT_DATA_SAMPLE_FILENAME}')
            with open(path, 'r') as f:
                res.append(int(f.read()))
        return res
    def get_y(self, folders):
        return folders
    def fake_X(self, n_samples=None):
        return
    def fake_y(self, n_samples=None):
        return
    def get_predictions(self, path):
        return 0
    def save_predictions(self, y_pred, path):
        with open(path, 'w') as f:
            return json.dump(y_pred, f)
"""
    )
    dataset = client.add_dataset(spec)
    specs = [factory.create_data_sample(content=str(idx), datasets=[dataset]) for idx in range(batch_size)]
    spec = sbt.factory.DataSampleBatchSpec.from_data_sample_specs(specs)
    keys = client.add_data_samples(spec)
    assert len(keys) == batch_size

    spec = factory.create_algo(
        category=AlgoCategory.simple,
        py_script=f"""
import json
import substratools as tools
import os
class TestAlgo(tools.Algo):
    def train(self, X, y, models, rank):
        assert X == list(range({batch_size})), X
        return 0
    def predict(self, X, model):
        return 1
    def load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)

if __name__ == '__main__':
    tools.algo.execute(TestAlgo())
""",
    )
    algo = client.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo, inputs=FLTaskInputGenerator.tuple(opener_key=dataset.key, data_sample_keys=keys)
    )
    traintuple = client.add_traintuple(spec)
    traintuple = client.wait(traintuple)
    assert traintuple.status == Status.done
