import pytest
from substra.sdk.models import Status

import substratest as sbt
from substratest.factory import DEFAULT_DATA_SAMPLE_FILENAME
from substratest.fl_interface import AlgoCategory
from substratest.fl_interface import FLTaskInputGenerator
from substratest.fl_interface import InputIdentifiers
from substratest.fl_interface import OutputIdentifiers

OPENER_SCRIPT = """
import json
import substratools as tools
class TestOpener(tools.Opener):
    def get_data(self, folders):
        return folders
    def fake_data(self, n_samples=None):
        pass
"""

TEMPLATE_ALGO_SCRIPT = f"""
import json
import substratools as tools


@tools.register
def train(inputs, outputs, task_properties):

    models = []
    for m_path in inputs.get('{InputIdentifiers.models}', []):
        models.append(load_model(m_path))

    # Check that the order of X is the same as the one passed to add_task
    datasample_keys = [d.split("/")[-1] for d in inputs['{InputIdentifiers.datasamples}']]
    assert datasample_keys == {{data_sample_keys}}, datasample_keys

    save_model(([0, 1], [0, 2]), outputs['{OutputIdentifiers.model}'])

@tools.register
def predict(inputs, outputs, task_properties):
    # Check that the order of X is the same as the one passed to add_task
    datasamples = inputs['{InputIdentifiers.datasamples}']
    datasample_keys = [d.split("/")[-1] for d in datasamples]
    model = load_model(inputs['{InputIdentifiers.model}'])
    assert datasample_keys == {{data_sample_keys}}, datasample_keys
    save_predictions(datasamples, outputs['{OutputIdentifiers.predictions}'])

def load_model(path):
    with open(path) as f:
        return json.load(f)

def save_model(model, path):
    with open(path, 'w') as f:
        return json.dump(model, f)

def save_predictions(predictions, path):
    with open(path, 'w') as f:
        return json.dump(predictions, f)

if __name__ == '__main__':
    tools.execute()
"""

TEMPLATE_COMPOSITE_ALGO_SCRIPT = f"""
import json
import substratools as tools

@tools.register
def train(inputs, outputs, task_properties):
    # Check that the order of X is the same as the one passed to add_task

    data_samples = inputs['{InputIdentifiers.datasamples}']

    data_sample_keys = [folder.split('/')[-1] for folder in data_samples]
    assert data_sample_keys == {{data_sample_keys}}, data_sample_keys

    save_head_model([0, 1], outputs['{OutputIdentifiers.local}'])
    save_trunk_model([0, 2], outputs['{OutputIdentifiers.shared}'])

@tools.register
def predict(inputs, outputs, task_properties):
    # Check that the order of X is the same as the one passed to add_task
    data_samples = inputs['{InputIdentifiers.datasamples}']

    data_sample_keys = [folder.split('/')[-1] for folder in data_samples]
    assert data_sample_keys == {{data_sample_keys}}, data_sample_keys

    save_predictions(data_samples, outputs['{OutputIdentifiers.predictions}'])

def load_head_model(path):
    return _load_model(path)

def save_head_model(model, path):
    return _save_model(model, path)

def load_trunk_model(path):
    return _load_model(path)

def save_trunk_model(model, path):
    return _save_model(model, path)

def _load_model(path):
    with open(path) as f:
        return json.load(f)

def _save_model(model, path):
    with open(path, 'w') as f:
        return json.dump(model, f)

def save_predictions(predictions, path):
    with open(path, 'w') as f:
        return json.dump(predictions, f)

if __name__ == '__main__':
    tools.execute()
"""

TEMPLATE_METRIC_SCRIPT = f"""
import substratools as tools

import json

@tools.register
def score(inputs, outputs, task_properties):
    datasamples = inputs['{InputIdentifiers.datasamples}']
    y_pred = _load_predictions(inputs['{InputIdentifiers.predictions}'])
    y_pred_data_sample_keys = [folder.split('/')[-1] for folder in y_pred]
    assert y_pred_data_sample_keys == {{data_sample_keys}}

    y_true_data_sample_keys = [folder.split('/')[-1] for folder in datasamples]
    assert y_true_data_sample_keys == {{data_sample_keys}}

    # y_true is a list of unordered data samples
    # since the Algo returns y==x, y_pred should respect the same order

    assert  y_true_data_sample_keys == y_pred_data_sample_keys, (y_true_data_sample_keys, y_pred_data_sample_keys)

    tools.save_performance(1.0, outputs['{OutputIdentifiers.performance}'])

def _load_predictions(path):
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    tools.execute()
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
            spec = factory.create_data_sample(datasets=[dataset])
            client.add_data_sample(spec)

        # create test data samples
        for _ in range(2):
            spec = factory.create_data_sample(datasets=[dataset])
            client.add_data_sample(spec)

        self.dataset = client.get_dataset(dataset.key)
        self.data_sample_keys = _shuffle(self.dataset.data_sample_keys)
        self.data_inputs = FLTaskInputGenerator.tuple(
            opener_key=dataset.key,
            data_sample_keys=self.data_sample_keys,
        )


@pytest.fixture
def dataset(factory, client):
    return Dataset(factory, client)


def test_task_data_samples_relative_order(factory, client, dataset, worker):

    # Format TEMPLATE_ALGO_SCRIPT with current data_sample_keys
    algo_script = TEMPLATE_ALGO_SCRIPT.format(
        data_sample_keys=dataset.data_sample_keys,
        models=None,
    )
    algo_spec = factory.create_algo(category=AlgoCategory.simple, py_script=algo_script)
    algo = client.add_algo(algo_spec)

    predict_algo_spec = factory.create_algo(category=AlgoCategory.predict, py_script=algo_script)
    predict_algo = client.add_algo(predict_algo_spec)

    metric_script = TEMPLATE_METRIC_SCRIPT.format(data_sample_keys=dataset.data_sample_keys)
    metric_spec = factory.create_algo(category=AlgoCategory.metric, py_script=metric_script)
    metric = client.add_algo(metric_spec)

    traintuple_spec = factory.create_traintuple(algo=algo, inputs=dataset.data_inputs, worker=worker)
    traintuple = client.add_task(traintuple_spec)

    # Ensure the order of the data sample keys is correct at 2 levels: :
    #  1. In the returned traintuple
    #  2. In the train method of the algo. If the order is incorrect, wait() will fail.
    assert [
        i.asset_key for i in traintuple.inputs if i.identifier == InputIdentifiers.datasamples
    ] == dataset.data_sample_keys
    client.wait(traintuple)

    predict_input_models = FLTaskInputGenerator.train_to_predict(traintuple.key)
    predicttuple_spec = factory.create_predicttuple(
        algo=predict_algo, inputs=dataset.data_inputs + predict_input_models, worker=worker
    )
    predicttuple = client.add_task(predicttuple_spec)

    test_input_models = FLTaskInputGenerator.predict_to_test(predicttuple.key)

    testtuple_spec = factory.create_testtuple(
        algo=metric, inputs=dataset.data_inputs + test_input_models, worker=worker
    )
    testtuple = client.add_task(testtuple_spec)

    # Assert order is correct in the metric. If not, wait() will fail.
    client.wait(testtuple)


def test_composite_traintuple_data_samples_relative_order(factory, client, dataset, worker):
    # Format TEMPLATE_COMPOSITE_ALGO_SCRIPT with current data_sample_keys
    composite_algo_script = TEMPLATE_COMPOSITE_ALGO_SCRIPT.format(
        data_sample_keys=dataset.data_sample_keys,
        models=None,
    )
    algo_spec = factory.create_algo(AlgoCategory.composite, py_script=composite_algo_script)
    composite_algo = client.add_algo(algo_spec)

    predict_algo_script = TEMPLATE_COMPOSITE_ALGO_SCRIPT.format(
        data_sample_keys=dataset.data_sample_keys,
        models=None,
    )
    predict_algo_spec = factory.create_algo(AlgoCategory.predict_composite, py_script=predict_algo_script)
    predict_algo = client.add_algo(predict_algo_spec)

    metric_script = TEMPLATE_METRIC_SCRIPT.format(
        data_sample_keys=dataset.data_sample_keys,
    )
    metric_spec = factory.create_algo(category=AlgoCategory.metric, py_script=metric_script)
    metric = client.add_algo(metric_spec)

    traintuple_spec = factory.create_composite_traintuple(
        algo=composite_algo, inputs=dataset.data_inputs, worker=worker
    )
    composite_traintuple = client.add_task(traintuple_spec)
    # Ensure the order of the data sample keys is correct at 2 levels: :
    #  1. In the returned composite traintuple
    #  2. In the train method of the algo. If the order is incorrect, wait() will fail.
    assert [
        i.asset_key for i in composite_traintuple.inputs if i.identifier == InputIdentifiers.datasamples
    ] == dataset.data_sample_keys
    client.wait(composite_traintuple)

    predict_input_models = FLTaskInputGenerator.composite_to_predict(composite_traintuple.key)

    predicttuple_spec = factory.create_predicttuple(
        algo=predict_algo, inputs=dataset.data_inputs + predict_input_models, worker=worker
    )
    predicttuple = client.add_task(predicttuple_spec)

    test_input_models = FLTaskInputGenerator.predict_to_test(predicttuple.key)

    testtuple_spec = factory.create_testtuple(
        algo=metric, inputs=dataset.data_inputs + test_input_models, worker=worker
    )
    testtuple = client.add_task(testtuple_spec)

    # Assert order is correct in the metric. If not, wait() will fail.
    client.wait(testtuple)


@pytest.mark.slow
def test_execution_data_sample_values(factory, client, worker):
    """Check data samples order is preserved when adding data samples by batch."""
    batch_size = 10
    spec = factory.create_dataset(
        py_script=f"""
import json
import os
import substratools as tools
class TestOpener(tools.Opener):
    def get_data(self, folders):
        res = []
        for folder in folders:
            path = os.path.join(folder, '{DEFAULT_DATA_SAMPLE_FILENAME}')
            with open(path, 'r') as f:
                res.append(int(f.read()))
        return res
    def fake_data(self, n_samples=None):
        return
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

@tools.register
def train(inputs, outputs, task_properties):

    datasamples = inputs['{InputIdentifiers.datasamples}']
    assert datasamples == list(range({batch_size})), datasamples
    save_model(0, outputs['{OutputIdentifiers.model}'])

@tools.register
def predict(inputs, outputs, task_properties):
    save_predictions(1, outputs['{OutputIdentifiers.predictions}'])

def load_model(path):
    with open(path) as f:
        return json.load(f)

def save_model(model, path):
    with open(path, 'w') as f:
        return json.dump(model, f)

def save_predictions(predictions, path):
    with open(path, 'w') as f:
        return json.dump(predictions, f)

if __name__ == '__main__':
    tools.execute()
""",
    )
    algo = client.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        inputs=FLTaskInputGenerator.tuple(opener_key=dataset.key, data_sample_keys=keys),
        worker=worker,
    )
    traintuple = client.add_task(spec)
    traintuple = client.wait(traintuple)
    assert traintuple.status == Status.done
