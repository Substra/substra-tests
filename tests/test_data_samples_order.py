import pytest
from substra.sdk.models import Status

import substratest as sbt
from substratest.factory import DEFAULT_DATA_SAMPLE_FILENAME
from substratest.fl_interface import FLTaskInputGenerator
from substratest.fl_interface import FunctionCategory
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

TEMPLATE_FUNCTION_SCRIPT = f"""
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
    assert datasample_keys == {{test_data_sample_keys}}, datasample_keys
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

TEMPLATE_COMPOSITE_FUNCTION_SCRIPT = f"""
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

    test_data_sample_keys = [folder.split('/')[-1] for folder in data_samples]
    assert test_data_sample_keys == {{test_data_sample_keys}}, test_data_sample_keys

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
    # since the Function returns y==x, y_pred should respect the same order

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

        self.dataset = client.get_dataset(dataset.key)
        self.train_data_sample_keys = _shuffle(self.dataset.data_sample_keys)
        self.test_data_sample_keys = self.train_data_sample_keys[:2]
        self.train_data_inputs = FLTaskInputGenerator.task(
            opener_key=dataset.key,
            data_sample_keys=self.train_data_sample_keys,
        )
        self.test_data_inputs = FLTaskInputGenerator.task(
            opener_key=dataset.key,
            data_sample_keys=self.test_data_sample_keys,
        )


@pytest.fixture
def dataset(factory, client):
    return Dataset(factory, client)


def test_task_data_samples_relative_order(factory, client, dataset, worker):
    # Format TEMPLATE_FUNCTION_SCRIPT with current data_sample_keys
    function_script = TEMPLATE_FUNCTION_SCRIPT.format(
        data_sample_keys=dataset.train_data_sample_keys,
        test_data_sample_keys=dataset.test_data_sample_keys,
        models=None,
    )
    function_spec = factory.create_function(category=FunctionCategory.simple, py_script=function_script)
    function = client.add_function(function_spec)

    predict_function_spec = factory.create_function(category=FunctionCategory.predict, py_script=function_script)
    predict_function = client.add_function(predict_function_spec)

    metric_script = TEMPLATE_METRIC_SCRIPT.format(data_sample_keys=dataset.test_data_sample_keys)
    metric_spec = factory.create_function(category=FunctionCategory.metric, py_script=metric_script)
    metric = client.add_function(metric_spec)

    traintask_spec = factory.create_traintask(function=function, inputs=dataset.train_data_inputs, worker=worker)
    traintask = client.add_task(traintask_spec)

    # Ensure the order of the data sample keys is correct at 2 levels: :
    #  1. In the returned traintask
    #  2. In the train method of the function. If the order is incorrect, wait() will fail.
    assert [
        i.asset_key for i in traintask.inputs if i.identifier == InputIdentifiers.datasamples
    ] == dataset.train_data_sample_keys
    client.wait(traintask)

    predict_input_models = FLTaskInputGenerator.train_to_predict(traintask.key)
    predicttask_spec = factory.create_predicttask(
        function=predict_function, inputs=dataset.test_data_inputs + predict_input_models, worker=worker
    )
    predicttask = client.add_task(predicttask_spec)

    test_input_models = FLTaskInputGenerator.predict_to_test(predicttask.key)

    testtask_spec = factory.create_testtask(
        function=metric, inputs=dataset.test_data_inputs + test_input_models, worker=worker
    )
    testtask = client.add_task(testtask_spec)

    # Assert order is correct in the metric. If not, wait() will fail.
    client.wait(testtask)


def test_composite_traintask_data_samples_relative_order(factory, client, dataset, worker):
    # Format TEMPLATE_COMPOSITE_FUNCTION_SCRIPT with current data_sample_keys
    composite_function_script = TEMPLATE_COMPOSITE_FUNCTION_SCRIPT.format(
        data_sample_keys=dataset.train_data_sample_keys,
        test_data_sample_keys=dataset.test_data_sample_keys,
        models=None,
    )
    function_spec = factory.create_function(FunctionCategory.composite, py_script=composite_function_script)
    composite_function = client.add_function(function_spec)

    predict_function_script = TEMPLATE_COMPOSITE_FUNCTION_SCRIPT.format(
        data_sample_keys=dataset.train_data_sample_keys,
        test_data_sample_keys=dataset.test_data_sample_keys,
        models=None,
    )
    predict_function_spec = factory.create_function(
        FunctionCategory.predict_composite, py_script=predict_function_script
    )
    predict_function = client.add_function(predict_function_spec)

    metric_script = TEMPLATE_METRIC_SCRIPT.format(
        data_sample_keys=dataset.test_data_sample_keys,
    )
    metric_spec = factory.create_function(category=FunctionCategory.metric, py_script=metric_script)
    metric = client.add_function(metric_spec)

    traintask_spec = factory.create_composite_traintask(
        function=composite_function, inputs=dataset.train_data_inputs, worker=worker
    )
    composite_traintask = client.add_task(traintask_spec)
    # Ensure the order of the data sample keys is correct at 2 levels: :
    #  1. In the returned composite traintask
    #  2. In the train method of the function. If the order is incorrect, wait() will fail.
    assert [
        i.asset_key for i in composite_traintask.inputs if i.identifier == InputIdentifiers.datasamples
    ] == dataset.train_data_sample_keys
    client.wait(composite_traintask)

    predict_input_models = FLTaskInputGenerator.composite_to_predict(composite_traintask.key)

    predicttask_spec = factory.create_predicttask(
        function=predict_function, inputs=dataset.test_data_inputs + predict_input_models, worker=worker
    )
    predicttask = client.add_task(predicttask_spec)

    test_input_models = FLTaskInputGenerator.predict_to_test(predicttask.key)

    testtask_spec = factory.create_testtask(
        function=metric, inputs=dataset.test_data_inputs + test_input_models, worker=worker
    )
    testtask = client.add_task(testtask_spec)

    # Assert order is correct in the metric. If not, wait() will fail.
    client.wait(testtask)


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

    spec = factory.create_function(
        category=FunctionCategory.simple,
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
    function = client.add_function(spec)

    spec = factory.create_traintask(
        function=function,
        inputs=FLTaskInputGenerator.task(opener_key=dataset.key, data_sample_keys=keys),
        worker=worker,
    )
    traintask = client.add_task(spec)
    traintask = client.wait(traintask)
    assert traintask.status == Status.done
