import abc
import os
import pathlib
import shutil
import string
import tempfile
import typing
import uuid

import pydantic
import substra
from substra.sdk.schemas import ComputePlanTaskSpec
from substra.sdk.schemas import FunctionSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import TaskSpec
from substra.sdk.schemas import UpdateComputePlanSpec
from substra.sdk.schemas import UpdateDatasetSpec
from substra.sdk.schemas import UpdateFunctionSpec

from substratest.fl_interface import FLFunctionInputs
from substratest.fl_interface import FLFunctionOutputs
from substratest.fl_interface import FLTaskInputGenerator
from substratest.fl_interface import FLTaskOutputGenerator
from substratest.fl_interface import FunctionCategory
from substratest.fl_interface import InputIdentifiers
from substratest.fl_interface import OutputIdentifiers

from . import utils
from .settings import PytestConfig

DEFAULT_DATA_SAMPLE_FILENAME = "data.csv"

DEFAULT_OPENER_SCRIPT = f"""
import csv
import json
import os
import substratools as tools
class TestOpener(tools.Opener):
    def get_data(self, folders):
        X, y = [], []
        for folder in folders:
            with open(os.path.join(folder, '{DEFAULT_DATA_SAMPLE_FILENAME}'), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    X.append(int(row[0]))
                    y.append(int(row[1]))

        print(f'X: {{X}}') # returns a list of 1's
        print(f'y: {{y}}') # returns a list of 2's
        return X, y

    def fake_data(self, n_samples=None):
        if n_samples is None:
            n_samples = 1
        X = [10] * n_samples
        y = [30] * n_samples
        print(f'fake X: {{X}}')
        print(f'fake y: {{y}}')
        return X, y
"""

TEMPLATED_DEFAULT_METRICS_SCRIPT = string.Template(
    f"""
import json
import substratools as tools

@tools.register
def score(inputs, outputs, task_properties):
    y_true = inputs['{InputIdentifiers.datasamples.value}'][1]
    y_pred = _load_predictions(inputs['{InputIdentifiers.predictions.value}'])
    res = sum(y_pred) - sum(y_true)
    print(f'metrics, y_true: {{y_true}}, y_pred: {{y_pred}}, result: {{res}}')
    tools.save_performance(res + $offset, outputs['{OutputIdentifiers.performance.value}'])

def _load_predictions(path):
    with open(path) as f:
        return json.load(f)

if __name__ == '__main__':
    tools.execute()
"""
)  # noqa

DEFAULT_FUNCTION_SCRIPT = f"""
import json
import substratools as tools

@tools.register
def train(inputs, outputs, task_properties):

    X = inputs['{InputIdentifiers.datasamples.value}'][0]
    y = inputs['{InputIdentifiers.datasamples.value}'][1]
    rank = task_properties['{InputIdentifiers.rank.value}']

    models = []
    for m_path in inputs.get('{InputIdentifiers.shared.value}', []):
        models.append(load_model(m_path))

    print(f'Train, get X: {{X}}, y: {{y}}, models: {{models}}')

    ratio = sum(y) / sum(X)
    err = 0.1 * ratio  # Add a small error

    if len(models) == 0:
        res = {{'value': ratio + err }}
    else:
        ratios = [m['value'] for m in models]
        avg = sum(ratios) / len(ratios)
        res = {{'value': avg + err }}

    print(f'Train, return {{res}}')
    save_model(res, outputs['{OutputIdentifiers.shared.value}'])

@tools.register
def predict(inputs, outputs, task_properties):
    X = inputs['{InputIdentifiers.datasamples.value}'][0]
    model = load_model(inputs['{InputIdentifiers.shared.value}'])

    res = [x * model['value'] for x in X]
    print(f'Predict, get X: {{X}}, model: {{model}}, return {{res}}')
    save_predictions(res, outputs['{OutputIdentifiers.predictions.value}'])

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
"""  # noqa

DEFAULT_AGGREGATE_FUNCTION_SCRIPT = f"""
import json
import substratools as tools

@tools.register
def aggregate(inputs, outputs, task_properties):
    rank = task_properties['{InputIdentifiers.rank.value}']
    models = []
    for m_path in inputs['{InputIdentifiers.shared.value}']:
        models.append(load_model(m_path))

    print(f'Aggregate models: {{models}}')
    values = [m['value'] for m in models]
    avg = sum(values) / len(values)
    res = {{'value': avg}}
    print(f'Aggregate result: {{res}}')
    save_model(res, outputs['{OutputIdentifiers.shared.value}'])

@tools.register
def predict(inputs, outputs, task_properties):
    X = inputs['{InputIdentifiers.datasamples.value}'][0]
    model = load_model(inputs['{InputIdentifiers.shared.value}'])

    res = [x * model['value'] for x in X]
    print(f'Predict, get X: {{X}}, model: {{model}}, return {{res}}')
    save_predictions(res, outputs['{OutputIdentifiers.predictions.value}'])

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
"""  # noqa

# TODO we should have a different serializer for head and trunk models

DEFAULT_COMPOSITE_FUNCTION_SCRIPT = f"""
import json
import substratools as tools

@tools.register
def train(inputs, outputs, task_properties):
    X = inputs['{InputIdentifiers.datasamples.value}'][0]
    y = inputs['{InputIdentifiers.datasamples.value}'][1]
    rank = task_properties['{InputIdentifiers.rank.value}']
    head_model = load_head_model(inputs['{InputIdentifiers.local.value}']) if inputs.get('{InputIdentifiers.local.value}') else None
    trunk_model = load_trunk_model(inputs['{InputIdentifiers.shared.value}']) if inputs.get('{InputIdentifiers.shared.value}') else None


    print(f'Composite function train X: {{X}}, y: {{y}}, head_model: {{head_model}}, trunk_model: {{trunk_model}}')

    ratio = sum(y) / sum(X)
    err_head = 0.1 * ratio  # Add a small error
    err_trunk = 0.2 * ratio  # Add a small error

    if head_model:
        res_head = head_model['value']
    else:
        res_head = ratio

    if trunk_model:
        res_trunk = trunk_model['value']
    else:
        res_trunk = ratio

    res_head_model = {{'value' : res_head + err_head }}
    res_trunk_model =  {{'value' : res_trunk + err_trunk }}

    print(f'Composite function train head, trunk result: {{res_head_model}}, {{res_trunk_model}}')
    save_head_model(res_head_model, outputs['{OutputIdentifiers.local.value}'])
    save_trunk_model(res_trunk_model, outputs['{OutputIdentifiers.shared.value}'])

@tools.register
def predict(inputs, outputs, task_properties):
    X = inputs['{InputIdentifiers.datasamples.value}'][0]
    head_model = load_head_model(inputs['{InputIdentifiers.local.value}'])
    trunk_model = load_trunk_model(inputs['{InputIdentifiers.shared.value}'])

    print(f'Composite function predict X: {{X}}, head_model: {{head_model}}, trunk_model: {{trunk_model}}')
    ratio_sum = head_model['value'] + trunk_model['value']
    res = [x * ratio_sum for x in X]
    print(f'Composite function predict result: {{res}}')

    save_predictions(res, outputs['{OutputIdentifiers.predictions.value}'])

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
"""  # noqa

DEFAULT_FUNCTION_NAME = {
    FunctionCategory.simple: "train",
    FunctionCategory.composite: "train",
    FunctionCategory.aggregate: "aggregate",
    FunctionCategory.predict: "predict",
    FunctionCategory.metric: "score",
    FunctionCategory.predict_composite: "predict",
}

INVALID_FUNCTION_SCRIPT = DEFAULT_FUNCTION_SCRIPT.replace("train", "naitr")
INVALID_COMPOSITE_FUNCTION_SCRIPT = DEFAULT_COMPOSITE_FUNCTION_SCRIPT.replace("train", "naitr")
INVALID_AGGREGATE_FUNCTION_SCRIPT = DEFAULT_AGGREGATE_FUNCTION_SCRIPT.replace("aggregate", "etagergga")


def random_uuid():
    return str(uuid.uuid4())


def _shorten_name(name):
    """Format asset name to ensure they match the backend requirements."""
    if len(name) < 100:
        return name
    return name[:75] + "..." + name[:20]


class Counter:
    def __init__(self):
        self._idx = -1

    def inc(self):
        self._idx += 1
        return self._idx


class _Spec(pydantic.BaseModel, abc.ABC):
    """Asset specification base class."""


class PrivatePermissions(pydantic.BaseModel):
    public: bool
    authorized_ids: typing.List[str]


DEFAULT_PERMISSIONS = Permissions(public=True, authorized_ids=[])
DEFAULT_OUT_TRUNK_MODEL_PERMISSIONS = PrivatePermissions(public=False, authorized_ids=[])
SERVER_MEDIA_PATH = "/var/substra/servermedias/"


class DataSampleSpec(_Spec):
    path: str
    data_manager_keys: typing.List[str]

    def move_data_to_server(self, destination_folder, minikube=False):
        destination_folder = destination_folder if destination_folder.endswith("/") else destination_folder + "/"

        if not minikube:
            destination = tempfile.mkdtemp(dir=destination_folder)
            shutil.move(self.path, destination)
            os.chmod(destination, 0o777)  # Workaround for kind (https://kind.sigs.k8s.io/)
        else:
            destination = os.path.join(destination_folder, random_uuid()[0:8])

            minikube_private_key = "~/.minikube/machines/minikube/id_rsa"
            minikube_ssh = "docker@$(minikube ip)"

            os.system(f"scp -r -i {minikube_private_key} {self.path} {minikube_ssh}:/tmp/")
            os.system(
                f"ssh -i {minikube_private_key} -oStrictHostKeyChecking=no {minikube_ssh} "
                f'"sudo mkdir -p {destination} && sudo mv /tmp/{os.path.basename(self.path)} {destination}"'
            )

            # Clean path after copy
            shutil.rmtree(self.path)

        self.path = os.path.join(
            destination.replace(destination_folder, SERVER_MEDIA_PATH), os.path.basename(self.path)
        )


class DataSampleBatchSpec(_Spec):
    paths: typing.List[str]
    data_manager_keys: typing.List[str]

    @classmethod
    def from_data_sample_specs(cls, specs: typing.List[DataSampleSpec]):
        def _all_equal(iterator):
            iterator = iter(iterator)
            first = next(iterator)
            return all(first == x for x in iterator)

        assert len(specs)
        assert _all_equal([s.data_manager_keys for s in specs])

        return cls(
            paths=[s.path for s in specs],
            data_manager_keys=specs[0].data_manager_keys,
        )


class DatasetSpec(substra.sdk.schemas.DatasetSpec):
    def read_opener(self):
        with open(self.data_opener, "rb") as f:
            return f.read()

    def read_description(self):
        with open(self.description, "r") as f:
            return f.read()


DEFAULT_FUNCTION_SCRIPTS = {
    FunctionCategory.simple: DEFAULT_FUNCTION_SCRIPT,
    FunctionCategory.composite: DEFAULT_COMPOSITE_FUNCTION_SCRIPT,
    FunctionCategory.aggregate: DEFAULT_AGGREGATE_FUNCTION_SCRIPT,
    FunctionCategory.predict: DEFAULT_FUNCTION_SCRIPT,
    FunctionCategory.predict_composite: DEFAULT_COMPOSITE_FUNCTION_SCRIPT,
}


class AugmentedDataset:
    def __init__(self, dataset) -> None:
        """Augment a dataset to create train and test datasamples.

        Args:
            dataset: dataset link to the datasamples.
        """
        self.key = dataset.key
        self.owner = dataset.owner
        self.data_sample_keys = dataset.data_sample_keys
        self.opener_input = FLTaskInputGenerator.opener(dataset.key)

    def set_train_test_dasamples(self, train_data_sample_keys=(), test_data_sample_keys=()):
        self._check_data_sample_keys(train_data_sample_keys)
        self._check_data_sample_keys(test_data_sample_keys)

        self.train_data_sample_keys = train_data_sample_keys
        self.test_data_sample_keys = test_data_sample_keys

        self.train_data_sample_inputs = FLTaskInputGenerator.data_samples(train_data_sample_keys)
        self.test_data_sample_inputs = FLTaskInputGenerator.data_samples(test_data_sample_keys)

        self.train_data_inputs = FLTaskInputGenerator.task(
            opener_key=self.key,
            data_sample_keys=train_data_sample_keys,
        )
        self.test_data_inputs = FLTaskInputGenerator.task(
            opener_key=self.key,
            data_sample_keys=test_data_sample_keys,
        )

    def _check_data_sample_keys(self, data_sample_keys):
        for data_sample_key in data_sample_keys:
            if data_sample_key not in self.data_sample_keys:
                raise Exception(f"{data_sample_key} not in the dataset data samples.")


class _ComputePlanSpecFactory:
    def create_traintask(
        self, function, worker, inputs=None, outputs=None, tag="", metadata=None
    ) -> ComputePlanTaskSpec:
        spec = ComputePlanTaskSpec(
            function_key=function.key,
            task_id=random_uuid(),
            inputs=inputs or [],
            outputs=outputs if outputs is not None else FLTaskOutputGenerator.traintask(),
            tag=tag,
            metadata=metadata,
            worker=worker,
        )
        self.tasks.append(spec)
        return spec

    def create_aggregatetask(
        self, aggregate_function, worker, inputs=None, outputs=None, tag="", metadata=None
    ) -> ComputePlanTaskSpec:
        spec = ComputePlanTaskSpec(
            task_id=random_uuid(),
            function_key=aggregate_function.key,
            worker=worker,
            inputs=inputs or [],
            outputs=outputs if outputs is not None else FLTaskOutputGenerator.aggregatetask(),
            tag=tag,
            metadata=metadata,
        )
        self.tasks.append(spec)
        return spec

    def create_composite_traintask(
        self,
        composite_function,
        worker,
        inputs=None,
        outputs=None,
        tag="",
        metadata=None,
    ) -> ComputePlanTaskSpec:
        spec = ComputePlanTaskSpec(
            task_id=random_uuid(),
            function_key=composite_function.key,
            inputs=inputs or [],
            outputs=outputs if outputs is not None else FLTaskOutputGenerator.composite_traintask(),
            tag=tag,
            metadata=metadata,
            worker=worker,
        )
        self.tasks.append(spec)
        return spec

    def create_predicttask(
        self, function, worker, inputs=None, outputs=None, tag="", metadata=None
    ) -> ComputePlanTaskSpec:
        spec = ComputePlanTaskSpec(
            task_id=random_uuid(),
            function_key=function.key,
            inputs=inputs or [],
            outputs=outputs if outputs is not None else FLTaskOutputGenerator.predicttask(),
            tag=tag,
            metadata=metadata,
            worker=worker,
        )
        self.tasks.append(spec)
        return spec

    def create_testtask(
        self, function, worker, inputs=None, outputs=None, tag="", metadata=None
    ) -> ComputePlanTaskSpec:
        spec = ComputePlanTaskSpec(
            task_id=random_uuid(),
            function_key=function.key,
            inputs=inputs or [],
            outputs=outputs if outputs is not None else FLTaskOutputGenerator.testtask(),
            tag=tag,
            metadata=metadata,
            worker=worker,
        )
        self.tasks.append(spec)
        return spec


class ComputePlanSpec(_ComputePlanSpecFactory, substra.sdk.schemas.ComputePlanSpec):
    pass


class UpdateComputePlanTasksSpec(_ComputePlanSpecFactory, substra.sdk.schemas.UpdateComputePlanTasksSpec):
    pass


class AssetsFactory:
    def __init__(self, name, cfg: PytestConfig, client_debug_local=False):
        self._data_sample_counter = Counter()
        self._dataset_counter = Counter()
        self._metric_counter = Counter()
        self._function_counter = Counter()
        self._workdir = pathlib.Path(tempfile.mkdtemp(prefix="/tmp/"))
        self._uuid = name
        self._cfg = cfg
        self._client_debug_local = client_debug_local

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        shutil.rmtree(str(self._workdir), ignore_errors=True)

    @property
    def default_tools_image(self):
        return self._cfg.substra_tools.image_local if self._client_debug_local else self._cfg.substra_tools.image_remote

    # We need to adapt the image base name base on the fact that
    # we run the cp in the docker context (debug)
    # or the kaniko pod (remote) to be able to pull the image
    def default_function_dockerfile(self, method_name):
        return (
            f"FROM {self.default_tools_image}\nCOPY function.py .\n"
            f'ENTRYPOINT ["python3", "function.py", "--function-name", "{method_name}"]\n'
        )

    def create_data_sample(self, content=None, datasets=None):
        idx = self._data_sample_counter.inc()
        tmpdir = self._workdir / f"data-{idx}"
        tmpdir.mkdir()

        content = content or "10,20"
        content = content.encode("utf-8")

        data_filepath = tmpdir / DEFAULT_DATA_SAMPLE_FILENAME
        with open(data_filepath, "wb") as f:
            f.write(content)

        datasets = datasets or []

        return DataSampleSpec(
            path=str(tmpdir),
            data_manager_keys=[d.key for d in datasets],
        )

    def create_dataset(self, permissions=None, metadata=None, py_script=None, logs_permission=None):
        idx = self._dataset_counter.inc()
        tmpdir = self._workdir / f"dataset-{idx}"
        tmpdir.mkdir()
        name = _shorten_name(f"{self._uuid} - Dataset {idx}")

        description_path = tmpdir / "description.md"
        description_content = name
        with open(description_path, "w") as f:
            f.write(description_content)

        opener_path = tmpdir / "opener.py"
        with open(opener_path, "w") as f:
            f.write(py_script or DEFAULT_OPENER_SCRIPT)

        return DatasetSpec(
            name=name,
            data_opener=str(opener_path),
            metadata=metadata,
            description=str(description_path),
            permissions=permissions or DEFAULT_PERMISSIONS,
            logs_permission=logs_permission or DEFAULT_PERMISSIONS,
        )

    def create_function(
        self, category, py_script=None, dockerfile=None, permissions=None, metadata=None, offset=0
    ) -> FunctionSpec:
        idx = self._function_counter.inc()
        tmpdir = self._workdir / f"function-{idx}"
        tmpdir.mkdir()
        name = _shorten_name(f"{self._uuid} - Function {idx}")

        description_path = tmpdir / "description.md"
        description_content = name
        with open(description_path, "w") as f:
            f.write(description_content)

        try:
            if category == FunctionCategory.metric:
                function_content = py_script or TEMPLATED_DEFAULT_METRICS_SCRIPT.substitute(offset=offset)
            else:
                function_content = py_script or DEFAULT_FUNCTION_SCRIPTS[category]
        except KeyError:
            raise Exception("Invalid function category", category)

        dockerfile = dockerfile or self.default_function_dockerfile(method_name=DEFAULT_FUNCTION_NAME[category])

        function_zip = utils.create_archive(
            tmpdir / "function",
            ("function.py", function_content),
            ("Dockerfile", dockerfile),
        )

        return FunctionSpec(
            inputs=FLFunctionInputs[category],
            outputs=FLFunctionOutputs[category],
            name=name,
            description=str(description_path),
            file=str(function_zip),
            permissions=permissions or DEFAULT_PERMISSIONS,
            metadata=metadata,
        )

    def create_traintask(
        self,
        function=None,
        inputs=None,
        outputs=None,
        tag=None,
        compute_plan_key=None,
        rank=None,
        metadata=None,
        worker=None,
    ) -> TaskSpec:
        return TaskSpec(
            function_key=function.key if function else None,
            tag=tag,
            metadata=metadata,
            compute_plan_key=compute_plan_key,
            inputs=inputs or [],
            outputs=outputs if outputs is not None else FLTaskOutputGenerator.traintask(),
            rank=rank,
            worker=worker,
        )

    def create_aggregatetask(
        self,
        function=None,
        worker=None,
        inputs=None,
        outputs=None,
        tag=None,
        compute_plan_key=None,
        rank=None,
        metadata=None,
    ) -> TaskSpec:
        return TaskSpec(
            function_key=function.key if function else None,
            worker=worker,
            inputs=inputs or [],
            outputs=outputs if outputs is not None else FLTaskOutputGenerator.aggregatetask(),
            tag=tag,
            metadata=metadata,
            compute_plan_key=compute_plan_key,
            rank=rank,
        )

    def create_composite_traintask(
        self,
        function=None,
        inputs=None,
        outputs=None,
        tag=None,
        compute_plan_key=None,
        rank=None,
        metadata=None,
        worker=None,
    ) -> TaskSpec:
        return TaskSpec(
            function_key=function.key if function else None,
            inputs=inputs or [],
            outputs=outputs if outputs is not None else FLTaskOutputGenerator.composite_traintask(),
            tag=tag,
            metadata=metadata,
            compute_plan_key=compute_plan_key,
            rank=rank,
            worker=worker,
        )

    def create_predicttask(self, function, worker, inputs=None, outputs=None, tag=None, metadata=None) -> TaskSpec:
        return TaskSpec(
            function_key=function.key,
            inputs=inputs or [],
            outputs=outputs if outputs is not None else FLTaskOutputGenerator.predicttask(),
            tag=tag,
            metadata=metadata,
            worker=worker,
        )

    def create_testtask(self, function, worker, inputs=None, outputs=None, tag=None, metadata=None) -> TaskSpec:
        return TaskSpec(
            function_key=function.key,
            inputs=inputs or [],
            outputs=outputs if outputs is not None else FLTaskOutputGenerator.testtask(),
            tag=tag,
            metadata=metadata,
            worker=worker,
        )

    def create_compute_plan(self, key=None, tag="", name="Test compute plan", clean_models=False, metadata=None):
        return ComputePlanSpec(
            key=key or random_uuid(),
            tasks=[],
            tag=tag,
            name=name,
            metadata=metadata,
            clean_models=clean_models,
        )

    def add_compute_plan_tasks(self, compute_plan):
        return UpdateComputePlanTasksSpec(
            tasks=[],
            key=compute_plan.key,
            name=compute_plan.name,
        )

    def update_function(self, name):
        return UpdateFunctionSpec(name=name)

    def update_compute_plan(self, name):
        return UpdateComputePlanSpec(name=name)

    def update_dataset(self, name):
        return UpdateDatasetSpec(name=name)
