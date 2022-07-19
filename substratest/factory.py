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
from substra.sdk import models
from substra.sdk.schemas import AggregatetupleSpec
from substra.sdk.schemas import AlgoCategory
from substra.sdk.schemas import AlgoSpec
from substra.sdk.schemas import CompositeTraintupleSpec
from substra.sdk.schemas import ComputePlanPredicttupleSpec
from substra.sdk.schemas import ComputePlanTesttupleSpec
from substra.sdk.schemas import ComputeTaskOutput
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import PredicttupleSpec
from substra.sdk.schemas import TesttupleSpec
from substra.sdk.schemas import TraintupleSpec

from . import utils
from .settings import Settings

DEFAULT_DATA_SAMPLE_FILENAME = "data.csv"

DEFAULT_OPENER_SCRIPT = f"""
import csv
import json
import os
import substratools as tools
class TestOpener(tools.Opener):
    def get_X(self, folders):
        res = []
        for folder in folders:
            with open(os.path.join(folder, '{DEFAULT_DATA_SAMPLE_FILENAME}'), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    res.append(int(row[0]))
        print(f'get_X: {{res}}')
        return res  # returns a list of 1's
    def get_y(self, folders):
        res = []
        for folder in folders:
            with open(os.path.join(folder, '{DEFAULT_DATA_SAMPLE_FILENAME}'), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    res.append(int(row[1]))
        print(f'get_y: {{res}}')
        return res  # returns a list of 2's
    def fake_X(self, n_samples=None):
        if n_samples is None:
            n_samples = 1
        res = [10] * n_samples
        print(f'fake_X: {{res}}')
        return res
    def fake_y(self, n_samples=None):
        if n_samples is None:
            n_samples = 1
        res = [30] * n_samples
        print(f'fake_y: {{res}}')
        return res
    def get_predictions(self, path):
        with open(path) as f:
            return json.load(f)
    def save_predictions(self, y_pred, path):
        with open(path, 'w') as f:
            return json.dump(y_pred, f)
"""

TEMPLATED_DEFAULT_METRICS_SCRIPT = string.Template(
    """
import json
import substratools as tools
class TestMetrics(tools.Metrics):
    def score(self, y_true, y_pred):
        res = sum(y_pred) - sum(y_true)
        print(f'metrics, y_true: {{y_true}}, y_pred: {{y_pred}}, result: {{res}}')
        return res + $offset
if __name__ == '__main__':
    tools.metrics.execute(TestMetrics())
"""
)  # noqa

DEFAULT_ALGO_SCRIPT = f"""
import json
import substratools as tools
class TestAlgo(tools.Algo):
    def train(self, X, y, models, rank):
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
        return res

    def predict(self, X, model):
        res = [x * model['value'] for x in X]
        print(f'Predict, get X: {{X}}, model: {{model}}, return {{res}}')
        return res

    def load_model(self, path):
        with open(path) as f:
            return json.load(f)

    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)

if __name__ == '__main__':
    tools.algo.execute(TestAlgo())
"""  # noqa

DEFAULT_AGGREGATE_ALGO_SCRIPT = f"""
import json
import substratools as tools
class TestAggregateAlgo(tools.AggregateAlgo):
    def aggregate(self, models, rank):
        print(f'Aggregate models: {{models}}')
        values = [m['value'] for m in models]
        avg = sum(values) / len(values)
        res = {{'value': avg}}
        print(f'Aggregate result: {{res}}')
        return res
    def predict(self, X, model):
        res = [x * model['value'] for x in X]
        print(f'Predict, get X: {{X}}, model: {{model}}, return {{res}}')
        return res
    def load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)
if __name__ == '__main__':
    tools.algo.execute(TestAggregateAlgo())
"""  # noqa

# TODO we should have a different serializer for head and trunk models

DEFAULT_COMPOSITE_ALGO_SCRIPT = f"""
import json
import substratools as tools
class TestCompositeAlgo(tools.CompositeAlgo):
    def train(self, X, y, head_model, trunk_model, rank):

        print(f'Composite algo train X: {{X}}, y: {{y}}, head_model: {{head_model}}, trunk_model: {{trunk_model}}')

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

        res = {{'value' : res_head + err_head }}, {{'value' : res_trunk + err_trunk }}
        print(f'Composite algo train head, trunk result: {{res}}')
        return res

    def predict(self, X, head_model, trunk_model):
        print(f'Composite algo predict X: {{X}}, head_model: {{head_model}}, trunk_model: {{trunk_model}}')
        ratio_sum = head_model['value'] + trunk_model['value']
        res = [x * ratio_sum for x in X]
        print(f'Composite algo predict result: {{res}}')
        return res

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
"""  # noqa

INVALID_ALGO_SCRIPT = DEFAULT_ALGO_SCRIPT.replace("train", "naitr")
INVALID_COMPOSITE_ALGO_SCRIPT = DEFAULT_COMPOSITE_ALGO_SCRIPT.replace("train", "naitr")
INVALID_AGGREGATE_ALGO_SCRIPT = DEFAULT_AGGREGATE_ALGO_SCRIPT.replace("aggregate", "etagergga")

DEFAULT_TRAINTUPLE_OUTPUTS = {"model": ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[]))}
DEFAULT_AGGREGATETUPLE_OUTPUTS = {"model": ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[]))}
DEFAULT_PREDICTTUPLE_OUTPUTS = {
    "predictions": ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[]))
}
DEFAULT_TESTTUPLE_OUTPUTS = {"performance": ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[]))}
DEFAULT_COMPOSITE_TRAINTUPLE_OUTPUTS = {
    "shared": ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[])),
    "local": ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[])),
}


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
    test_only: bool
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
    test_only: bool
    data_manager_keys: typing.List[str]

    @classmethod
    def from_data_sample_specs(cls, specs: typing.List[DataSampleSpec]):
        def _all_equal(iterator):
            iterator = iter(iterator)
            first = next(iterator)
            return all(first == x for x in iterator)

        assert len(specs)
        assert _all_equal([s.test_only for s in specs])
        assert _all_equal([s.data_manager_keys for s in specs])

        return cls(
            paths=[s.path for s in specs],
            test_only=specs[0].test_only,
            data_manager_keys=specs[0].data_manager_keys,
        )


class DatasetSpec(substra.sdk.schemas.DatasetSpec):
    def read_opener(self):
        with open(self.data_opener, "rb") as f:
            return f.read()

    def read_description(self):
        with open(self.description, "r") as f:
            return f.read()


DEFAULT_ALGO_SCRIPTS = {
    AlgoCategory.simple: DEFAULT_ALGO_SCRIPT,
    AlgoCategory.composite: DEFAULT_COMPOSITE_ALGO_SCRIPT,
    AlgoCategory.aggregate: DEFAULT_AGGREGATE_ALGO_SCRIPT,
    AlgoCategory.predict: DEFAULT_ALGO_SCRIPT,
}


class ComputePlanTraintupleSpec(substra.sdk.schemas.ComputePlanTraintupleSpec):
    @property
    def id(self):
        return self.traintuple_id


class ComputePlanAggregatetupleSpec(substra.sdk.schemas.ComputePlanAggregatetupleSpec):
    @property
    def id(self):
        return self.aggregatetuple_id


class ComputePlanCompositeTraintupleSpec(substra.sdk.schemas.ComputePlanCompositeTraintupleSpec):
    @property
    def id(self):
        return self.composite_traintuple_id


def _get_key(obj, field="key"):
    """Get key from asset/spec or key."""
    if isinstance(obj, str):
        return obj
    return getattr(obj, field)


def _get_keys(obj, field="key"):
    """Get keys from asset/spec or key.

    This is particularly useful for data samples to accept as input args a list of keys
    and a list of data samples.
    """
    if not obj:
        return []
    return [_get_key(x, field=field) for x in obj]


class _ComputePlanSpecFactory:
    def create_traintuple(
        self, algo, dataset, data_samples, in_models=None, outputs=None, tag="", metadata=None
    ) -> ComputePlanTraintupleSpec:
        in_models = in_models or []
        spec = ComputePlanTraintupleSpec(
            algo_key=algo.key,
            traintuple_id=random_uuid(),
            data_manager_key=dataset.key,
            train_data_sample_keys=_get_keys(data_samples),
            in_models_ids=[t.id for t in in_models],
            outputs=outputs if outputs is not None else DEFAULT_TRAINTUPLE_OUTPUTS,
            tag=tag,
            metadata=metadata,
        )
        self.traintuples.append(spec)
        return spec

    def create_aggregatetuple(
        self, aggregate_algo, worker, in_models=None, outputs=None, tag="", metadata=None
    ) -> ComputePlanAggregatetupleSpec:
        in_models = in_models or []

        for t in in_models:
            assert isinstance(t, (ComputePlanTraintupleSpec, ComputePlanCompositeTraintupleSpec))

        spec = ComputePlanAggregatetupleSpec(
            aggregatetuple_id=random_uuid(),
            algo_key=aggregate_algo.key,
            worker=worker,
            in_models_ids=[t.id for t in in_models],
            outputs=outputs if outputs is not None else DEFAULT_AGGREGATETUPLE_OUTPUTS,
            tag=tag,
            metadata=metadata,
        )
        self.aggregatetuples.append(spec)
        return spec

    def create_composite_traintuple(
        self,
        composite_algo,
        dataset=None,
        data_samples=None,
        in_head_model=None,
        in_trunk_model=None,
        outputs=None,
        tag="",
        metadata=None,
    ) -> ComputePlanCompositeTraintupleSpec:
        data_samples = data_samples or []

        if in_head_model and in_trunk_model:
            assert isinstance(in_head_model, ComputePlanCompositeTraintupleSpec)
            assert isinstance(in_trunk_model, (ComputePlanCompositeTraintupleSpec, ComputePlanAggregatetupleSpec))

        spec = ComputePlanCompositeTraintupleSpec(
            composite_traintuple_id=random_uuid(),
            algo_key=composite_algo.key,
            data_manager_key=dataset.key if dataset else None,
            train_data_sample_keys=_get_keys(data_samples),
            in_head_model_id=in_head_model.id if in_head_model else None,
            in_trunk_model_id=in_trunk_model.id if in_trunk_model else None,
            outputs=outputs if outputs is not None else DEFAULT_COMPOSITE_TRAINTUPLE_OUTPUTS,
            tag=tag,
            metadata=metadata,
        )
        self.composite_traintuples.append(spec)
        return spec

    def create_predicttuple(
        self, algo, traintuple_spec, dataset, data_samples, outputs=None, tag="", metadata=None
    ) -> ComputePlanPredicttupleSpec:
        spec = ComputePlanPredicttupleSpec(
            predicttuple_id=random_uuid(),
            algo_key=algo.key,
            traintuple_id=traintuple_spec.id,
            data_manager_key=dataset.key,
            test_data_sample_keys=_get_keys(data_samples),
            outputs=outputs if outputs is not None else DEFAULT_PREDICTTUPLE_OUTPUTS,
            tag=tag,
            metadata=metadata,
        )
        self.predicttuples.append(spec)
        return spec

    def create_testtuple(
        self, algo, predicttuple_spec, dataset, data_samples, outputs=None, tag="", metadata=None
    ) -> ComputePlanTesttupleSpec:
        spec = ComputePlanTesttupleSpec(
            algo_key=algo.key,
            predicttuple_id=predicttuple_spec.predicttuple_id,
            data_manager_key=dataset.key,
            test_data_sample_keys=_get_keys(data_samples),
            outputs=outputs if outputs is not None else DEFAULT_TESTTUPLE_OUTPUTS,
            tag=tag,
            metadata=metadata,
        )
        self.testtuples.append(spec)
        return spec


class ComputePlanSpec(_ComputePlanSpecFactory, substra.sdk.schemas.ComputePlanSpec):
    pass


class UpdateComputePlanSpec(_ComputePlanSpecFactory, substra.sdk.schemas.UpdateComputePlanSpec):
    pass


class AssetsFactory:
    def __init__(self, name, cfg: Settings, client_debug_local=False):
        self._data_sample_counter = Counter()
        self._dataset_counter = Counter()
        self._metric_counter = Counter()
        self._algo_counter = Counter()
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
        return self._cfg.connect_tools.image_local if self._client_debug_local else self._cfg.connect_tools.image_remote

    # We need to adapt the image base name base on the fact that we run the cp in the docker context (debug)
    # or the kaniko pod (remote) to be able to pull the image
    @property
    def default_algo_dockerfile(self):
        return f'FROM {self.default_tools_image}\nCOPY algo.py .\nENTRYPOINT ["python3", "algo.py"]\n'

    def create_data_sample(self, content=None, datasets=None, test_only=False):
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
            test_only=test_only,
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
            type="Test",
            metadata=metadata,
            description=str(description_path),
            permissions=permissions or DEFAULT_PERMISSIONS,
            logs_permission=logs_permission or DEFAULT_PERMISSIONS,
        )

    def create_algo(
        self, category, py_script=None, dockerfile=None, permissions=None, metadata=None, offset=0
    ) -> AlgoSpec:
        idx = self._algo_counter.inc()
        tmpdir = self._workdir / f"algo-{idx}"
        tmpdir.mkdir()
        name = _shorten_name(f"{self._uuid} - Algo {idx}")

        description_path = tmpdir / "description.md"
        description_content = name
        with open(description_path, "w") as f:
            f.write(description_content)

        try:
            if category == AlgoCategory.metric:
                algo_content = py_script or TEMPLATED_DEFAULT_METRICS_SCRIPT.substitute(offset=offset)
            else:
                algo_content = py_script or DEFAULT_ALGO_SCRIPTS[category]
        except KeyError:
            raise Exception("Invalid algo category", category)

        dockerfile = dockerfile or self.default_algo_dockerfile

        algo_zip = utils.create_archive(
            tmpdir / "algo",
            ("algo.py", algo_content),
            ("Dockerfile", dockerfile),
        )

        return AlgoSpec(
            category=category,
            name=name,
            description=str(description_path),
            file=str(algo_zip),
            permissions=permissions or DEFAULT_PERMISSIONS,
            metadata=metadata,
        )

    def create_traintuple(
        self,
        algo=None,
        dataset=None,
        data_samples=None,
        traintuples=None,
        outputs=None,
        tag=None,
        compute_plan_key=None,
        rank=None,
        metadata=None,
    ) -> TraintupleSpec:
        data_samples = data_samples or []
        traintuples = traintuples or []

        for t in traintuples:
            assert isinstance(t, (models.Traintuple, models.Aggregatetuple))

        return TraintupleSpec(
            algo_key=algo.key if algo else None,
            data_manager_key=dataset.key if dataset else None,
            train_data_sample_keys=_get_keys(data_samples),
            in_models_keys=[t.key for t in traintuples],
            tag=tag,
            metadata=metadata,
            compute_plan_key=compute_plan_key,
            outputs=outputs if outputs is not None else DEFAULT_TRAINTUPLE_OUTPUTS,
            rank=rank,
        )

    def create_aggregatetuple(
        self,
        algo=None,
        worker=None,
        traintuples=None,
        outputs=None,
        tag=None,
        compute_plan_key=None,
        rank=None,
        metadata=None,
    ) -> AggregatetupleSpec:
        traintuples = traintuples or []

        for t in traintuples:
            assert isinstance(t, (models.Traintuple, models.CompositeTraintuple, models.Aggregatetuple))

        return AggregatetupleSpec(
            algo_key=algo.key if algo else None,
            worker=worker,
            in_models_keys=[t.key for t in traintuples],
            outputs=outputs if outputs is not None else DEFAULT_AGGREGATETUPLE_OUTPUTS,
            tag=tag,
            metadata=metadata,
            compute_plan_key=compute_plan_key,
            rank=rank,
        )

    def create_composite_traintuple(
        self,
        algo=None,
        dataset=None,
        data_samples=None,
        head_traintuple=None,
        trunk_traintuple=None,
        outputs=None,
        tag=None,
        compute_plan_key=None,
        rank=None,
        permissions=None,
        metadata=None,
    ) -> CompositeTraintupleSpec:
        data_samples = data_samples or []

        if head_traintuple and trunk_traintuple:
            assert isinstance(head_traintuple, models.CompositeTraintuple)
            assert isinstance(trunk_traintuple, (models.CompositeTraintuple, models.Aggregatetuple))
            in_head_model_key = head_traintuple.key
            in_trunk_model_key = trunk_traintuple.key
        else:
            in_head_model_key = None
            in_trunk_model_key = None

        return CompositeTraintupleSpec(
            algo_key=algo.key if algo else None,
            data_manager_key=dataset.key if dataset else None,
            train_data_sample_keys=_get_keys(data_samples),
            in_head_model_key=in_head_model_key,
            in_trunk_model_key=in_trunk_model_key,
            outputs=outputs if outputs is not None else DEFAULT_COMPOSITE_TRAINTUPLE_OUTPUTS,
            tag=tag,
            metadata=metadata,
            compute_plan_key=compute_plan_key,
            rank=rank,
        )

    def create_predicttuple(
        self, algo, traintuple, dataset, data_samples, outputs=None, tag=None, metadata=None
    ) -> PredicttupleSpec:
        return PredicttupleSpec(
            algo_key=algo.key,
            traintuple_key=traintuple.key,
            data_manager_key=dataset.key,
            test_data_sample_keys=_get_keys(data_samples),
            outputs=outputs if outputs is not None else DEFAULT_PREDICTTUPLE_OUTPUTS,
            tag=tag,
            metadata=metadata,
        )

    def create_testtuple(
        self, algo, predicttuple, dataset, data_samples, outputs=None, tag=None, metadata=None
    ) -> TesttupleSpec:
        return TesttupleSpec(
            algo_key=algo.key,
            predicttuple_key=predicttuple.key,
            data_manager_key=dataset.key,
            test_data_sample_keys=_get_keys(data_samples),
            outputs=outputs if outputs is not None else DEFAULT_TESTTUPLE_OUTPUTS,
            tag=tag,
            metadata=metadata,
        )

    def create_compute_plan(self, key=None, tag="", name="Test compute plan", clean_models=False, metadata=None):
        return ComputePlanSpec(
            key=key or random_uuid(),
            traintuples=[],
            composite_traintuples=[],
            aggregatetuples=[],
            testtuples=[],
            predicttuples=[],
            tag=tag,
            name=name,
            metadata=metadata,
            clean_models=clean_models,
        )

    def add_compute_plan_tuples(self, compute_plan):
        return UpdateComputePlanSpec(
            traintuples=[],
            composite_traintuples=[],
            aggregatetuples=[],
            testtuples=[],
            predicttuples=[],
            key=compute_plan.key,
        )
