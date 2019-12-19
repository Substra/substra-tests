import pathlib
import random
import shutil
import tempfile

from .. import utils, assets
from . import specs, experiment


def _shorten_name(name):
    """Format asset name to ensure they match the backend requirements."""
    if len(name) < 100:
        return name
    return name[:75] + '...' + name[:20]


class Counter:
    def __init__(self):
        self._idx = -1

    def inc(self):
        self._idx += 1
        return self._idx


def randomize_pyscript(pyscript):
    rdm = random.random()
    return f'# random={rdm} \n' + pyscript


def randomize_markdown(markdown):
    rdm = random.random()
    return markdown + f'\n# random={rdm}'


class AssetsFactory:

    def __init__(self, name, experiment_module):
        self._data_sample_counter = Counter()
        self._dataset_counter = Counter()
        self._objective_counter = Counter()
        self._algo_counter = Counter()
        self._workdir = pathlib.Path(tempfile.mkdtemp())

        self._experiment = experiment.create(experiment_module)
        self._uuid = name

    @property
    def experiment(self):
        return self._experiment

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        shutil.rmtree(str(self._workdir), ignore_errors=True)

    def create_data_sample(self, datasets=None, test_only=False):
        idx = self._data_sample_counter.inc()
        tmpdir = self._workdir / f'data-{idx}'
        tmpdir.mkdir()

        content = next(self.experiment.data_sample_generator)

        data_filepath = tmpdir / 'data.bin'
        with open(data_filepath, 'wb') as f:
            f.write(content)

        datasets = datasets or []

        return specs.DataSample(
            path=str(tmpdir),
            test_only=test_only,
            data_manager_keys=[d.key for d in datasets],
        )

    def create_dataset(self, permissions=None):
        idx = self._dataset_counter.inc()
        tmpdir = self._workdir / f'dataset-{idx}'
        tmpdir.mkdir()
        name = _shorten_name(f'{self._uuid} - Dataset {idx}')

        description_path = tmpdir / 'description.md'
        description_content = name
        with open(description_path, 'w') as f:
            f.write(description_content)

        opener_path = tmpdir / 'opener.py'
        opener_content = randomize_pyscript(self.experiment.opener_script)
        with open(opener_path, 'w') as f:
            f.write(opener_content)

        return specs.Dataset(
            name=name,
            data_opener=str(opener_path),
            type='Test',
            description=str(description_path),
            permissions=permissions or specs.DEFAULT_PERMISSIONS,
        )

    def create_objective(self, dataset=None, data_samples=None, permissions=None):
        idx = self._objective_counter.inc()
        tmpdir = self._workdir / f'objective-{idx}'
        tmpdir.mkdir()
        name = _shorten_name(f'{self._uuid} - Objective {idx}')

        description_path = tmpdir / 'description.md'
        # description markdown must be randomize as the description is used to
        # compute the objective unique key
        # https://github.com/SubstraFoundation/substra-chaincode/issues/33
        description_content = randomize_markdown(f'# Description of {name})')
        with open(description_path, 'w') as f:
            f.write(description_content)

        metrics_content = randomize_pyscript(self.experiment.metrics_script)

        metrics_zip = utils.create_archive(
            tmpdir / 'metrics',
            ('metrics.py', metrics_content),
            ('Dockerfile', self.experiment.metrics_dockerfile),
        )

        data_samples = data_samples or []

        return specs.Objective(
            name=name,
            description=str(description_path),
            metrics_name='test metrics',
            metrics=str(metrics_zip),
            permissions=permissions or specs.DEFAULT_PERMISSIONS,
            test_data_sample_keys=specs.get_keys(data_samples),
            test_data_manager_key=dataset.key if dataset else None,
        )

    def _create_algo(self, py_script, permissions=None):
        idx = self._algo_counter.inc()
        tmpdir = self._workdir / f'algo-{idx}'
        tmpdir.mkdir()
        name = _shorten_name(f'{self._uuid} - Algo {idx}')

        description_path = tmpdir / 'description.md'
        with open(description_path, 'w') as f:
            f.write(f'# Description of {name})')

        algo_content = randomize_pyscript(py_script)

        algo_zip = utils.create_archive(
            tmpdir / 'algo',
            ('algo.py', algo_content),
            ('Dockerfile', self.experiment.algo_dockerfile),
        )

        return specs.Algo(
            name=name,
            description=str(description_path),
            file=str(algo_zip),
            permissions=permissions or specs.DEFAULT_PERMISSIONS,
        )

    def create_algo(self, py_script=None, permissions=None):
        return self._create_algo(
            py_script or self.experiment.algo_script,
            permissions=permissions,
        )

    def create_aggregate_algo(self, py_script=None, permissions=None):
        return self._create_algo(
            py_script or self.experiment.aggregate_algo_script,
            permissions=permissions,
        )

    def create_composite_algo(self, py_script=None, permissions=None):
        return self._create_algo(
            py_script or self.experiment.composite_algo_script,
            permissions=permissions,
        )

    def create_traintuple(self, algo=None, dataset=None,
                          data_samples=None, traintuples=None, tag=None,
                          compute_plan_id=None, rank=None):
        data_samples = data_samples or []
        traintuples = traintuples or []

        for t in traintuples:
            assert isinstance(t, assets.Traintuple)

        return specs.Traintuple(
            algo_key=algo.key if algo else None,
            data_manager_key=dataset.key if dataset else None,
            train_data_sample_keys=specs.get_keys(data_samples),
            in_models_keys=[t.key for t in traintuples],
            tag=tag,
            compute_plan_id=compute_plan_id,
            rank=rank,
        )

    def create_aggregatetuple(self, algo=None, worker=None,
                              traintuples=None, tag=None, compute_plan_id=None,
                              rank=None):
        traintuples = traintuples or []

        for t in traintuples:
            assert isinstance(t, (assets.Traintuple, assets.CompositeTraintuple))

        return specs.Aggregatetuple(
            algo_key=algo.key if algo else None,
            worker=worker,
            in_models_keys=[t.key for t in traintuples],
            tag=tag,
            compute_plan_id=compute_plan_id,
            rank=rank,
        )

    def create_composite_traintuple(self, algo=None, dataset=None,
                                    data_samples=None, head_traintuple=None,
                                    trunk_traintuple=None, tag=None,
                                    compute_plan_id=None, rank=None,
                                    permissions=None):
        data_samples = data_samples or []

        if head_traintuple and trunk_traintuple:
            assert isinstance(head_traintuple, assets.CompositeTraintuple)
            assert isinstance(
                trunk_traintuple,
                (assets.CompositeTraintuple, assets.Aggregatetuple)
            )
            in_head_model_key = head_traintuple.key
            in_trunk_model_key = trunk_traintuple.key
        else:
            in_head_model_key = None
            in_trunk_model_key = None

        return specs.CompositeTraintuple(
            algo_key=algo.key if algo else None,
            data_manager_key=dataset.key if dataset else None,
            train_data_sample_keys=specs.get_keys(data_samples),
            in_head_model_key=in_head_model_key,
            in_trunk_model_key=in_trunk_model_key,
            tag=tag,
            compute_plan_id=compute_plan_id,
            rank=rank,
            out_trunk_model_permissions=permissions or specs.DEFAULT_PERMISSIONS,
        )

    def create_testtuple(self, objective=None, traintuple=None, tag=None):
        return specs.Testtuple(
            objective_key=objective.key if objective else None,
            traintuple_key=traintuple.key if traintuple else None,
            tag=tag,
        )

    def create_compute_plan(self):
        return specs.ComputePlan(
            traintuples=[],
            composite_traintuples=[],
            aggregatetuples=[],
            testtuples=[],
        )
