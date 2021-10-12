import pytest
import docker

from substra.sdk import models
from substratest.factory import AlgoCategory, Permissions


def docker_available() -> bool:
    try:
        docker.from_env()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not docker_available(), reason="requires docker")


@pytest.mark.remote_only
@pytest.mark.slow
def test_execution_debug(client, debug_client, factory, default_dataset, default_metric):

    spec = factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    #  Add the traintuple
    # create traintuple
    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys[0],
    )
    traintuple = debug_client.add_traintuple(spec)
    assert traintuple.status == models.Status.done
    assert len(traintuple.train.models) != 0

    # Add the testtuple
    spec = factory.create_testtuple(
        metrics=[default_metric],
        traintuple=traintuple,
        dataset=default_dataset,
        data_samples=[default_dataset.test_data_sample_keys[0]],
    )
    testtuple = debug_client.add_testtuple(spec)
    assert testtuple.status == models.Status.done
    assert list(testtuple.test.perfs.values())[0] == 3


@pytest.mark.remote_only
@pytest.mark.slow
def test_debug_compute_plan_aggregate_composite(client, debug_client, factory, default_datasets, default_metrics):
    """
    Debug / Compute plan version of the
    `test_aggregate_composite_traintuples` method from `test_execution.py`
    """
    aggregate_worker = debug_client.node_id
    number_of_rounds = 2

    # register algos on first node
    spec = factory.create_algo(AlgoCategory.composite)
    composite_algo = client.add_algo(spec)
    spec = factory.create_algo(AlgoCategory.aggregate)
    aggregate_algo = client.add_algo(spec)

    # launch execution
    previous_aggregatetuple_spec = None
    previous_composite_traintuple_specs = []

    cp_spec = factory.create_compute_plan()

    for round_ in range(number_of_rounds):
        # create composite traintuple on each node
        composite_traintuple_specs = []
        for index, dataset in enumerate(default_datasets):
            kwargs = {}
            if previous_aggregatetuple_spec:
                kwargs = {
                    'in_head_model': previous_composite_traintuple_specs[index],
                    'in_trunk_model': previous_aggregatetuple_spec,
                }
            spec = cp_spec.add_composite_traintuple(
                composite_algo=composite_algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0 + round_]],
                out_trunk_model_permissions=Permissions(public=False, authorized_ids=[debug_client.node_id]),
                **kwargs,
            )
            composite_traintuple_specs.append(spec)

        # create aggregate on its node
        spec = cp_spec.add_aggregatetuple(
            aggregate_algo=aggregate_algo,
            worker=aggregate_worker,
            in_models=composite_traintuple_specs,
        )

        # save state of round
        previous_aggregatetuple_spec = spec
        previous_composite_traintuple_specs = composite_traintuple_specs

    # last round: create associated testtuple
    for composite_traintuple_spec, dataset, metric in zip(
            previous_composite_traintuple_specs, default_datasets, default_metrics):
        cp_spec.add_testtuple(
            metrics=[metric],
            dataset=dataset,
            data_samples=dataset.test_data_sample_keys,
            traintuple_spec=composite_traintuple_spec,
        )

    cp = debug_client.add_compute_plan(cp_spec)
    traintuples = debug_client.list_compute_plan_traintuples(cp.key)
    composite_traintuples = client.list_compute_plan_composite_traintuples(cp.key)
    aggregatetuples = client.list_compute_plan_aggregatetuples(cp.key)
    testtuples = client.list_compute_plan_testtuples(cp.key)

    tuples = traintuples + composite_traintuples + aggregatetuples + testtuples
    for t in tuples:
        assert t.status == models.Status.done


@pytest.mark.remote_only
def test_debug_download_dataset(debug_client, default_dataset):
    debug_client.download_opener(default_dataset.key)
