import docker
import pytest
from substra.sdk import models
from substra.sdk.exceptions import InvalidRequest

from substratest.factory import DEFAULT_COMPOSITE_ALGO_SCRIPT
from substratest.factory import AlgoCategory
from substratest.factory import Permissions


def docker_available() -> bool:
    try:
        docker.from_env()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not docker_available(), reason="requires docker")


@pytest.mark.remote_only
@pytest.mark.slow
def test_execution_debug(client, debug_client, debug_factory, default_dataset):

    spec = debug_factory.create_algo(AlgoCategory.simple)
    simple_algo = client.add_algo(spec)
    spec = debug_factory.create_algo(AlgoCategory.predict)
    predict_algo = client.add_algo(spec)

    metric_spec = debug_factory.create_algo(category=AlgoCategory.metric)
    metric = client.add_algo(metric_spec)

    #  Add the traintuple
    # create traintuple
    spec = debug_factory.create_traintuple(
        algo=simple_algo,
        dataset=default_dataset,
        data_samples=[default_dataset.train_data_sample_keys[0]],
    )
    traintuple = debug_client.add_traintuple(spec)
    assert traintuple.status == models.Status.done
    assert len(traintuple.train.models) != 0

    # Add the testtuple
    spec = debug_factory.create_predicttuple(
        algo=predict_algo,
        traintuple=traintuple,
        dataset=default_dataset,
        data_samples=[default_dataset.test_data_sample_keys[0]],
    )
    predicttuple = debug_client.add_predicttuple(spec)
    assert predicttuple.status == models.Status.done

    spec = debug_factory.create_testtuple(
        algo=metric,
        predicttuple=predicttuple,
        dataset=default_dataset,
        data_samples=[default_dataset.test_data_sample_keys[0]],
    )
    testtuple = debug_client.add_testtuple(spec)
    assert testtuple.status == models.Status.done
    assert list(testtuple.test.perfs.values())[0] == 3


@pytest.mark.remote_only
@pytest.mark.slow
def test_debug_compute_plan_aggregate_composite(network, client, debug_client, debug_factory, default_datasets):
    """
    Debug / Compute plan version of the
    `test_aggregate_composite_traintuples` method from `test_execution.py`
    """
    aggregate_worker = debug_client.organization_id
    number_of_rounds = 2

    # register algos on first organization
    spec = debug_factory.create_algo(AlgoCategory.composite)
    composite_algo = client.add_algo(spec)
    spec = debug_factory.create_algo(AlgoCategory.aggregate)
    aggregate_algo = client.add_algo(spec)
    spec = debug_factory.create_algo(AlgoCategory.predict, py_script=DEFAULT_COMPOSITE_ALGO_SCRIPT)
    predict_algo_composite = client.add_algo(spec)

    # launch execution
    previous_aggregatetuple_spec = None
    previous_composite_traintuple_specs = []

    cp_spec = debug_factory.create_compute_plan()

    for round_ in range(number_of_rounds):
        # create composite traintuple on each organization
        composite_traintuple_specs = []
        for index, dataset in enumerate(default_datasets):
            kwargs = {}
            if previous_aggregatetuple_spec:
                kwargs = {
                    "in_head_model": previous_composite_traintuple_specs[index],
                    "in_trunk_model": previous_aggregatetuple_spec,
                }
            spec = cp_spec.create_composite_traintuple(
                composite_algo=composite_algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0 + round_]],
                out_trunk_model_permissions=Permissions(public=False, authorized_ids=[debug_client.organization_id]),
                **kwargs,
            )
            composite_traintuple_specs.append(spec)

        # create aggregate on its organization
        spec = cp_spec.create_aggregatetuple(
            aggregate_algo=aggregate_algo,
            worker=aggregate_worker,
            in_models=composite_traintuple_specs,
        )

        # save state of round
        previous_aggregatetuple_spec = spec
        previous_composite_traintuple_specs = composite_traintuple_specs

    metrics = []
    for index, client in enumerate(network.clients):
        # create metric
        spec = debug_factory.create_algo(category=AlgoCategory.metric, offset=index)
        metric = client.add_algo(spec)
        metrics.append(metric)

    # last round: create associated testtuple
    for composite_traintuple_spec, dataset, metric in zip(
        previous_composite_traintuple_specs, default_datasets, metrics
    ):

        spec = cp_spec.create_predicttuple(
            algo=predict_algo_composite,
            dataset=dataset,
            data_samples=dataset.test_data_sample_keys,
            traintuple_spec=composite_traintuple_spec,
        )
        cp_spec.create_testtuple(
            algo=metric,
            dataset=dataset,
            data_samples=dataset.test_data_sample_keys,
            predicttuple_spec=spec,
        )

    cp = debug_client.add_compute_plan(cp_spec)
    traintuples = debug_client.list_compute_plan_traintuples(cp.key)
    composite_traintuples = client.list_compute_plan_composite_traintuples(cp.key)
    aggregatetuples = client.list_compute_plan_aggregatetuples(cp.key)
    predicttuples = client.list_compute_plan_predicttuples(cp.key)
    testtuples = client.list_compute_plan_testtuples(cp.key)

    tuples = traintuples + composite_traintuples + aggregatetuples + predicttuples + testtuples
    for t in tuples:
        assert t.status == models.Status.done


@pytest.mark.remote_only
def test_debug_download_dataset(debug_client, default_dataset):
    debug_client.download_opener(default_dataset.key)


@pytest.mark.remote_only
@pytest.mark.slow
def test_test_data_traintuple(client, debug_client, debug_factory, default_dataset):
    """Check that we can't use test data samples for traintuples"""
    spec = debug_factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    #  Add the traintuple
    # create traintuple
    spec = debug_factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[default_dataset.test_data_sample_keys[0]],
    )

    with pytest.raises(InvalidRequest) as e:
        debug_client.add_traintuple(spec)
    assert "Cannot create train task with test data" in str(e.value)


@pytest.mark.remote_only
@pytest.mark.slow
def test_fake_data_sample_key(client, debug_client, debug_factory, default_dataset):
    """Check that a traintuple can't run with a fake train_data_sample_keys"""
    spec = debug_factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    #  Add the traintuple
    # create traintuple
    spec = debug_factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=["fake_key"],
    )

    with pytest.raises(InvalidRequest) as e:
        debug_client.add_traintuple(spec)
    assert "Could not get all the data_samples in the database with the given data_sample_keys" in str(e.value)
