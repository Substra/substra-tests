import docker
import pytest
from substra.sdk import models
from substra.sdk.exceptions import InvalidRequest

from substratest.factory import AlgoCategory
from substratest.fl_interface import FLTaskInputGenerator
from substratest.fl_interface import OutputIdentifiers


def docker_available() -> bool:
    try:
        docker.from_env()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not docker_available(), reason="requires docker")


@pytest.mark.remote_only
@pytest.mark.slow
def test_execution_debug(client, hybrid_client, debug_factory, default_dataset):

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
        inputs=default_dataset.opener_input + default_dataset.train_data_sample_inputs[:1],
        worker=hybrid_client.organization_info().organization_id,
    )
    traintuple = hybrid_client.add_task(spec)
    assert traintuple.status == models.Status.done
    assert traintuple.outputs[OutputIdentifiers.model].value is not None

    # Add the testtuple
    spec = debug_factory.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.opener_input
        + default_dataset.train_data_sample_inputs[:1]
        + FLTaskInputGenerator.train_to_predict(traintuple.key),
        worker=hybrid_client.organization_info().organization_id,
    )
    predicttuple = hybrid_client.add_task(spec)
    assert predicttuple.status == models.Status.done

    spec = debug_factory.create_testtuple(
        algo=metric,
        inputs=default_dataset.opener_input
        + default_dataset.train_data_sample_inputs[:1]
        + FLTaskInputGenerator.predict_to_test(predicttuple.key),
        worker=hybrid_client.organization_info().organization_id,
    )
    testtuple = hybrid_client.add_task(spec)
    assert testtuple.status == models.Status.done
    assert testtuple.outputs[OutputIdentifiers.performance].value == 3


@pytest.mark.remote_only
@pytest.mark.slow
def test_debug_compute_plan_aggregate_composite(network, client, hybrid_client, debug_factory, default_datasets):
    """
    Debug / Compute plan version of the
    `test_aggregate_composite_traintuples` method from `test_execution.py`
    """
    worker = hybrid_client.organization_id
    number_of_rounds = 2

    # register algos on first organization
    spec = debug_factory.create_algo(AlgoCategory.composite)
    composite_algo = client.add_algo(spec)
    spec = debug_factory.create_algo(AlgoCategory.aggregate)
    aggregate_algo = client.add_algo(spec)
    spec = debug_factory.create_algo(AlgoCategory.predict_composite)
    predict_algo_composite = client.add_algo(spec)

    # launch execution
    previous_aggregate_tuple_key = None
    previous_composite_traintuple_keys = []

    cp_spec = debug_factory.create_compute_plan()

    for round_ in range(number_of_rounds):
        # create composite traintuple on each organization
        composite_traintuple_keys = []
        for index, dataset in enumerate(default_datasets):

            if previous_aggregate_tuple_key:
                input_models = FLTaskInputGenerator.composite_to_local(
                    previous_composite_traintuple_keys[index]
                ) + FLTaskInputGenerator.aggregate_to_shared(previous_aggregate_tuple_key)

            else:
                input_models = []

            spec = cp_spec.create_composite_traintuple(
                composite_algo=composite_algo,
                inputs=dataset.opener_input + [dataset.train_data_sample_inputs[0 + round_]] + input_models,
                worker=worker,
            )
            composite_traintuple_keys.append(spec.task_id)

        # create aggregate on its organization
        spec = cp_spec.create_aggregatetuple(
            aggregate_algo=aggregate_algo,
            worker=worker,
            inputs=FLTaskInputGenerator.composites_to_aggregate(composite_traintuple_keys),
        )

        # save state of round
        previous_aggregate_tuple_key = spec.task_id
        previous_composite_traintuple_keys = composite_traintuple_keys

    metrics = []
    for index, client in enumerate(network.clients):
        # create metric
        spec = debug_factory.create_algo(category=AlgoCategory.metric, offset=index)
        metric = client.add_algo(spec)
        metrics.append(metric)

    # last round: create associated testtuple
    for composite_traintuple_key, dataset, metric in zip(previous_composite_traintuple_keys, default_datasets, metrics):

        spec = cp_spec.create_predicttuple(
            algo=predict_algo_composite,
            inputs=dataset.train_data_inputs + FLTaskInputGenerator.composite_to_predict(composite_traintuple_key),
            worker=worker,
        )
        cp_spec.create_testtuple(
            algo=metric,
            inputs=dataset.train_data_inputs + FLTaskInputGenerator.predict_to_test(spec.task_id),
            worker=worker,
        )

    cp = hybrid_client.add_compute_plan(cp_spec)
    traintuples = hybrid_client.list_compute_plan_tasks(cp.key)
    composite_traintuples = client.list_compute_plan_tasks(cp.key)
    aggregatetuples = client.list_compute_plan_tasks(cp.key)
    predicttuples = client.list_compute_plan_tasks(cp.key)
    testtuples = client.list_compute_plan_tasks(cp.key)

    tuples = traintuples + composite_traintuples + aggregatetuples + predicttuples + testtuples
    for t in tuples:
        assert t.status == models.Status.done


@pytest.mark.remote_only
def test_debug_download_dataset(hybrid_client, default_dataset):
    hybrid_client.download_opener(default_dataset.key)


@pytest.mark.remote_only
@pytest.mark.slow
def test_test_data_traintuple(client, hybrid_client, debug_factory, default_dataset):
    """Check that we can't use test data samples for traintuples"""
    spec = debug_factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    #  Add the traintuple
    # create traintuple
    spec = debug_factory.create_traintuple(
        algo=algo,
        inputs=default_dataset.opener_input + default_dataset.test_data_sample_inputs[:1],
        worker=hybrid_client.organization_info().organization_id,
    )

    with pytest.raises(InvalidRequest) as e:
        hybrid_client.add_task(spec)
    assert "Cannot create train task with test data" in str(e.value)


@pytest.mark.remote_only
@pytest.mark.slow
def test_fake_data_sample_key(client, hybrid_client, debug_factory, default_dataset):
    """Check that a traintuple can't run with a fake train_data_sample_keys"""
    spec = debug_factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    #  Add the traintuple
    # create traintuple
    spec = debug_factory.create_traintuple(
        algo=algo,
        inputs=default_dataset.opener_input + FLTaskInputGenerator.data_samples(["fake_key"]),
        worker=hybrid_client.organization_info().organization_id,
    )

    with pytest.raises(InvalidRequest) as e:
        hybrid_client.add_task(spec)
    assert "Could not get all the data_samples in the database with the given data_sample_keys" in str(e.value)
