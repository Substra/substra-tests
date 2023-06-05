import docker
import pytest
from substra.sdk import models
from substra.sdk.exceptions import InvalidRequest

from substratest.factory import FunctionCategory
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
    spec = debug_factory.create_function(FunctionCategory.simple)
    simple_function = client.add_function(spec)
    spec = debug_factory.create_function(FunctionCategory.predict)
    predict_function = client.add_function(spec)

    metric_spec = debug_factory.create_function(category=FunctionCategory.metric)
    metric = client.add_function(metric_spec)

    #  Add the traintask
    # create traintask
    spec = debug_factory.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input + default_dataset.train_data_sample_inputs[:1],
        worker=hybrid_client.organization_info().organization_id,
    )
    traintask = hybrid_client.add_task(spec)
    assert traintask.status == models.Status.done

    # Raises an exception if the output asset have not been created
    hybrid_client.get_task_output_asset(traintask.key, OutputIdentifiers.model)

    # Add the testtask
    spec = debug_factory.create_predicttask(
        function=predict_function,
        inputs=default_dataset.opener_input
        + default_dataset.train_data_sample_inputs[:1]
        + FLTaskInputGenerator.train_to_predict(traintask.key),
        worker=hybrid_client.organization_info().organization_id,
    )
    predicttask = hybrid_client.add_task(spec)
    assert predicttask.status == models.Status.done

    spec = debug_factory.create_testtask(
        function=metric,
        inputs=default_dataset.opener_input
        + default_dataset.train_data_sample_inputs[:1]
        + FLTaskInputGenerator.predict_to_test(predicttask.key),
        worker=hybrid_client.organization_info().organization_id,
    )
    testtask = hybrid_client.add_task(spec)
    assert testtask.status == models.Status.done
    performance = hybrid_client.get_task_output_asset(testtask.key, OutputIdentifiers.performance)
    assert performance.asset == 3


@pytest.mark.remote_only
@pytest.mark.slow
def test_debug_compute_plan_aggregate_composite(network, client, hybrid_client, debug_factory, default_datasets):
    """
    Debug / Compute plan version of the
    `test_aggregate_composite_traintasks` method from `test_execution.py`
    """
    worker = hybrid_client.organization_id
    number_of_rounds = 2

    # register functions on first organization
    spec = debug_factory.create_function(FunctionCategory.composite)
    composite_function = client.add_function(spec)
    spec = debug_factory.create_function(FunctionCategory.aggregate)
    aggregate_function = client.add_function(spec)
    spec = debug_factory.create_function(FunctionCategory.predict_composite)
    predict_function_composite = client.add_function(spec)

    # launch execution
    previous_aggregate_task_key = None
    previous_composite_traintask_keys = []

    cp_spec = debug_factory.create_compute_plan()

    for round_ in range(number_of_rounds):
        # create composite traintask on each organization
        composite_traintask_keys = []
        for index, dataset in enumerate(default_datasets):
            if previous_aggregate_task_key:
                input_models = FLTaskInputGenerator.composite_to_local(
                    previous_composite_traintask_keys[index]
                ) + FLTaskInputGenerator.aggregate_to_shared(previous_aggregate_task_key)

            else:
                input_models = []

            spec = cp_spec.create_composite_traintask(
                composite_function=composite_function,
                inputs=dataset.opener_input + [dataset.train_data_sample_inputs[0 + round_]] + input_models,
                worker=worker,
            )
            composite_traintask_keys.append(spec.task_id)

        # create aggregate on its organization
        spec = cp_spec.create_aggregatetask(
            aggregate_function=aggregate_function,
            worker=worker,
            inputs=FLTaskInputGenerator.composites_to_aggregate(composite_traintask_keys),
        )

        # save state of round
        previous_aggregate_task_key = spec.task_id
        previous_composite_traintask_keys = composite_traintask_keys

    metrics = []
    for index, client in enumerate(network.clients):
        # create metric
        spec = debug_factory.create_function(category=FunctionCategory.metric, offset=index)
        metric = client.add_function(spec)
        metrics.append(metric)

    # last round: create associated testtask
    for composite_traintask_key, dataset, metric in zip(previous_composite_traintask_keys, default_datasets, metrics):
        spec = cp_spec.create_predicttask(
            function=predict_function_composite,
            inputs=dataset.train_data_inputs + FLTaskInputGenerator.composite_to_predict(composite_traintask_key),
            worker=worker,
        )
        cp_spec.create_testtask(
            function=metric,
            inputs=dataset.train_data_inputs + FLTaskInputGenerator.predict_to_test(spec.task_id),
            worker=worker,
        )

    cp = hybrid_client.add_compute_plan(cp_spec)
    traintasks = hybrid_client.list_compute_plan_tasks(cp.key)
    composite_traintasks = client.list_compute_plan_tasks(cp.key)
    aggregatetasks = client.list_compute_plan_tasks(cp.key)
    predicttasks = client.list_compute_plan_tasks(cp.key)
    testtasks = client.list_compute_plan_tasks(cp.key)

    tasks = traintasks + composite_traintasks + aggregatetasks + predicttasks + testtasks
    for t in tasks:
        assert t.status == models.Status.done


@pytest.mark.remote_only
def test_debug_download_dataset(hybrid_client, default_dataset):
    hybrid_client.download_opener(default_dataset.key)


@pytest.mark.remote_only
@pytest.mark.slow
def test_fake_data_sample_key(client, hybrid_client, debug_factory, default_dataset):
    """Check that a traintask can't run with a fake train_data_sample_keys"""
    spec = debug_factory.create_function(FunctionCategory.simple)
    function = client.add_function(spec)

    #  Add the traintask
    # create traintask
    spec = debug_factory.create_traintask(
        function=function,
        inputs=default_dataset.opener_input + FLTaskInputGenerator.data_samples(["fake_key"]),
        worker=hybrid_client.organization_info().organization_id,
    )

    with pytest.raises(InvalidRequest) as e:
        hybrid_client.add_task(spec)
    assert "Could not get all the data_samples in the database with the given data_sample_keys" in str(e.value)
