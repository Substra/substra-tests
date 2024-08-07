import os

import pytest
import substra
from substra.sdk.exceptions import InternalServerError
from substra.sdk.exceptions import TaskAssetNotFoundError
from substra.sdk.models import ComputeTaskStatus
from substra.sdk.models import InputRef
from substra.sdk.schemas import AssetKind
from substra.sdk.schemas import ComputeTaskOutputSpec
from substra.sdk.schemas import FunctionOutputSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import TaskSpec

import substratest as sbt
from substratest.factory import AugmentedDataset
from substratest.factory import FunctionCategory
from substratest.fl_interface import FLTaskInputGenerator
from substratest.fl_interface import FLTaskOutputGenerator
from substratest.fl_interface import InputIdentifiers
from substratest.fl_interface import OutputIdentifiers


@pytest.mark.slow
def test_tasks_execution_on_same_organization(factory, network, client, default_dataset, default_metric, worker):
    """Execution of a traintask, a following testtask and a following traintask."""

    spec = factory.create_function(FunctionCategory.simple)
    function = client.add_function(spec)

    predict_function_spec = factory.create_function(FunctionCategory.predict)
    predict_function = client.add_function(predict_function_spec)

    # create traintask
    def get_traintask_spec() -> TaskSpec:
        return factory.create_traintask(
            function=function,
            inputs=default_dataset.train_data_inputs,
            metadata={"foo": "bar"},
            worker=worker,
        )

    spec = get_traintask_spec()
    traintask = client.add_task(spec)
    traintask = client.wait_task(traintask.key, raise_on_failure=True)
    assert traintask.error_type is None
    assert traintask.metadata == {"foo": "bar"}
    assert len(traintask.outputs) == 1
    # Raises an exception if the output asset have not been created
    output = client.get_task_output_asset(traintask.key, OutputIdentifiers.shared)

    if network.options.enable_model_download:
        model = output.asset
        assert client.get_model_content(model.key) == b'{"value": 2.2}'

    # check we can add twice the same traintask
    spec = get_traintask_spec()
    client.add_task(spec)

    # create testtask
    spec = factory.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask.key),
        worker=worker,
    )

    predicttask = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    predicttask = client.wait_task(predicttask.key, raise_on_failure=True)
    assert predicttask.error_type is None

    spec = factory.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask.key),
        worker=worker,
    )
    testtask = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    testtask = client.wait_task(testtask.key, raise_on_failure=True)
    assert testtask.error_type is None
    performance = client.get_task_output_asset(testtask.key, OutputIdentifiers.performance)
    assert performance.asset == pytest.approx(2)

    # add a traintask depending on first traintask
    first_traintask_key = traintask.key
    spec = factory.create_traintask(
        function=function,
        inputs=default_dataset.train_data_inputs + FLTaskInputGenerator.trains_to_train([first_traintask_key]),
        metadata=None,
        worker=worker,
    )
    traintask = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    traintask = client.wait_task(traintask.key, raise_on_failure=True)
    assert testtask.error_type is None
    assert traintask.metadata == {}

    expected_inputs = default_dataset.train_data_inputs + FLTaskInputGenerator.trains_to_train([first_traintask_key])
    assert traintask.inputs == [InputRef.model_validate(exp_input.model_dump()) for exp_input in expected_inputs]


@pytest.mark.slow
def test_federated_learning_workflow(factory, client, default_datasets, workers):
    """Test federated learning workflow on each organization."""

    # create test environment
    spec = factory.create_function(FunctionCategory.simple)
    function = client.add_function(spec)

    # create 1 traintask per dataset and chain them
    traintask = None
    rank = 0
    compute_plan_key = None

    # default_datasets contains datasets on each organization and
    # that has a result we can use for federated learning
    for index, dataset in enumerate(default_datasets):
        traintasks = [traintask.key] if traintask else []

        spec = factory.create_traintask(
            function=function,
            inputs=dataset.train_data_inputs + FLTaskInputGenerator.trains_to_train(traintasks),
            tag="foo",
            rank=rank,
            compute_plan_key=compute_plan_key,
            worker=workers[index],
        )
        traintask = client.add_task(spec)
        traintask = client.wait_task(traintask.key, raise_on_failure=True)
        out_models = client.get_task_models(traintask.key)
        assert traintask.error_type is None
        assert len(traintask.outputs) == 1
        assert len(out_models) == 1
        assert traintask.tag == "foo"
        assert traintask.compute_plan_key  # check it is not None or ''

        rank += 1
        compute_plan_key = traintask.compute_plan_key

    # check a compute plan has been created and its status is at done
    cp = client.get_compute_plan(compute_plan_key)
    assert cp.status == "PLAN_STATUS_DONE"


@pytest.mark.slow
@pytest.mark.remote_only
def test_tasks_execution_on_different_organizations(
    factory,
    client_1,
    client_2,
    default_metric_1,
    default_dataset_1,
    default_dataset_2,
    workers,
):
    """Execution of a traintask on organization 1 and the following testtask on organization 2."""
    # add test data samples / dataset / metric on organization 1
    spec = factory.create_function(FunctionCategory.simple)
    function_2 = client_2.add_function(spec)

    predict_function_spec = factory.create_function(FunctionCategory.predict)
    predict_function_2 = client_2.add_function(predict_function_spec)

    client_2.wait_function(function_2.key)
    client_2.wait_function(predict_function_2.key)

    # add traintask on organization 2; should execute on organization 2 (dataset located on organization 2)
    spec = factory.create_traintask(
        function=function_2,
        inputs=default_dataset_2.train_data_inputs,
        worker=workers[1],
    )
    traintask = client_1.add_task(spec)
    # `raises = True`, will fail if task not successful
    traintask = client_1.wait_task(traintask.key, raise_on_failure=True)
    assert traintask.error_type is None
    assert len(traintask.outputs) == 1
    # Raises an exception if the output asset have not been created
    client_1.get_task_output_asset(traintask.key, OutputIdentifiers.shared)
    assert traintask.worker == client_2.organization_id

    # add testtask; should execute on organization 1 (default_dataset_1 is located on organization 1)
    spec = factory.create_predicttask(
        function=predict_function_2,
        inputs=default_dataset_1.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask.key),
        worker=workers[0],
    )
    predicttask = client_1.add_task(spec)
    # `raises = True`, will fail if task not successful
    predicttask = client_1.wait_task(predicttask.key, raise_on_failure=True)
    assert predicttask.error_type is None
    assert predicttask.worker == client_1.organization_id

    spec = factory.create_testtask(
        function=default_metric_1,
        inputs=default_dataset_1.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask.key),
        worker=workers[0],
    )
    testtask = client_1.add_task(spec)
    # `raises = True`, will fail if task not successful
    testtask = client_1.wait_task(testtask.key, raise_on_failure=True)
    assert testtask.error_type is None
    assert testtask.worker == client_1.organization_id
    performance = client_2.get_task_output_asset(testtask.key, OutputIdentifiers.performance)
    assert performance.asset == pytest.approx(2)


@pytest.mark.slow
@pytest.mark.subprocess_skip
def test_function_build_failure(factory, network, default_dataset_1, worker):
    """Invalid Dockerfile is causing compute task failure."""

    dockerfile = factory.default_function_dockerfile(
        method_name=sbt.factory.DEFAULT_FUNCTION_NAME[FunctionCategory.simple]
    )
    dockerfile += "\nRUN invalid_command"
    spec = factory.create_function(category=FunctionCategory.simple, dockerfile=dockerfile)
    function = network.clients[0].add_function(spec)

    spec = factory.create_traintask(function=function, inputs=default_dataset_1.train_data_inputs, worker=worker)

    if network.clients[0].backend_mode != substra.BackendType.REMOTE:
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.BuildError):
            network.clients[0].add_task(spec)
    else:
        traintask = network.clients[0].add_task(spec)
        traintask = network.clients[0].wait_task(traintask.key, raise_on_failure=False)

        assert traintask.status == ComputeTaskStatus.failed
        assert traintask.error_type == substra.sdk.models.TaskErrorType.build
        with pytest.raises(TaskAssetNotFoundError):
            network.clients[0].get_task_output_asset(traintask.key, OutputIdentifiers.shared)

        for client in (network.clients[0], network.clients[1]):
            logs = client.download_logs(traintask.key)
            assert "invalid_command: not found" in logs
            assert client.get_logs(traintask.key) == logs


@pytest.mark.slow
@pytest.mark.subprocess_skip
def test_function_build_failure_different_backend(factory, network, default_dataset_1, worker, channel):
    """Invalid Dockerfile is causing compute task failure - function is built on a different backend."""

    dockerfile = factory.default_function_dockerfile(
        method_name=sbt.factory.DEFAULT_FUNCTION_NAME[FunctionCategory.simple]
    )
    dockerfile += "\nRUN invalid_command"
    spec = factory.create_function(category=FunctionCategory.simple, dockerfile=dockerfile)
    function = network.clients[1].add_function(spec)
    channel.wait_for_asset_synchronized(function)

    spec = factory.create_traintask(function=function, inputs=default_dataset_1.train_data_inputs, worker=worker)

    if network.clients[1].backend_mode != substra.BackendType.REMOTE:
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.BuildError):
            network.clients[0].add_task(spec)
    else:
        traintask = network.clients[0].add_task(spec)
        traintask = network.clients[0].wait_task(traintask.key, raise_on_failure=False)

        assert traintask.status == ComputeTaskStatus.failed
        assert traintask.error_type == substra.sdk.models.TaskErrorType.build
        with pytest.raises(TaskAssetNotFoundError):
            network.clients[0].get_task_output_asset(traintask.key, OutputIdentifiers.shared)

        for client in (network.clients[0], network.clients[1]):
            logs = client.download_logs(traintask.key)
            assert "invalid_command: not found" in logs
            assert client.get_logs(traintask.key) == logs


@pytest.mark.slow
def test_task_execution_failure(factory, network, default_dataset_1, worker):
    """Invalid function script is causing compute task failure."""

    spec = factory.create_function(category=FunctionCategory.simple, py_script=sbt.factory.INVALID_FUNCTION_SCRIPT)
    function = network.clients[0].add_function(spec)

    spec = factory.create_traintask(function=function, inputs=default_dataset_1.train_data_inputs, worker=worker)

    if network.clients[0].backend_mode != substra.BackendType.REMOTE:
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.ExecutionError):
            network.clients[0].add_task(spec)
    else:
        traintask = network.clients[0].add_task(spec)
        traintask = network.clients[0].wait_task(traintask.key, raise_on_failure=False)

        assert traintask.status == ComputeTaskStatus.failed
        assert traintask.error_type == substra.sdk.models.TaskErrorType.execution
        with pytest.raises(TaskAssetNotFoundError):
            network.clients[0].get_task_output_asset(traintask.key, OutputIdentifiers.shared)

        for client in (network.clients[0], network.clients[1]):
            logs = client.download_logs(traintask.key)
            assert "Traceback (most recent call last):" in logs
            assert client.get_logs(traintask.key) == logs


@pytest.mark.slow
def test_testtask_with_several_outputs(factory, client, default_dataset, worker):
    """Test with a test task with several performance output."""

    identifier_1 = "name_1"
    identifier_2 = "name_2"

    custom_metric_script = f"""
import json
import substratools as tools

@tools.register
def score(inputs, outputs, task_properties):
    tools.save_performance(1, outputs['{identifier_1}'])
    tools.save_performance(2, outputs['{identifier_2}'])

if __name__ == '__main__':
    tools.execute()
    """

    # add train and predict functions
    spec = factory.create_function(FunctionCategory.simple)
    function = client.add_function(spec)

    predict_function_spec = factory.create_function(FunctionCategory.predict)
    predict_function = client.add_function(predict_function_spec)

    # add traintask
    spec = factory.create_traintask(
        function=function,
        inputs=default_dataset.train_data_inputs,
        worker=worker,
    )
    traintask = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    traintask = client.wait_task(traintask.key, raise_on_failure=True)

    # add predicttask
    spec = factory.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask.key),
        worker=worker,
    )
    predicttask = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    predicttask = client.wait_task(predicttask.key, raise_on_failure=True)

    # add metric function
    spec = factory.create_function(category=FunctionCategory.metric, py_script=custom_metric_script)
    spec.outputs = [
        FunctionOutputSpec(identifier=identifier_1, kind=AssetKind.performance.value, multiple=False),
        FunctionOutputSpec(identifier=identifier_2, kind=AssetKind.performance.value, multiple=False),
    ]

    metric = client.add_function(spec)

    spec = factory.create_testtask(
        function=metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask.key),
        outputs={
            identifier_1: ComputeTaskOutputSpec(permissions=Permissions(public=True, authorized_ids=[])),
            identifier_2: ComputeTaskOutputSpec(permissions=Permissions(public=True, authorized_ids=[])),
        },
        worker=worker,
    )

    testtask = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    testtask = client.wait_task(testtask.key, raise_on_failure=True)
    assert testtask.error_type is None
    output_1 = client.get_task_output_asset(testtask.key, identifier_1)
    output_2 = client.get_task_output_asset(testtask.key, identifier_2)
    assert output_1.asset == 1
    assert output_2.asset == 2


def test_testtask_with_same_output_identifer(factory, client):
    identifier_1 = "same_name"
    identifier_2 = "same_name"

    spec = factory.create_function(category=FunctionCategory.metric)
    spec.outputs = [
        FunctionOutputSpec(identifier=identifier_1, kind=AssetKind.performance.value, multiple=False),
        FunctionOutputSpec(identifier=identifier_2, kind=AssetKind.performance.value, multiple=False),
    ]

    with pytest.raises(ValueError):
        client.add_function(spec)


@pytest.mark.slow
def test_composite_traintask_execution_failure(factory, client, default_dataset, worker):
    """Invalid composite function script is causing traintask failure."""

    spec = factory.create_function(FunctionCategory.composite, py_script=sbt.factory.INVALID_COMPOSITE_FUNCTION_SCRIPT)
    function = client.add_function(spec)

    spec = factory.create_composite_traintask(
        function=function, inputs=default_dataset.train_data_inputs, worker=worker
    )
    if client.backend_mode == substra.BackendType.REMOTE:
        composite_traintask = client.add_task(spec)
        composite_traintask = client.wait_task(composite_traintask.key, raise_on_failure=False)

        assert composite_traintask.status == ComputeTaskStatus.failed
        assert composite_traintask.error_type == substra.sdk.models.TaskErrorType.execution
        with pytest.raises(TaskAssetNotFoundError):
            client.get_task_output_asset(composite_traintask.key, OutputIdentifiers.local)

        with pytest.raises(TaskAssetNotFoundError):
            client.get_task_output_asset(composite_traintask.key, OutputIdentifiers.shared)
        assert "Traceback (most recent call last):" in client.download_logs(composite_traintask.key)

    elif client.backend_mode in (substra.BackendType.LOCAL_SUBPROCESS, substra.BackendType.LOCAL_DOCKER):
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.ExecutionError):
            composite_traintask = client.add_task(spec)

    else:
        raise NotImplementedError(f"Backend mode '{client.backend_mode}' is not supported.")


@pytest.mark.slow
def test_aggregatetask_execution_failure(factory, client, default_dataset, worker):
    """Invalid function script is causing aggregatetask failure."""

    spec = factory.create_function(FunctionCategory.composite)
    composite_function = client.add_function(spec)

    spec = factory.create_function(FunctionCategory.aggregate, py_script=sbt.factory.INVALID_AGGREGATE_FUNCTION_SCRIPT)
    aggregate_function = client.add_function(spec)

    composite_traintask_keys = []
    for i in [0, 1]:
        spec = factory.create_composite_traintask(
            function=composite_function,
            inputs=default_dataset.opener_input + [default_dataset.train_data_sample_inputs[i]],
            worker=worker,
        )
        composite_traintask_keys.append(client.add_task(spec).key)

    spec = factory.create_aggregatetask(
        function=aggregate_function,
        inputs=FLTaskInputGenerator.composites_to_aggregate(composite_traintask_keys),
        worker=client.organization_id,
    )

    if client.backend_mode == substra.BackendType.REMOTE:
        aggregatetask = client.add_task(spec)
        aggregatetask = client.wait_task(aggregatetask.key, raise_on_failure=False)

        for composite_traintask_key in composite_traintask_keys:
            composite_traintask = client.get_task(composite_traintask_key)
            assert composite_traintask.status == ComputeTaskStatus.done
            assert composite_traintask.error_type is None

        assert aggregatetask.status == ComputeTaskStatus.failed
        assert aggregatetask.error_type == substra.sdk.models.TaskErrorType.execution
        with pytest.raises(TaskAssetNotFoundError):
            client.get_task_output_asset(aggregatetask.key, OutputIdentifiers.shared)
        assert "Traceback (most recent call last):" in client.download_logs(aggregatetask.key)

    elif client.backend_mode in (substra.BackendType.LOCAL_SUBPROCESS, substra.BackendType.LOCAL_DOCKER):
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.ExecutionError):
            aggregatetask = client.add_task(spec)

    else:
        raise NotImplementedError(f"Backend mode '{client.backend_mode}' is not supported.")


@pytest.mark.slow
def test_composite_traintasks_execution(factory, client, default_dataset, default_metric, worker):
    """Execution of composite traintasks."""

    spec = factory.create_function(FunctionCategory.composite)
    function = client.add_function(spec)

    spec = factory.create_function(FunctionCategory.predict_composite)
    predict_function = client.add_function(spec)

    # first composite traintask
    spec = factory.create_composite_traintask(
        function=function,
        inputs=default_dataset.train_data_inputs,
        worker=worker,
    )
    composite_traintask_1 = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    composite_traintask_1 = client.wait_task(composite_traintask_1.key, raise_on_failure=True)
    assert composite_traintask_1.error_type is None
    assert len(composite_traintask_1.outputs) == 2

    # second composite traintask
    spec = factory.create_composite_traintask(
        function=function,
        inputs=default_dataset.train_data_inputs
        + FLTaskInputGenerator.composite_to_composite(composite_traintask_1.key),
        worker=worker,
    )
    composite_traintask_2 = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    composite_traintask_2 = client.wait_task(composite_traintask_2.key, raise_on_failure=True)
    assert composite_traintask_2.error_type is None
    assert len(composite_traintask_2.outputs) == 2

    # add a 'composite' testtask
    spec = factory.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.composite_to_predict(composite_traintask_2.key),
        worker=worker,
    )
    predicttask = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    predicttask = client.wait_task(predicttask.key, raise_on_failure=True)
    assert predicttask.status == ComputeTaskStatus.done
    assert predicttask.error_type is None

    spec = factory.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask.key),
        worker=worker,
    )
    testtask = client.add_task(spec)
    testtask = client.wait_task(testtask.key, raise_on_failure=True)
    assert testtask.error_type is None
    performance = client.get_task_output_asset(testtask.key, OutputIdentifiers.performance)
    assert performance.asset == pytest.approx(32)

    # list composite traintask
    composite_traintasks = client.list_task()
    composite_traintask_keys = set([t.key for t in composite_traintasks])
    assert set([composite_traintask_1.key, composite_traintask_2.key]).issubset(composite_traintask_keys)


@pytest.mark.slow
def test_aggregatetask(factory, client, default_metric, default_dataset, worker):
    """Execution of aggregatetask aggregating traintasks. (traintasks -> aggregatetask)"""

    number_of_traintasks_to_aggregate = 3

    train_data_sample_inputs = default_dataset.train_data_sample_inputs[:number_of_traintasks_to_aggregate]

    spec = factory.create_function(FunctionCategory.simple)
    function = client.add_function(spec)

    spec = factory.create_function(FunctionCategory.predict)
    predict_function = client.add_function(spec)

    # add traintasks
    traintask_keys = []
    for data_sample_input in train_data_sample_inputs:
        spec = factory.create_traintask(
            function=function, inputs=default_dataset.opener_input + [data_sample_input], worker=worker
        )
        traintask = client.add_task(spec)
        traintask_keys.append(traintask.key)

    spec = factory.create_function(FunctionCategory.aggregate)
    aggregate_function = client.add_function(spec)

    spec = factory.create_aggregatetask(
        function=aggregate_function,
        worker=client.organization_id,
        inputs=FLTaskInputGenerator.trains_to_aggregate(traintask_keys),
    )
    aggregatetask = client.add_task(spec)
    assert (
        len([i for i in aggregatetask.inputs if i.identifier == InputIdentifiers.shared])
        == number_of_traintasks_to_aggregate
    )

    spec = factory.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.aggregate_to_predict(aggregatetask.key),
        worker=worker,
    )
    predicttask = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    client.wait_task(predicttask.key, raise_on_failure=True)

    spec = factory.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask.key),
        worker=worker,
    )
    testtask = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    client.wait_task(testtask.key, raise_on_failure=True)


@pytest.mark.slow
def test_aggregatetask_chained(factory, client, default_dataset, worker):
    """Execution of 2 chained aggregatetask (traintask -> aggregatetask -> aggregatetask)."""

    number_of_traintasks_to_aggregate = 1

    train_data_sample_input = default_dataset.train_data_sample_inputs[:1]

    spec = factory.create_function(FunctionCategory.simple)
    function = client.add_function(spec)

    # add traintasks
    spec = factory.create_traintask(
        function=function,
        inputs=default_dataset.opener_input + train_data_sample_input,
        worker=worker,
    )
    traintask = client.add_task(spec)

    spec = factory.create_function(FunctionCategory.aggregate)
    aggregate_function = client.add_function(spec)

    # add first layer of aggregatetasks
    spec = factory.create_aggregatetask(
        function=aggregate_function,
        worker=client.organization_id,
        inputs=FLTaskInputGenerator.trains_to_aggregate([traintask.key]),
    )

    aggregatetask_1 = client.add_task(spec)
    assert (
        len([i for i in aggregatetask_1.inputs if i.identifier == InputIdentifiers.shared])
        == number_of_traintasks_to_aggregate
    )

    # add second layer of aggregatetask
    spec = factory.create_aggregatetask(
        function=aggregate_function,
        worker=client.organization_id,
        inputs=FLTaskInputGenerator.trains_to_aggregate([aggregatetask_1.key]),
    )

    aggregatetask_2 = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    client.wait_task(aggregatetask_2.key, raise_on_failure=True)
    assert aggregatetask_2.error_type is None
    assert len([i for i in aggregatetask_2.inputs if i.identifier == InputIdentifiers.shared]) == 1


@pytest.mark.slow
def test_aggregatetask_traintask(factory, client, default_dataset, worker):
    """Execution of traintask after an aggregatetask (traintasks -> aggregatetask -> traintasks)"""

    number_of_traintasks = 2

    train_data_sample_inputs = default_dataset.train_data_sample_inputs[:number_of_traintasks]
    train_data_sample_input_1 = train_data_sample_inputs[0]
    train_data_sample_input_2 = train_data_sample_inputs[1]

    spec = factory.create_function(FunctionCategory.simple)
    function = client.add_function(spec)

    # add first part of the traintasks
    spec = factory.create_traintask(
        function=function,
        inputs=default_dataset.opener_input + [train_data_sample_input_1],
        worker=worker,
    )
    traintask_1 = client.add_task(spec)

    spec = factory.create_function(FunctionCategory.aggregate)
    aggregate_function = client.add_function(spec)

    # add aggregatetask
    spec = factory.create_aggregatetask(
        function=aggregate_function,
        worker=client.organization_id,
        inputs=FLTaskInputGenerator.trains_to_aggregate([traintask_1.key]),
    )
    aggregatetask = client.add_task(spec)
    assert len([i for i in aggregatetask.inputs if i.identifier == InputIdentifiers.shared]) == 1

    # add second part of the traintasks
    spec = factory.create_traintask(
        function=function,
        inputs=default_dataset.opener_input
        + [train_data_sample_input_2]
        + FLTaskInputGenerator.trains_to_train([aggregatetask.key]),
        worker=worker,
    )

    traintask_2 = client.add_task(spec)
    # `raises = True`, will fail if task not successful
    traintask_2 = client.wait_task(traintask_2.key, raise_on_failure=True)

    assert traintask_2.status == ComputeTaskStatus.done
    assert traintask_2.error_type is None


@pytest.mark.slow
@pytest.mark.remote_only
def test_composite_traintask_2_organizations_to_composite_traintask(factory, clients, default_datasets, workers):
    """A composite traintask which take as input a composite traintask (input_head_model) from
    organization 1 and another composite traintask (inpute_trunk_model) from organization 2
    """

    spec = factory.create_function(FunctionCategory.composite)
    composite_function = clients[0].add_function(spec)

    # composite traintasks on organization 1 and organization 2
    composite_traintask_keys = []
    for index, dataset in enumerate(default_datasets):
        spec = factory.create_composite_traintask(
            function=composite_function,
            inputs=dataset.opener_input + dataset.train_data_sample_inputs[:1],
            outputs=FLTaskOutputGenerator.composite_traintask(
                shared_authorized_ids=[c.organization_id for c in clients],
                local_authorized_ids=[dataset.owner],
            ),
            worker=workers[index],
        )
        composite_traintask_key = clients[0].add_task(spec).key
        composite_traintask_keys.append(composite_traintask_key)

    spec = factory.create_composite_traintask(
        function=composite_function,
        inputs=default_datasets[0].train_data_inputs
        + FLTaskInputGenerator.composite_to_composite(composite_traintask_keys[0], composite_traintask_keys[1]),
        rank=1,
        outputs=FLTaskOutputGenerator.composite_traintask(
            shared_authorized_ids=[c.organization_id for c in clients],
            local_authorized_ids=[dataset.owner],
        ),
        worker=workers[0],
    )
    composite_traintask = clients[0].add_task(spec)
    # `raises = True`, will fail if task not successful
    clients[0].wait_task(composite_traintask.key, raise_on_failure=True)


@pytest.mark.slow
def test_aggregate_composite_traintasks(factory, network, clients, default_datasets, default_metrics, workers):
    """Do 2 rounds of composite traintasks aggregations on multiple organizations.

    Compute plan details:

    Round 1:
    - Create 2 composite traintasks executed on two datasets located on organization 1 and
      organization 2.
    - Create an aggregatetask on organization 1, aggregating the two previous composite
      traintasks (trunk models aggregation).

    Round 2:
    - Create 2 composite traintasks executed on each organizations that depend on: the
      aggregated task and the previous composite traintask executed on this organization. That
      is to say, the previous round aggregated trunk models from all organizations and the
      previous round head model from this organization.
    - Create an aggregatetask on organization 1, aggregating the two previous composite
      traintasks (similar to round 1 aggregatetask).
    - Create a testtask for each previous composite traintasks and aggregate task
      created during this round.

    (optional) if the option "enable_intermediate_model_removal" is True:
    - Since option "enable_intermediate_model_removal" is True, the aggregate model created on round 1 should
      have been deleted from the backend after round 2 has completed.
    - Create a traintask that depends on the aggregate task created on round 1. Ensure that it fails to start.

    This test refers to the model composition use case.
    """

    aggregate_worker = clients[0].organization_id
    number_of_rounds = 2

    # register functions on first organization
    spec = factory.create_function(FunctionCategory.composite)
    composite_function = clients[0].add_function(spec)
    spec = factory.create_function(FunctionCategory.aggregate)
    aggregate_function = clients[0].add_function(spec)
    spec = factory.create_function(FunctionCategory.predict)
    predict_function = clients[0].add_function(spec)
    spec = factory.create_function(FunctionCategory.predict_composite)
    predict_function_composite = clients[0].add_function(spec)

    # launch execution
    previous_aggregatetask_key = None
    previous_composite_traintask_keys = []

    for round_ in range(number_of_rounds):
        # create composite traintask on each organization
        composite_traintask_keys = []
        for index, dataset in enumerate(default_datasets):
            if previous_aggregatetask_key:
                input_models = FLTaskInputGenerator.composite_to_local(
                    previous_composite_traintask_keys[index]
                ) + FLTaskInputGenerator.aggregate_to_shared(previous_aggregatetask_key)

            else:
                input_models = []

            spec = factory.create_composite_traintask(
                function=composite_function,
                inputs=[dataset.train_data_sample_inputs[0 + round_]] + dataset.opener_input + input_models,
                outputs=FLTaskOutputGenerator.composite_traintask(
                    shared_authorized_ids=[c.organization_id for c in clients],
                    local_authorized_ids=[dataset.owner],
                ),
                worker=workers[index],
            )

            t = clients[0].add_task(spec)
            clients[0].wait_task(t.key, raise_on_failure=True)
            composite_traintask_keys.append(t.key)

        # create aggregate on its organization
        spec = factory.create_aggregatetask(
            function=aggregate_function,
            worker=aggregate_worker,
            inputs=FLTaskInputGenerator.composites_to_aggregate(composite_traintask_keys),
        )
        aggregatetask = clients[0].add_task(spec)
        clients[0].wait_task(aggregatetask.key, raise_on_failure=True)

        # save state of round
        previous_aggregatetask_key = aggregatetask.key
        previous_composite_traintask_keys = composite_traintask_keys

    # last round: create associated testtask for composite and aggregate
    for index, (traintask_key, metric, dataset) in enumerate(
        zip(previous_composite_traintask_keys, default_metrics, default_datasets)
    ):
        spec = factory.create_predicttask(
            function=predict_function_composite,
            inputs=dataset.test_data_inputs + FLTaskInputGenerator.composite_to_predict(traintask_key),
            worker=workers[index],
        )
        predicttask = clients[0].add_task(spec)
        clients[0].wait_task(predicttask.key, raise_on_failure=True)

        spec = factory.create_testtask(
            function=metric,
            inputs=dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask.key),
            worker=workers[index],
        )
        testtask = clients[0].add_task(spec)
        clients[0].wait_task(testtask.key, raise_on_failure=True)
        # y_true: [20], y_pred: [52.0], result: 32.0
        performance = clients[0].get_task_output_asset(testtask.key, OutputIdentifiers.performance)
        assert performance.asset == pytest.approx(32 + index)

    spec = factory.create_predicttask(
        function=predict_function,
        inputs=default_datasets[0].test_data_inputs
        + FLTaskInputGenerator.aggregate_to_predict(previous_aggregatetask_key),
        worker=workers[0],
    )
    predicttask = clients[0].add_task(spec)
    clients[0].wait_task(predicttask.key, raise_on_failure=True)

    spec = factory.create_testtask(
        function=default_metrics[0],
        inputs=default_datasets[0].test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask.key),
        worker=workers[0],
    )
    testtask = clients[0].add_task(spec)
    clients[0].wait_task(testtask.key, raise_on_failure=True)
    # y_true: [20], y_pred: [28.0], result: 8.0
    performance = clients[0].get_task_output_asset(testtask.key, OutputIdentifiers.performance)
    assert performance.asset == pytest.approx(8)

    if network.options.enable_model_download:
        # Optional (if "enable_model_download" is True): ensure we can export out-models.
        #
        # - One out-model download is not proxified (direct download)
        # - One out-model download is proxified (as it belongs to another org)
        for key in previous_composite_traintask_keys:
            assert clients[0].get_model_content_from_task(key, identifier=OutputIdentifiers.shared) == b'{"value": 2.8}'

    if network.options.enable_intermediate_model_removal:
        # Optional (if "enable_intermediate_model_removal" is True): ensure the aggregatetask of round 1 has been
        # deleted.
        #
        # We do this by creating a new traintask that depends on the deleted aggregatatask, and ensuring that starting
        # the traintask fails.
        #
        # Ideally it would be better to try to do a request "as a backend" to get the deleted model. This would be
        # closer to what we want to test and would also check that this request is correctly handled when the model
        # has been deleted. Here, we cannot know for sure the failure reason. Unfortunately this cannot be done now
        # as the username/password are not available in the settings files.

        client = clients[0]
        dataset = default_datasets[0]
        function = client.add_function(spec)

        spec = factory.create_traintask(function=function, inputs=dataset.train_data_inputs, worker=workers[0])
        traintask = client.add_task(spec)
        traintask = client.wait_task(traintask.key)
        assert traintask.status == ComputeTaskStatus.failed
        assert traintask.error_type == substra.sdk.models.TaskErrorType.execution


@pytest.mark.remote_only
def test_use_data_sample_located_in_shared_path(factory, network, client, organization_cfg, default_metric, worker):
    if not organization_cfg.shared_path:
        pytest.skip("requires a shared path")

    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    spec = factory.create_data_sample(datasets=[dataset])
    spec.move_data_to_server(organization_cfg.shared_path, network.options.minikube)
    data_sample_key = client.add_data_sample(spec, local=False)  # should not raise

    dataset = AugmentedDataset(client.get_dataset(dataset.key))
    dataset.set_train_test_dasamples(
        train_data_sample_keys=[data_sample_key],
    )

    spec = factory.create_function(FunctionCategory.simple)
    function = client.add_function(spec)

    spec = factory.create_function(FunctionCategory.predict)
    predict_function = client.add_function(spec)

    spec = factory.create_traintask(function=function, inputs=dataset.train_data_inputs, worker=worker)
    traintask = client.add_task(spec)
    traintask = client.wait_task(traintask.key, raise_on_failure=True)
    assert traintask.error_type is None

    # Raises an exception if the output asset have not been created
    client.get_task_output_asset(traintask.key, OutputIdentifiers.shared)

    # create testtask
    spec = factory.create_predicttask(
        function=predict_function, traintask=traintask, dataset=dataset, data_samples=[data_sample_key], worker=worker
    )
    predicttask = client.add_task(spec)
    predicttask = client.wait_task(predicttask.key, raise_on_failure=True)
    assert predicttask.error_type is None

    spec = factory.create_testtask(
        function=default_metric, predicttask=predicttask, dataset=dataset, data_samples=[data_sample_key], worker=worker
    )
    testtask = client.add_task(spec)
    testtask = client.wait_task(testtask.key, raise_on_failure=True)
    assert testtask.error_type is None
    performance = client.get_task_output_asset(testtask.key, OutputIdentifiers.performance)
    assert performance.asset == pytest.approx(2)


@pytest.mark.subprocess_skip
def test_user_creates_model_folder(factory, client, default_dataset, worker):
    """Check that the model folder is not overwritten by substra"""
    substratools_git_ref = os.getenv("SUBSTRATOOLS_GIT_REF", "main")
    dockerfile = (
        f"FROM {factory.default_tools_image}\n"
        "RUN apt-get update -y && apt-get install -y git\nRUN python3 -m pip install -U pip\n"
        f"RUN python3 -m pip install git+https://github.com/Substra/substra-tools.git@{substratools_git_ref}\n"
        "COPY function.py .\nRUN mkdir model\n"
        + 'RUN echo \'{"name":"Jane"}\' >> model/model'
        + '\nENTRYPOINT ["python3", "function.py", "--function-name", "train"]\n'
    )
    function_script = f"""
import json
import substratools as tools

from pathlib import Path

@tools.register
def train(inputs, outputs, task_properties):
    model_path = Path.cwd() / 'model' / 'model'
    assert model_path.is_file()
    loaded = json.loads(model_path.read_text())
    assert loaded == {{'name':'Jane'}}
    save_model(dict(), outputs['{OutputIdentifiers.shared.value}'])


@tools.register
def predict(inputs, outputs, task_properties):
    save_predictions(None, outputs['{OutputIdentifiers.predictions.value}'])

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
    spec = factory.create_function(FunctionCategory.simple, py_script=function_script, dockerfile=dockerfile)
    function = client.add_function(spec)
    spec = factory.create_traintask(function=function, inputs=default_dataset.train_data_inputs, worker=worker)
    traintask = client.add_task(spec)
    client.wait_task(traintask.key, raise_on_failure=True)


WRITE_TO_HOME_DIRECTORY_FUNCTION = f"""
import json
import substratools as tools


@tools.register
def train(inputs, outputs, task_properties):

    from pathlib import Path
    with open(f"{{str(Path.home())}}/foo", "w") as f:
        f.write("test")

    save_model({{'value': 42 }}, outputs['{OutputIdentifiers.shared.value}'])

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


@pytest.mark.subprocess_skip
def test_write_to_home_directory(factory, client, default_dataset, worker):
    """The function writes to the home directory (~/foo)"""

    spec = factory.create_function(FunctionCategory.simple, WRITE_TO_HOME_DIRECTORY_FUNCTION)
    function = client.add_function(spec)
    spec = factory.create_traintask(function=function, inputs=default_dataset.train_data_inputs, worker=worker)
    traintask = client.add_task(spec)
    traintask = client.wait_task(traintask.key, raise_on_failure=True)

    assert traintask.error_type is None


@pytest.mark.remote_only
def test_function_build_failure_error_orchestrator_task_creation(factory, client, network, default_dataset_1, worker):
    """Check if transient outputs are correctly deleted when a build failure is propagated to a task created after the
    failure."""

    dockerfile = factory.default_function_dockerfile(
        method_name=sbt.factory.DEFAULT_FUNCTION_NAME[FunctionCategory.simple]
    )
    dockerfile += "\nRUN invalid_command"
    spec = factory.create_function(category=FunctionCategory.simple, dockerfile=dockerfile)
    function_1 = network.clients[0].add_function(spec)

    network.clients[0].wait_function(function_1.key, raise_on_failure=False)

    task_1 = factory.create_traintask(
        function=function_1,
        inputs=default_dataset_1.train_data_inputs,
        outputs=FLTaskOutputGenerator.traintask(transient=True),
        worker=worker,
    )
    network.clients[0].add_task(task_1)

    task_2 = factory.create_traintask(
        function=function_1,
        inputs=default_dataset_1.train_data_inputs + FLTaskInputGenerator.trains_to_train([task_1.key]),
        worker=worker,
    )
    # Needed otherwise cleaning of transient output oin final task are not cleaned
    network.clients[1].add_task(task_2)

    try:
        # Calling this endpoint will do a synchronous call to the orchestrator
        network.clients[0].organization_info()
    except InternalServerError:
        pytest.fail(
            "Trying to remove transient output assets of a task created in a non-succesful end status. These output "
            "assets were not created"
        )


@pytest.mark.remote_only
def test_function_build_failure_error_orchestrator_task_update(factory, client, network, default_dataset_1, worker):
    """Check if transient outputs are correctly deleted when a build failure is propagated to a task created before the
    failure."""

    dockerfile = factory.default_function_dockerfile(
        method_name=sbt.factory.DEFAULT_FUNCTION_NAME[FunctionCategory.simple]
    )
    dockerfile += "\nRUN invalid_command"
    spec = factory.create_function(category=FunctionCategory.simple, dockerfile=dockerfile)
    function_1 = network.clients[0].add_function(spec)

    task_1 = factory.create_traintask(
        function=function_1,
        inputs=default_dataset_1.train_data_inputs,
        outputs=FLTaskOutputGenerator.traintask(transient=True),
        worker=worker,
    )
    network.clients[0].add_task(task_1)
    network.clients[0].wait_task(task_1.key, raise_on_failure=False)

    # Needed otherwise cleaning of transient output oin final task are not cleaned
    task_2 = factory.create_traintask(
        function=function_1,
        inputs=default_dataset_1.train_data_inputs + FLTaskInputGenerator.trains_to_train([task_1.key]),
        worker=worker,
    )
    network.clients[1].add_task(task_2)

    try:
        # Calling this endpoint will do a synchronous call to the orchestrator
        network.clients[0].organization_info()
    except InternalServerError:
        pytest.fail(
            "Trying to remove transient output assets of a task created in a non-succesful end status. These output "
            "assets were not created"
        )
