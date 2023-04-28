import uuid

import pytest
import substra
from substra.sdk import models

import substratest as sbt
from substratest.client import Client
from substratest.factory import AssetsFactory
from substratest.fl_interface import FLTaskInputGenerator
from substratest.fl_interface import FLTaskOutputGenerator
from substratest.fl_interface import FunctionCategory
from substratest.fl_interface import InputIdentifiers
from substratest.fl_interface import OutputIdentifiers


def test_compute_plan_simple(
    factory,
    client_1,
    client_2,
    default_dataset_1,
    default_dataset_2,
    default_metrics,
    channel,
    workers,
):
    """Execution of a compute plan containing multiple traintasks:
    - 1 traintask executed on organization 1
    - 1 traintask executed on organization 2
    - 1 traintask executed on organization 1 depending on previous traintasks
    - 1 predicttask executed on organization 1 depending on previous traintasks
    - 1 testtask executed on organization 1 depending on the last predicttask and on one metric
    """

    cp_key = str(uuid.uuid4())

    simple_function_spec = factory.create_function(FunctionCategory.simple)
    simple_function_2 = client_2.add_function(simple_function_spec)

    predict_function_spec = factory.create_function(FunctionCategory.predict)
    predict_function_2 = client_2.add_function(predict_function_spec)

    channel.wait_for_asset_synchronized(simple_function_2)

    # create compute plan
    cp_spec = factory.create_compute_plan(
        key=cp_key,
        tag="foo",
        name="Bar",
        metadata={"foo": "bar"},
    )

    traintask_spec_1 = cp_spec.create_traintask(
        function=simple_function_2, inputs=default_dataset_1.train_data_inputs, metadata=None, worker=workers[0]
    )

    traintask_spec_2 = cp_spec.create_traintask(
        function=simple_function_2, inputs=default_dataset_2.train_data_inputs, metadata={}, worker=workers[1]
    )

    traintask_spec_3 = cp_spec.create_traintask(
        function=simple_function_2,
        inputs=default_dataset_1.train_data_inputs
        + FLTaskInputGenerator.trains_to_train([traintask_spec_1.task_id, traintask_spec_2.task_id]),
        metadata={"foo": "bar"},
        worker=workers[0],
    )

    predicttask_spec_3 = cp_spec.create_predicttask(
        function=predict_function_2,
        inputs=default_dataset_1.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_spec_3.task_id),
        metadata={"foo": "bar"},
        worker=workers[0],
    )

    testtask_spec = cp_spec.create_testtask(
        function=default_metrics[0],
        inputs=default_dataset_1.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask_spec_3.task_id),
        metadata={"foo": "bar"},
        worker=workers[0],
    )

    # submit compute plan and wait for it to complete
    cp_added = client_1.add_compute_plan(cp_spec)

    cp = client_1.wait(cp_added)
    assert cp.key == cp_key
    assert cp.tag == "foo"
    assert cp.metadata == {"foo": "bar"}
    assert cp.task_count == cp.done_count == 5
    assert cp.todo_count == cp.waiting_count == cp.doing_count == cp.canceled_count == cp.failed_count == 0
    assert cp.end_date is not None
    assert cp.duration is not None

    tasks = client_1.list_compute_plan_tasks(cp.key)
    assert len(tasks) == 5

    # check all tasks are done and check they have been executed on the expected organization
    for t in tasks:
        assert t.status == models.Status.done
        assert t.start_date is not None
        assert t.end_date is not None

    full_tasks = [client_1.get_task(t.key) for t in tasks]

    traintask_1 = [t for t in full_tasks if t.key == traintask_spec_1.task_id][0]
    traintask_2 = [t for t in full_tasks if t.key == traintask_spec_2.task_id][0]
    traintask_3 = [t for t in full_tasks if t.key == traintask_spec_3.task_id][0]

    assert len([i for i in traintask_3.inputs if i.identifier == InputIdentifiers.models]) == 2

    predicttask = [t for t in full_tasks if t.key == predicttask_spec_3.task_id][0]
    testtask = [t for t in full_tasks if t.key == testtask_spec.task_id][0]

    # check tasks metadata
    assert traintask_1.metadata == {}
    assert traintask_2.metadata == {}
    assert traintask_3.metadata == {"foo": "bar"}
    assert predicttask.metadata == {"foo": "bar"}
    assert testtask.metadata == {"foo": "bar"}

    # check tasks rank
    assert traintask_1.rank == 0
    assert traintask_2.rank == 0
    assert traintask_3.rank == 1
    assert predicttask.rank == 2
    assert testtask.rank == predicttask.rank + 1

    # check testtask perfs
    assert len(testtask.outputs) == 1
    assert testtask.outputs[OutputIdentifiers.performance].value == pytest.approx(4)

    # check compute plan perfs
    performances = client_1.get_performances(cp.key)
    assert all(len(val) == 1 for val in performances.dict().values())
    assert testtask.outputs[OutputIdentifiers.performance].value == performances.performance[0]

    # XXX as the first two tasks have the same rank, there is currently no way to know
    #     which one will be returned first
    workers_rank_0 = {traintask_1.worker, traintask_2.worker}
    assert workers_rank_0 == {client_1.organization_id, client_2.organization_id}
    assert traintask_3.worker == client_1.organization_id
    assert predicttask.worker == client_1.organization_id
    assert testtask.worker == client_1.organization_id

    # check mapping
    traintask_id_1 = traintask_spec_1.task_id
    traintask_id_2 = traintask_spec_2.task_id
    traintask_id_3 = traintask_spec_3.task_id
    generated_ids = [traintask_id_1, traintask_id_2, traintask_id_3]
    rank_0_traintask_keys = [traintask_1.key, traintask_2.key]
    assert set(generated_ids) == {traintask_id_1, traintask_id_2, traintask_id_3}
    assert set(rank_0_traintask_keys) == {traintask_id_1, traintask_id_2}
    assert traintask_3.key == traintask_id_3


@pytest.mark.slow
def test_compute_plan_single_client_success(factory, client, default_dataset, default_metric, worker):
    """A compute plan with 3 traintasks and 3 associated testtasks"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintask + testtask
    # 2. traintask + testtask
    # 3. traintask + testtask

    data_sample_1_input, data_sample_2_input, data_sample_3_input, _ = default_dataset.train_data_sample_inputs

    simple_function_spec = factory.create_function(FunctionCategory.simple)
    simple_function = client.add_function(simple_function_spec)

    predict_function_spec = factory.create_function(FunctionCategory.predict)
    predict_function = client.add_function(predict_function_spec)

    cp_spec = factory.create_compute_plan()

    traintask_spec_1 = cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input + [data_sample_1_input],
        worker=worker,
    )

    predicttask_spec_1 = cp_spec.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_spec_1.task_id),
        worker=worker,
    )

    cp_spec.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask_spec_1.task_id),
        worker=worker,
    )

    traintask_spec_2 = cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input
        + [data_sample_2_input]
        + FLTaskInputGenerator.trains_to_train([traintask_spec_1.task_id]),
        worker=worker,
    )
    predicttask_spec_2 = cp_spec.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_spec_2.task_id),
        worker=worker,
    )
    cp_spec.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask_spec_2.task_id),
        worker=worker,
    )

    traintask_spec_3 = cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input
        + [data_sample_3_input]
        + FLTaskInputGenerator.trains_to_train([traintask_spec_2.task_id]),
        worker=worker,
    )
    predicttask_spec_3 = cp_spec.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_spec_3.task_id),
        worker=worker,
    )
    cp_spec.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask_spec_3.task_id),
        worker=worker,
    )

    # Submit compute plan and wait for it to complete
    cp_added = client.add_compute_plan(cp_spec)
    cp = client.wait(cp_added)

    assert cp.status == "PLAN_STATUS_DONE"
    assert cp.end_date is not None
    assert cp.duration is not None

    # All the train/test tasks should succeed
    for t in (
        client.list_compute_plan_tasks(cp.key)
        + client.list_compute_plan_tasks(cp.key)
        + client.list_compute_plan_tasks(cp.key)
    ):
        assert t.status == models.Status.done


@pytest.mark.slow
def test_compute_plan_update(factory, client, default_dataset, default_metric, worker):
    """A compute plan with 3 traintasks and 3 associated testtasks.

    This is done by sending 3 requests (one create and two updates).
    """

    data_sample_1_input, data_sample_2_input, data_sample_3_input, _ = default_dataset.train_data_sample_inputs

    simple_function_spec = factory.create_function(FunctionCategory.simple)
    simple_function = client.add_function(simple_function_spec)

    predict_function_spec = factory.create_function(FunctionCategory.predict)
    predict_function = client.add_function(predict_function_spec)

    # Create a compute plan with traintask + testtask

    cp_spec = factory.create_compute_plan()
    traintask_spec_1 = cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input + [data_sample_1_input],
        worker=worker,
    )

    predicttask_spec_1 = cp_spec.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_spec_1.task_id),
        worker=worker,
    )

    cp_spec.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask_spec_1.task_id),
        worker=worker,
    )
    cp = client.add_compute_plan(cp_spec, auto_batching=True, batch_size=1)

    # Update compute plan with traintask + testtask

    cp_spec = factory.add_compute_plan_tasks(cp)
    traintask_spec_2 = cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input
        + [data_sample_2_input]
        + FLTaskInputGenerator.trains_to_train([traintask_spec_1.task_id]),
        metadata={"foo": "bar"},
        worker=worker,
    )
    predicttask_spec_2 = cp_spec.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_spec_2.task_id),
        metadata={"foo": "bar"},
        worker=worker,
    )
    testtask_spec_2 = cp_spec.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask_spec_2.task_id),
        metadata={"foo": "bar"},
        worker=worker,
    )
    cp = client.add_compute_plan_tasks(cp_spec, auto_batching=True, batch_size=1)

    # Update compute plan with traintask

    cp_spec = factory.add_compute_plan_tasks(cp)
    traintask_spec_3 = cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input
        + [data_sample_3_input]
        + FLTaskInputGenerator.trains_to_train([traintask_spec_2.task_id]),
        worker=worker,
    )
    predicttask_spec_3 = cp_spec.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_spec_3.task_id),
        worker=worker,
    )
    cp = client.add_compute_plan_tasks(cp_spec)

    # Update compute plan with testtask

    cp_spec = factory.add_compute_plan_tasks(cp)
    cp_spec.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask_spec_3.task_id),
        worker=worker,
    )
    cp = client.add_compute_plan_tasks(cp_spec)

    # All the train/test tasks should succeed
    cp_added = client.get_compute_plan(cp.key)
    cp = client.wait(cp_added)
    tasks = client.list_compute_plan_tasks(cp.key)
    assert len(tasks) == 9
    for t in tasks:
        assert t.status == models.Status.done

    # Check tasks metadata
    traintask = client.get_task(traintask_spec_2.task_id)
    predicttask = client.get_task(predicttask_spec_2.task_id)
    testtask = client.get_task(testtask_spec_2.task_id)

    assert traintask.metadata == {"foo": "bar"}
    assert predicttask.metadata == {"foo": "bar"}
    assert testtask.metadata == {"foo": "bar"}


@pytest.mark.slow
@pytest.mark.remote_only
def test_compute_plan_single_client_failure(factory, client, default_dataset, default_metric, worker):
    """In a compute plan with 3 traintasks, failing the root traintask
    should cancel its descendents and the associated testtasks"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintask + predicttask + testtask
    # 2. traintask + predicttask + testtask
    # 3. traintask + predicttask + testtask
    #
    # Intentionally use an invalid (broken) function.

    data_sample_1_input, data_sample_2_input, data_sample_3_input, _ = default_dataset.train_data_sample_inputs

    simple_function_spec = factory.create_function(
        FunctionCategory.simple, py_script=sbt.factory.INVALID_FUNCTION_SCRIPT
    )
    simple_function = client.add_function(simple_function_spec)

    predict_function_spec = factory.create_function(FunctionCategory.predict)
    predict_function = client.add_function(predict_function_spec)

    cp_spec = factory.create_compute_plan()

    traintask_spec_1 = cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input + [data_sample_1_input],
        worker=worker,
    )
    predicttask_spec_1 = cp_spec.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_spec_1.task_id),
        worker=worker,
    )
    cp_spec.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask_spec_1.task_id),
        worker=worker,
    )

    traintask_spec_2 = cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input
        + [data_sample_2_input]
        + FLTaskInputGenerator.trains_to_train([traintask_spec_1.task_id]),
        worker=worker,
    )
    predicttask_spec_2 = cp_spec.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_spec_2.task_id),
        worker=worker,
    )

    cp_spec.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask_spec_2.task_id),
        worker=worker,
    )

    traintask_spec_3 = cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input
        + [data_sample_3_input]
        + FLTaskInputGenerator.trains_to_train([traintask_spec_2.task_id]),
        worker=worker,
    )

    predicttask_spec_3 = cp_spec.create_predicttask(
        function=predict_function,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_spec_3.task_id),
        worker=worker,
    )
    cp_spec.create_testtask(
        function=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask_spec_3.task_id),
        worker=worker,
    )

    # Submit compute plan and wait for it to complete
    cp_added = client.add_compute_plan(cp_spec)
    cp = client.wait(cp_added, raises=False)

    assert cp.status == "PLAN_STATUS_FAILED"
    assert cp.end_date is not None
    assert cp.duration is not None


# FIXME: test_compute_plan_aggregate_composite_traintasks is too complex, consider refactoring
@pytest.mark.slow  # noqa: C901
def test_compute_plan_aggregate_composite_traintasks(  # noqa: C901
    factory,
    clients,
    default_datasets,
    default_metrics,
    workers,
):
    """
    Compute plan version of the `test_aggregate_composite_traintasks` method from `test_execution.py`
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
    previous_aggregatetask_spec = None
    previous_composite_traintask_specs = []

    cp_spec = factory.create_compute_plan()

    for round_ in range(number_of_rounds):
        # create composite traintask on each organization
        composite_traintask_specs = []
        for index, dataset in enumerate(default_datasets):
            if previous_aggregatetask_spec is not None:
                local_input = FLTaskInputGenerator.composite_to_local(previous_composite_traintask_specs[index].task_id)
                shared_input = FLTaskInputGenerator.aggregate_to_shared(previous_aggregatetask_spec.task_id)

            else:
                local_input = []
                shared_input = []

            spec = cp_spec.create_composite_traintask(
                composite_function=composite_function,
                inputs=dataset.opener_input
                + [dataset.train_data_sample_inputs[0 + round_]]
                + local_input
                + shared_input,
                outputs=FLTaskOutputGenerator.composite_traintask(
                    shared_authorized_ids=[client.organization_id for client in clients],
                    local_authorized_ids=[clients[index].organization_id],
                ),
                worker=workers[index],
            )

            composite_traintask_specs.append(spec)

        # create aggregate on its organization
        spec = cp_spec.create_aggregatetask(
            aggregate_function=aggregate_function,
            worker=aggregate_worker,
            inputs=FLTaskInputGenerator.composites_to_aggregate(
                [composite_traintask_spec.task_id for composite_traintask_spec in composite_traintask_specs]
            ),
        )

        # save state of round
        previous_aggregatetask_spec = spec
        previous_composite_traintask_specs = composite_traintask_specs

    # last round: create associated testtask
    for composite_traintask, dataset, metric, worker in zip(
        previous_composite_traintask_specs, default_datasets, default_metrics, workers
    ):
        spec = cp_spec.create_predicttask(
            function=predict_function_composite,
            inputs=dataset.test_data_inputs + FLTaskInputGenerator.composite_to_predict(composite_traintask.task_id),
            worker=worker,
        )
        cp_spec.create_testtask(
            function=metric,
            inputs=dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(spec.task_id),
            worker=worker,
        )

    predicttask_from_aggregate_spec = cp_spec.create_predicttask(
        function=predict_function,
        inputs=default_datasets[0].test_data_inputs
        + FLTaskInputGenerator.aggregate_to_predict(previous_aggregatetask_spec.task_id),
        worker=workers[0],
    )
    cp_spec.create_testtask(
        function=metric,
        inputs=default_datasets[0].test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttask_from_aggregate_spec.task_id),
        worker=workers[0],
    )

    cp_added = clients[0].add_compute_plan(cp_spec)
    cp = clients[0].wait(cp_added)

    tasks = clients[0].list_compute_plan_tasks(cp.key)

    for task in composite_traintask_specs:
        remote_task = clients[0].get_task(task.task_id)
        if len(task.inputs) > 2:
            assert (
                len(
                    [
                        i
                        for i in remote_task.inputs
                        if i.identifier == InputIdentifiers.local
                        and i.parent_task_key
                        == [x for x in task.inputs if x.identifier == InputIdentifiers.local][0].parent_task_key
                        and i.parent_task_output_identifier == InputIdentifiers.local
                    ]
                )
                == 1
            )
        print(task.inputs)
        if len(task.inputs) > 2:
            assert (
                len(
                    [
                        i
                        for i in remote_task.inputs
                        if i.identifier == InputIdentifiers.shared
                        and i.parent_task_key
                        == [x for x in task.inputs if x.identifier == InputIdentifiers.shared][0].parent_task_key
                        and i.parent_task_output_identifier == OutputIdentifiers.model
                    ]
                )
                == 1
            )

    for t in tasks:
        assert t.status == models.Status.done, t

    # Check that permissions were correctly set
    for task_id in [ct.task_id for ct in composite_traintask_specs]:
        task = clients[0].get_task(task_id)
        trunk = task.outputs[OutputIdentifiers.shared].value
        assert len(trunk.permissions.process.authorized_ids) == len(clients)


def test_compute_plan_circular_dependency_failure(factory, client, default_dataset, worker):
    spec = factory.create_function(FunctionCategory.simple)
    function = client.add_function(spec)

    cp_spec = factory.create_compute_plan()

    traintask_spec_1 = cp_spec.create_traintask(
        inputs=default_dataset.train_data_inputs,
        function=function,
        worker=worker,
    )

    traintask_spec_2 = cp_spec.create_traintask(
        inputs=default_dataset.train_data_inputs,
        function=function,
        worker=worker,
    )

    traintask_spec_1.inputs.append(FLTaskInputGenerator.trains_to_train([traintask_spec_2.task_id])[0])
    traintask_spec_2.inputs.append(FLTaskInputGenerator.trains_to_train([traintask_spec_1.task_id])[0])

    with pytest.raises(substra.exceptions.InvalidRequest) as e:
        client.add_compute_plan(cp_spec)

    assert "missing dependency among inModels IDs" in str(e.value)


@pytest.mark.slow
@pytest.mark.remote_only
def test_execution_compute_plan_canceled(factory, client, default_dataset, cfg, worker):
    # XXX A canceled compute plan can be done if it is canceled while it last tasks
    #     are executing on the workers. The compute plan status will in this case change
    #     from canceled to done.
    #     To increase our confidence that the compute plan won't be done, we create a
    #     compute plan with a large amount of tasks.
    nb_traintasks = 32

    spec = factory.create_function(FunctionCategory.simple)
    function = client.add_function(spec)

    cp_spec = factory.create_compute_plan()
    previous_traintask = None
    inputs = default_dataset.opener_input + default_dataset.train_data_sample_inputs[:1]

    for _ in range(nb_traintasks):
        input_models = (
            FLTaskInputGenerator.trains_to_train([previous_traintask.task_id]) if previous_traintask is not None else []
        )
        previous_traintask = cp_spec.create_traintask(
            function=function,
            inputs=inputs + input_models,
            worker=worker,
        )

    cp = client.add_compute_plan(cp_spec)

    # wait the first traintask to be executed to ensure that the compute plan is launched
    # and tasks are scheduled in the celery workers
    first_traintask = [t for t in client.list_compute_plan_tasks(cp.key) if t.rank == 0][0]
    first_traintask = client.wait(first_traintask)
    assert first_traintask.status == models.Status.done

    client.cancel_compute_plan(cp.key)
    # as cancel request do not directly update localrep we need to wait for the sync
    cp = client.wait(cp, raises=False, timeout=cfg.options.organization_sync_timeout)
    assert cp.status == models.ComputePlanStatus.canceled
    assert cp.end_date is not None
    assert cp.duration is not None

    # check that the status of the done task as not been updated
    first_traintask = [t for t in client.list_compute_plan_tasks(cp.key) if t.rank == 0][0]
    assert first_traintask.status == models.Status.done


@pytest.mark.slow
@pytest.mark.remote_only
def test_compute_plan_no_batching(factory, client, default_dataset, worker):
    spec = factory.create_function(FunctionCategory.simple)
    function = client.add_function(spec)

    # Create a compute plan
    cp_spec = factory.create_compute_plan()
    traintask_spec_1 = cp_spec.create_traintask(
        function=function,
        inputs=default_dataset.opener_input + default_dataset.train_data_sample_inputs[:1],
        worker=worker,
    )
    cp_added = client.add_compute_plan(cp_spec, auto_batching=False)
    cp = client.wait(cp_added)

    traintasks = client.list_compute_plan_tasks(cp.key)
    assert len(traintasks) == 1
    assert all([task_.status == models.Status.done for task_ in traintasks])

    # Update the compute plan
    cp_spec = factory.add_compute_plan_tasks(cp)
    cp_spec.create_traintask(
        function=function,
        inputs=default_dataset.opener_input
        + default_dataset.train_data_sample_inputs[1:2]
        + FLTaskInputGenerator.trains_to_train([traintask_spec_1.task_id]),
        metadata={"foo": "bar"},
        worker=worker,
    )
    cp_added = client.add_compute_plan_tasks(cp_spec, auto_batching=False)
    cp = client.wait(cp_added)

    traintasks = client.list_compute_plan_tasks(cp.key)
    assert len(traintasks) == 2
    assert all([task_.status == models.Status.done for task_ in traintasks])


@pytest.mark.slow
@pytest.mark.remote_only
def test_compute_plan_transient_outputs(factory: AssetsFactory, client: Client, default_dataset, worker: str):
    """
    Create a simple compute plan with tasks using transient inputs, check if the flag is set
    """
    data_sample_1_input, data_sample_2_input, _, _ = default_dataset.train_data_sample_inputs

    # Register the Function
    simple_function_spec = factory.create_function(FunctionCategory.simple)
    simple_function = client.add_function(simple_function_spec)

    cp_spec = factory.create_compute_plan()
    traintask_spec_1 = cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input + [data_sample_1_input],
        outputs=FLTaskOutputGenerator.traintask(transient=True),
        worker=worker,
    )

    cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input
        + [data_sample_2_input]
        + FLTaskInputGenerator.trains_to_train([traintask_spec_1.task_id]),
        worker=worker,
    )

    cp_added = client.add_compute_plan(cp_spec)
    client.wait(cp_added)

    traintask_1 = client.get_task(traintask_spec_1.task_id)
    assert traintask_1.outputs[OutputIdentifiers.model].is_transient is True

    # Validate that the transient model is properly deleted
    model = client.get_task_models(traintask_spec_1.task_id)[0]
    client.wait_model_deletion(model.key)

    # Validate that we can't create a new task that use this model
    traintask_spec_3 = factory.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input
        + [data_sample_2_input]
        + FLTaskInputGenerator.trains_to_train([traintask_spec_1.task_id]),
        worker=worker,
    )

    with pytest.raises(substra.exceptions.InvalidRequest) as err:
        client.add_task(traintask_spec_3)

    assert "has been disabled" in str(err.value)


@pytest.mark.slow
@pytest.mark.remote_only
def test_compute_task_profile(factory, client, default_dataset, worker):
    """
    Creates a simple task to check that tasks profiles are correctly produced
    """
    data_sample_1_input, _, _, _ = default_dataset.train_data_sample_inputs
    simple_function_spec = factory.create_function(FunctionCategory.simple)
    simple_function = client.add_function(simple_function_spec)

    cp_spec = factory.create_compute_plan()
    traintask_spec_1 = cp_spec.create_traintask(
        function=simple_function,
        inputs=default_dataset.opener_input + [data_sample_1_input],
        outputs=FLTaskOutputGenerator.traintask(transient=True),
        worker=worker,
    )

    cp_added = client.add_compute_plan(cp_spec)
    client.wait(cp_added)

    traintask_1_profile = client.get_compute_task_profiling(traintask_spec_1.task_id)
    assert len(traintask_1_profile["execution_rundown"]) == 4
