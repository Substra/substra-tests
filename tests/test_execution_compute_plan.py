import uuid

import pytest
import substra
from substra.sdk import models

import substratest as sbt
from substratest.client import Client
from substratest.factory import AssetsFactory
from substratest.fl_interface import AlgoCategory
from substratest.fl_interface import FLTaskInputGenerator
from substratest.fl_interface import FLTaskOutputGenerator
from substratest.fl_interface import InputIdentifiers
from substratest.fl_interface import OutputIdentifiers


def test_compute_plan_simple(
    factory, client_1, client_2, default_dataset_1, default_dataset_2, default_metrics, channel
):
    """Execution of a compute plan containing multiple traintuples:
    - 1 traintuple executed on organization 1
    - 1 traintuple executed on organization 2
    - 1 traintuple executed on organization 1 depending on previous traintuples
    - 1 predicttuple executed on organization 1 depending on previous traintuples
    - 1 testtuple executed on organization 1 depending on the last predicttuple and on one metric
    """

    cp_key = str(uuid.uuid4())

    simple_algo_spec = factory.create_algo(AlgoCategory.simple)
    simple_algo_2 = client_2.add_algo(simple_algo_spec)

    predict_algo_spec = factory.create_algo(AlgoCategory.predict)
    predict_algo_2 = client_2.add_algo(predict_algo_spec)

    channel.wait_for_asset_synchronized(simple_algo_2)

    # create compute plan
    cp_spec = factory.create_compute_plan(
        key=cp_key,
        tag="foo",
        name="Bar",
        metadata={"foo": "bar"},
    )

    traintuple_spec_1 = cp_spec.create_traintuple(
        algo=simple_algo_2,
        inputs=default_dataset_1.train_data_inputs,
        metadata=None,
    )

    traintuple_spec_2 = cp_spec.create_traintuple(
        algo=simple_algo_2,
        inputs=default_dataset_2.train_data_inputs,
        metadata={},
    )

    traintuple_spec_3 = cp_spec.create_traintuple(
        algo=simple_algo_2,
        inputs=default_dataset_1.train_data_inputs
        + FLTaskInputGenerator.trains_to_train([traintuple_spec_1.traintuple_id, traintuple_spec_2.traintuple_id]),
        metadata={"foo": "bar"},
    )

    predicttuple_spec_3 = cp_spec.create_predicttuple(
        algo=predict_algo_2,
        inputs=default_dataset_1.test_data_inputs
        + FLTaskInputGenerator.train_to_predict(traintuple_spec_3.traintuple_id),
        metadata={"foo": "bar"},
    )

    cp_spec.create_testtuple(
        algo=default_metrics[0],
        inputs=default_dataset_1.test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttuple_spec_3.predicttuple_id),
        metadata={"foo": "bar"},
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

    traintuples = client_1.list_compute_plan_traintuples(cp.key)
    assert len(traintuples) == 3

    predicttuples = client_1.list_compute_plan_predicttuples(cp.key)
    assert len(predicttuples) == 1

    testtuples = client_1.list_compute_plan_testtuples(cp.key)
    assert len(testtuples) == 1

    # check all tuples are done and check they have been executed on the expected organization
    for t in traintuples:
        assert t.status == models.Status.done
        assert t.start_date is not None
        assert t.end_date is not None

    traintuple_1, traintuple_2, traintuple_3 = traintuples

    assert len([i for i in traintuple_3.inputs if i.identifier == InputIdentifiers.models]) == 2

    for p in predicttuples:
        assert p.status == models.Status.done

    for t in testtuples:
        assert t.status == models.Status.done

    testtuple = client_1.get_testtuple(testtuples[0].key)
    predicttuple = client_1.get_predicttuple(predicttuples[0].key)

    # check tuples metadata
    assert traintuple_1.metadata == {}
    assert traintuple_2.metadata == {}
    assert traintuple_3.metadata == {"foo": "bar"}
    assert predicttuple.metadata == {"foo": "bar"}
    assert testtuple.metadata == {"foo": "bar"}

    # check tuples rank
    assert traintuple_1.rank == 0
    assert traintuple_2.rank == 0
    assert traintuple_3.rank == 1
    assert predicttuple.rank == 2
    assert testtuple.rank == predicttuple.rank + 1

    # check testtuple perfs
    assert len(testtuple.test.perfs) == 1
    assert len(set(testtuple.test.perfs.values())) == 1
    assert list(testtuple.test.perfs.values())[0] == pytest.approx(4)

    # check compute plan perfs
    performances = client_1.get_performances(cp.key)
    assert all(len(val) == 1 for val in performances.dict().values())
    assert set(testtuple.test.perfs.values()) == set(performances.performance)

    # XXX as the first two tuples have the same rank, there is currently no way to know
    #     which one will be returned first
    workers_rank_0 = set([traintuple_1.worker, traintuple_2.worker])
    assert workers_rank_0 == set([client_1.organization_id, client_2.organization_id])
    assert traintuple_3.worker == client_1.organization_id
    assert predicttuple.worker == client_1.organization_id
    assert testtuple.worker == client_1.organization_id

    # check mapping
    traintuple_id_1 = traintuple_spec_1.traintuple_id
    traintuple_id_2 = traintuple_spec_2.traintuple_id
    traintuple_id_3 = traintuple_spec_3.traintuple_id
    generated_ids = [traintuple_id_1, traintuple_id_2, traintuple_id_3]
    rank_0_traintuple_keys = [traintuple_1.key, traintuple_2.key]
    assert set(generated_ids) == set([traintuple_id_1, traintuple_id_2, traintuple_id_3])
    assert set(rank_0_traintuple_keys) == set([traintuple_id_1, traintuple_id_2])
    assert traintuple_3.key == traintuple_id_3


@pytest.mark.slow
def test_compute_plan_single_client_success(factory, client, default_dataset, default_metric):
    """A compute plan with 3 traintuples and 3 associated testtuples"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintuple + testtuple
    # 2. traintuple + testtuple
    # 3. traintuple + testtuple

    data_sample_1_input, data_sample_2_input, data_sample_3_input, _ = default_dataset.train_data_sample_inputs

    simple_algo_spec = factory.create_algo(AlgoCategory.simple)
    simple_algo = client.add_algo(simple_algo_spec)

    predict_algo_spec = factory.create_algo(AlgoCategory.predict)
    predict_algo = client.add_algo(predict_algo_spec)

    cp_spec = factory.create_compute_plan()

    traintuple_spec_1 = cp_spec.create_traintuple(
        algo=simple_algo, inputs=default_dataset.opener_input + [data_sample_1_input]
    )

    predicttuple_spec_1 = cp_spec.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.train_to_predict(traintuple_spec_1.traintuple_id),
    )

    cp_spec.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttuple_spec_1.predicttuple_id),
    )

    traintuple_spec_2 = cp_spec.create_traintuple(
        algo=simple_algo,
        inputs=default_dataset.opener_input
        + [data_sample_2_input]
        + FLTaskInputGenerator.trains_to_train([traintuple_spec_1.traintuple_id]),
    )
    predicttuple_spec_2 = cp_spec.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.train_to_predict(traintuple_spec_2.traintuple_id),
    )
    cp_spec.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttuple_spec_2.predicttuple_id),
    )

    traintuple_spec_3 = cp_spec.create_traintuple(
        algo=simple_algo,
        inputs=default_dataset.opener_input
        + [data_sample_3_input]
        + FLTaskInputGenerator.trains_to_train([traintuple_spec_2.traintuple_id]),
    )
    predicttuple_spec_3 = cp_spec.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.train_to_predict(traintuple_spec_3.traintuple_id),
    )
    cp_spec.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttuple_spec_3.predicttuple_id),
    )

    # Submit compute plan and wait for it to complete
    cp_added = client.add_compute_plan(cp_spec)
    cp = client.wait(cp_added)

    assert cp.status == "PLAN_STATUS_DONE"
    assert cp.end_date is not None
    assert cp.duration is not None

    # All the train/test tuples should succeed
    for t in (
        client.list_compute_plan_traintuples(cp.key)
        + client.list_compute_plan_predicttuples(cp.key)
        + client.list_compute_plan_testtuples(cp.key)
    ):
        assert t.status == models.Status.done


@pytest.mark.slow
def test_compute_plan_update(factory, client, default_dataset, default_metric):
    """A compute plan with 3 traintuples and 3 associated testtuples.

    This is done by sending 3 requests (one create and two updates).
    """

    data_sample_1_input, data_sample_2_input, data_sample_3_input, _ = default_dataset.train_data_sample_inputs

    simple_algo_spec = factory.create_algo(AlgoCategory.simple)
    simple_algo = client.add_algo(simple_algo_spec)

    predict_algo_spec = factory.create_algo(AlgoCategory.predict)
    predict_algo = client.add_algo(predict_algo_spec)

    # Create a compute plan with traintuple + testtuple

    cp_spec = factory.create_compute_plan()
    traintuple_spec_1 = cp_spec.create_traintuple(
        algo=simple_algo, inputs=default_dataset.opener_input + [data_sample_1_input]
    )

    predicttuple_spec_1 = cp_spec.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.train_to_predict(traintuple_spec_1.traintuple_id),
    )

    cp_spec.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttuple_spec_1.predicttuple_id),
    )
    cp = client.add_compute_plan(cp_spec, auto_batching=True, batch_size=1)

    # Update compute plan with traintuple + testtuple

    cp_spec = factory.add_compute_plan_tuples(cp)
    traintuple_spec_2 = cp_spec.create_traintuple(
        algo=simple_algo,
        inputs=default_dataset.opener_input
        + [data_sample_2_input]
        + FLTaskInputGenerator.trains_to_train([traintuple_spec_1.traintuple_id]),
        metadata={"foo": "bar"},
    )
    predicttuple_spec_2 = cp_spec.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.train_to_predict(traintuple_spec_2.traintuple_id),
        metadata={"foo": "bar"},
    )
    cp_spec.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttuple_spec_2.predicttuple_id),
        metadata={"foo": "bar"},
    )
    cp = client.add_compute_plan_tuples(cp_spec, auto_batching=True, batch_size=1)

    # Update compute plan with traintuple

    cp_spec = factory.add_compute_plan_tuples(cp)
    traintuple_spec_3 = cp_spec.create_traintuple(
        algo=simple_algo,
        inputs=default_dataset.opener_input
        + [data_sample_3_input]
        + FLTaskInputGenerator.trains_to_train([traintuple_spec_2.traintuple_id]),
    )
    predicttuple_spec_3 = cp_spec.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.train_to_predict(traintuple_spec_3.traintuple_id),
    )
    cp = client.add_compute_plan_tuples(cp_spec)

    # Update compute plan with testtuple

    cp_spec = factory.add_compute_plan_tuples(cp)
    cp_spec.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttuple_spec_3.predicttuple_id),
    )
    cp = client.add_compute_plan_tuples(cp_spec)

    # All the train/test tuples should succeed
    cp_added = client.get_compute_plan(cp.key)
    cp = client.wait(cp_added)
    traintuples = client.list_compute_plan_traintuples(cp.key)
    predicttuples = client.list_compute_plan_predicttuples(cp.key)
    testtuples = client.list_compute_plan_testtuples(cp.key)
    tuples = traintuples + predicttuples + testtuples
    assert len(tuples) == 9
    for t in tuples:
        assert t.status == models.Status.done

    # Check tuples metadata
    assert traintuples[1].metadata == {"foo": "bar"}
    assert predicttuples[1].metadata == {"foo": "bar"}
    assert testtuples[1].metadata == {"foo": "bar"}


@pytest.mark.slow
@pytest.mark.remote_only
def test_compute_plan_single_client_failure(factory, client, default_dataset, default_metric):
    """In a compute plan with 3 traintuples, failing the root traintuple
    should cancel its descendents and the associated testtuples"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintuple + predicttuple + testtuple
    # 2. traintuple + predicttuple + testtuple
    # 3. traintuple + predicttuple + testtuple
    #
    # Intentionally use an invalid (broken) algo.

    data_sample_1_input, data_sample_2_input, data_sample_3_input, _ = default_dataset.train_data_sample_inputs

    simple_algo_spec = factory.create_algo(AlgoCategory.simple, py_script=sbt.factory.INVALID_ALGO_SCRIPT)
    simple_algo = client.add_algo(simple_algo_spec)

    predict_algo_spec = factory.create_algo(AlgoCategory.predict)
    predict_algo = client.add_algo(predict_algo_spec)

    cp_spec = factory.create_compute_plan()

    traintuple_spec_1 = cp_spec.create_traintuple(
        algo=simple_algo, inputs=default_dataset.opener_input + [data_sample_1_input]
    )
    predicttuple_spec_1 = cp_spec.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.train_to_predict(traintuple_spec_1.traintuple_id),
    )
    cp_spec.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttuple_spec_1.predicttuple_id),
    )

    traintuple_spec_2 = cp_spec.create_traintuple(
        algo=simple_algo,
        inputs=default_dataset.opener_input
        + [data_sample_2_input]
        + FLTaskInputGenerator.trains_to_train([traintuple_spec_1.traintuple_id]),
    )
    predicttuple_spec_2 = cp_spec.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.train_to_predict(traintuple_spec_2.traintuple_id),
    )

    cp_spec.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttuple_spec_2.predicttuple_id),
    )

    traintuple_spec_3 = cp_spec.create_traintuple(
        algo=simple_algo,
        inputs=default_dataset.opener_input
        + [data_sample_3_input]
        + FLTaskInputGenerator.trains_to_train([traintuple_spec_2.traintuple_id]),
    )

    predicttuple_spec_3 = cp_spec.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.train_to_predict(traintuple_spec_3.traintuple_id),
    )
    cp_spec.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttuple_spec_3.predicttuple_id),
    )

    # Submit compute plan and wait for it to complete
    cp_added = client.add_compute_plan(cp_spec)
    cp = client.wait(cp_added, raises=False)

    assert cp.status == "PLAN_STATUS_FAILED"
    assert cp.failed_task.category == "TASK_TRAIN"
    assert cp.end_date is not None
    assert cp.duration is not None


# FIXME: test_compute_plan_aggregate_composite_traintuples is too complex, consider refactoring
@pytest.mark.slow  # noqa: C901
def test_compute_plan_aggregate_composite_traintuples(  # noqa: C901
    factory, clients, default_datasets, default_metrics
):
    """
    Compute plan version of the `test_aggregate_composite_traintuples` method from `test_execution.py`
    """
    aggregate_worker = clients[0].organization_id
    number_of_rounds = 2

    # register algos on first organization
    spec = factory.create_algo(AlgoCategory.composite)
    composite_algo = clients[0].add_algo(spec)
    spec = factory.create_algo(AlgoCategory.aggregate)
    aggregate_algo = clients[0].add_algo(spec)
    spec = factory.create_algo(AlgoCategory.predict)
    predict_algo = clients[0].add_algo(spec)
    spec = factory.create_algo(AlgoCategory.predict_composite)
    predict_algo_composite = clients[0].add_algo(spec)

    # launch execution
    previous_aggregatetuple_spec = None
    previous_composite_traintuple_specs = []

    cp_spec = factory.create_compute_plan()

    for round_ in range(number_of_rounds):
        # create composite traintuple on each organization
        composite_traintuple_specs = []
        for index, dataset in enumerate(default_datasets):
            if previous_aggregatetuple_spec is not None:
                local_input = FLTaskInputGenerator.composite_to_local(
                    previous_composite_traintuple_specs[index].composite_traintuple_id
                )
                shared_input = FLTaskInputGenerator.aggregate_to_shared(previous_aggregatetuple_spec.aggregatetuple_id)

            else:
                local_input = []
                shared_input = []

            spec = cp_spec.create_composite_traintuple(
                composite_algo=composite_algo,
                inputs=dataset.opener_input
                + [dataset.train_data_sample_inputs[0 + round_]]
                + local_input
                + shared_input,
                metadata={"foo": "bar"},
                outputs=FLTaskOutputGenerator.composite_traintuple(
                    shared_authorized_ids=[client.organization_id for client in clients],
                    local_authorized_ids=[clients[index].organization_id],
                ),
            )

            composite_traintuple_specs.append(spec)

        # create aggregate on its organization
        spec = cp_spec.create_aggregatetuple(
            aggregate_algo=aggregate_algo,
            worker=aggregate_worker,
            inputs=FLTaskInputGenerator.composites_to_aggregate(
                [
                    composite_traintuple_spec.composite_traintuple_id
                    for composite_traintuple_spec in composite_traintuple_specs
                ]
            ),
            metadata={"foo": "bar"},
        )

        # save state of round
        previous_aggregatetuple_spec = spec
        previous_composite_traintuple_specs = composite_traintuple_specs

    # last round: create associated testtuple
    for composite_traintuple, dataset, metric in zip(
        previous_composite_traintuple_specs, default_datasets, default_metrics
    ):
        spec = cp_spec.create_predicttuple(
            algo=predict_algo_composite,
            inputs=dataset.test_data_inputs
            + FLTaskInputGenerator.composite_to_predict(composite_traintuple.composite_traintuple_id),
        )
        cp_spec.create_testtuple(
            algo=metric, inputs=dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(spec.predicttuple_id)
        )

    predicttuple_from_aggregate_spec = cp_spec.create_predicttuple(
        algo=predict_algo,
        inputs=default_datasets[0].test_data_inputs
        + FLTaskInputGenerator.aggregate_to_predict(previous_aggregatetuple_spec.aggregatetuple_id),
    )
    cp_spec.create_testtuple(
        algo=metric,
        inputs=default_datasets[0].test_data_inputs
        + FLTaskInputGenerator.predict_to_test(predicttuple_from_aggregate_spec.predicttuple_id),
    )

    cp_added = clients[0].add_compute_plan(cp_spec)
    cp = clients[0].wait(cp_added)

    traintuples = clients[0].list_compute_plan_traintuples(cp.key)
    composite_traintuples = clients[0].list_compute_plan_composite_traintuples(cp.key)
    aggregatetuples = clients[0].list_compute_plan_aggregatetuples(cp.key)
    predicttuples = clients[0].list_compute_plan_predicttuples(cp.key)
    testtuples = clients[0].list_compute_plan_testtuples(cp.key)

    for task in composite_traintuple_specs:
        remote_task = clients[0].get_composite_traintuple(task.id)
        if task.in_head_model_id:
            assert (
                len(
                    [
                        i
                        for i in remote_task.inputs
                        if i.identifier == InputIdentifiers.local
                        and i.parent_task_key == task.in_head_model_id
                        and i.parent_task_output_identifier == InputIdentifiers.local
                    ]
                )
                == 1
            )
        if task.in_trunk_model_id:
            assert (
                len(
                    [
                        i
                        for i in remote_task.inputs
                        if i.identifier == InputIdentifiers.shared
                        and i.parent_task_key == task.in_trunk_model_id
                        and i.parent_task_output_identifier == OutputIdentifiers.model
                    ]
                )
                == 1
            )

    tuples = traintuples + composite_traintuples + aggregatetuples + predicttuples + testtuples
    for t in tuples:
        assert t.status == models.Status.done, t

    # Check tuples metadata
    for tuple in composite_traintuples + aggregatetuples:
        assert tuple.metadata == {"foo": "bar"}

    # Check that permissions were correctly set
    for task in composite_traintuples:
        task = clients[0].get_composite_traintuple(task.key)
        trunk = task.outputs[OutputIdentifiers.shared].value
        assert len(trunk.permissions.process.authorized_ids) == len(clients)


def test_compute_plan_circular_dependency_failure(factory, client, default_dataset):
    spec = factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    cp_spec = factory.create_compute_plan()

    traintuple_spec_1 = cp_spec.create_traintuple(inputs=default_dataset.train_data_inputs, algo=algo)

    traintuple_spec_2 = cp_spec.create_traintuple(inputs=default_dataset.train_data_inputs, algo=algo)

    traintuple_spec_1.in_models_ids.append(traintuple_spec_2.id)
    traintuple_spec_2.in_models_ids.append(traintuple_spec_1.id)
    traintuple_spec_1.inputs.append(FLTaskInputGenerator.trains_to_train([traintuple_spec_2.id])[0])
    traintuple_spec_2.inputs.append(FLTaskInputGenerator.trains_to_train([traintuple_spec_1.id])[0])

    with pytest.raises(substra.exceptions.InvalidRequest) as e:
        client.add_compute_plan(cp_spec)

    assert "missing dependency among inModels IDs" in str(e.value)


@pytest.mark.slow
@pytest.mark.remote_only
def test_execution_compute_plan_canceled(factory, client, default_dataset, cfg):
    # XXX A canceled compute plan can be done if the it is canceled while it last tuples
    #     are executing on the workers. The compute plan status will in this case change
    #     from canceled to done.
    #     To increase our confidence that the compute plan won't be done, we create a
    #     compute plan with a large amount of tuples.
    nb_traintuples = 32

    spec = factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    cp_spec = factory.create_compute_plan()
    previous_traintuple = None
    inputs = default_dataset.opener_input + default_dataset.train_data_sample_inputs[:1]

    for _ in range(nb_traintuples):
        input_models = (
            FLTaskInputGenerator.trains_to_train([previous_traintuple.traintuple_id])
            if previous_traintuple is not None
            else []
        )
        previous_traintuple = cp_spec.create_traintuple(algo=algo, inputs=inputs + input_models)

    cp = client.add_compute_plan(cp_spec)

    # wait the first traintuple to be executed to ensure that the compute plan is launched
    # and tuples are scheduled in the celery workers
    first_traintuple = [t for t in client.list_compute_plan_traintuples(cp.key) if t.rank == 0][0]
    first_traintuple = client.wait(first_traintuple)
    assert first_traintuple.status == models.Status.done

    client.cancel_compute_plan(cp.key)
    # as cancel request do not directly update localrep we need to wait for the sync
    cp = client.wait(cp, raises=False, timeout=cfg.options.organization_sync_timeout)
    assert cp.status == models.ComputePlanStatus.canceled
    assert cp.end_date is not None
    assert cp.duration is not None

    # check that the status of the done tuple as not been updated
    first_traintuple = [t for t in client.list_compute_plan_traintuples(cp.key) if t.rank == 0][0]
    assert first_traintuple.status == models.Status.done


@pytest.mark.slow
@pytest.mark.remote_only
def test_compute_plan_no_batching(factory, client, default_dataset):

    spec = factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    # Create a compute plan
    cp_spec = factory.create_compute_plan()
    traintuple_spec_1 = cp_spec.create_traintuple(
        algo=algo, inputs=default_dataset.opener_input + default_dataset.train_data_sample_inputs[:1]
    )
    cp_added = client.add_compute_plan(cp_spec, auto_batching=False)
    cp = client.wait(cp_added)

    traintuples = client.list_compute_plan_traintuples(cp.key)
    assert len(traintuples) == 1
    assert all([tuple_.status == models.Status.done for tuple_ in traintuples])

    # Update the compute plan
    cp_spec = factory.add_compute_plan_tuples(cp)
    cp_spec.create_traintuple(
        algo=algo,
        inputs=default_dataset.opener_input
        + default_dataset.train_data_sample_inputs[1:2]
        + FLTaskInputGenerator.trains_to_train([traintuple_spec_1.traintuple_id]),
        metadata={"foo": "bar"},
    )
    cp_added = client.add_compute_plan_tuples(cp_spec, auto_batching=False)
    cp = client.wait(cp_added)

    traintuples = client.list_compute_plan_traintuples(cp.key)
    assert len(traintuples) == 2
    assert all([tuple_.status == models.Status.done for tuple_ in traintuples])


@pytest.mark.slow
@pytest.mark.remote_only
def test_compute_plan_transient_outputs(factory: AssetsFactory, client: Client, default_dataset):
    """
    Create a simple compute plan with tasks using transient inputs, check if the flag is set
    """
    data_sample_1_input, data_sample_2_input, _, _ = default_dataset.train_data_sample_inputs

    # Register the Algo
    simple_algo_spec = factory.create_algo(AlgoCategory.simple)
    simple_algo = client.add_algo(simple_algo_spec)

    cp_spec = factory.create_compute_plan()
    traintuple_spec_1 = cp_spec.create_traintuple(
        algo=simple_algo,
        inputs=default_dataset.opener_input + [data_sample_1_input],
        outputs=FLTaskOutputGenerator.traintuple(transient=True),
    )

    cp_spec.create_traintuple(
        algo=simple_algo,
        inputs=default_dataset.opener_input
        + [data_sample_2_input]
        + FLTaskInputGenerator.trains_to_train([traintuple_spec_1.traintuple_id]),
    )

    cp_added = client.add_compute_plan(cp_spec)
    client.wait(cp_added)

    traintuple_1 = client.get_traintuple(traintuple_spec_1.traintuple_id)
    assert traintuple_1.outputs[OutputIdentifiers.model].is_transient is True

    # Validate that the transient model is properly deleted
    model = client.get_task_models(traintuple_spec_1.traintuple_id)[0]
    client.wait_model_deletion(model.key)

    # Validate that we can't create a new task that use this model
    traintuple_spec_3 = factory.create_traintuple(
        algo=simple_algo,
        inputs=default_dataset.opener_input
        + [data_sample_2_input]
        + FLTaskInputGenerator.trains_to_train([traintuple_spec_1.traintuple_id]),
    )

    with pytest.raises(substra.exceptions.InvalidRequest) as err:
        client.add_traintuple(traintuple_spec_3)

    assert "has been disabled" in str(err.value)


@pytest.mark.slow
@pytest.mark.remote_only
def test_compute_task_profile(factory, client, default_dataset):
    """
    Creates a simple task to check that tasks profiles are correctly produced
    """
    data_sample_1_input, _, _, _ = default_dataset.train_data_sample_inputs
    simple_algo_spec = factory.create_algo(AlgoCategory.simple)
    simple_algo = client.add_algo(simple_algo_spec)

    cp_spec = factory.create_compute_plan()
    traintuple_spec_1 = cp_spec.create_traintuple(
        algo=simple_algo,
        inputs=default_dataset.opener_input + [data_sample_1_input],
        outputs=FLTaskOutputGenerator.traintuple(transient=True),
    )

    cp_added = client.add_compute_plan(cp_spec)
    client.wait(cp_added)

    traintuple_1_profile = client.get_compute_task_profiling(traintuple_spec_1.traintuple_id)
    assert len(traintuple_1_profile["execution_rundown"]) == 4
