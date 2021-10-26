import pytest
import substra
from substra.sdk import models
import substratest as sbt
from substratest.factory import AlgoCategory, Permissions


@pytest.mark.remote_only
def test_compute_plan_simple(factory, client_1, client_2, default_dataset_1, default_dataset_2, default_metrics):
    """Execution of a compute plan containing multiple traintuples:
    - 1 traintuple executed on node 1
    - 1 traintuple executed on node 2
    - 1 traintuple executed on node 1 depending on previous traintuples
    - 1 testtuple executed on node 1 depending on the last traintuple and on multiple metrics
    """

    spec = factory.create_algo(AlgoCategory.simple, local=client_2.debug)
    algo_2 = client_2.add_algo(spec)

    # create compute plan
    cp_spec = factory.create_compute_plan(
        tag='foo',
        metadata={"foo": "bar"},
    )

    traintuple_spec_1 = cp_spec.add_traintuple(
        algo=algo_2,
        dataset=default_dataset_1,
        data_samples=default_dataset_1.train_data_sample_keys,
        metadata=None,
    )

    traintuple_spec_2 = cp_spec.add_traintuple(
        algo=algo_2,
        dataset=default_dataset_2,
        data_samples=default_dataset_2.train_data_sample_keys,
        metadata={},
    )

    traintuple_spec_3 = cp_spec.add_traintuple(
        algo=algo_2,
        dataset=default_dataset_1,
        data_samples=default_dataset_1.train_data_sample_keys,
        in_models=[traintuple_spec_1, traintuple_spec_2],
        metadata={"foo": "bar"},
    )

    cp_spec.add_testtuple(
        metrics=default_metrics,
        traintuple_spec=traintuple_spec_3,
        dataset=default_dataset_1,
        data_samples=default_dataset_1.test_data_sample_keys,
        metadata={'foo': 'bar'},
    )

    # submit compute plan and wait for it to complete
    cp_added = client_1.add_compute_plan(cp_spec)

    cp = client_1.wait(cp_added)
    assert cp.tag == 'foo'
    assert cp.metadata == {"foo": "bar"}

    traintuples = client_1.list_compute_plan_traintuples(cp.key)
    assert len(traintuples) == 3

    testtuples = client_1.list_compute_plan_testtuples(cp.key)
    assert len(testtuples) == 1

    # check all tuples are done and check they have been executed on the expected node
    for t in traintuples:
        assert t.status == models.Status.done
        assert t.start_date is not None
        assert t.end_date is not None

    traintuple_1, traintuple_2, traintuple_3 = traintuples

    assert len(traintuple_3.parent_task_keys) == 2

    for t in testtuples:
        assert t.status == models.Status.done

    testtuple = client_1.get_testtuple(testtuples[0].key)

    # check tuples metadata
    assert traintuple_1.metadata == {}
    assert traintuple_2.metadata == {}
    assert traintuple_3.metadata == {"foo": "bar"}
    assert testtuple.metadata == {"foo": "bar"}

    # check tuples rank
    assert traintuple_1.rank == 0
    assert traintuple_2.rank == 0
    assert traintuple_3.rank == 1
    assert testtuple.rank == traintuple_3.rank

    # check testtuple perfs
    assert len(testtuple.test.perfs) == len(default_metrics)
    assert set(testtuple.test.perfs.values()) == {4, 5, }

    # XXX as the first two tuples have the same rank, there is currently no way to know
    #     which one will be returned first
    workers_rank_0 = set([traintuple_1.worker, traintuple_2.worker])
    assert workers_rank_0 == set([client_1.node_id, client_2.node_id])
    assert traintuple_3.worker == client_1.node_id
    assert testtuple.worker == client_1.node_id

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
def test_compute_plan_single_client_success(factory, client, default_dataset, default_metric, default_metric_local):
    """A compute plan with 3 traintuples and 3 associated testtuples"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintuple + testtuple
    # 2. traintuple + testtuple
    # 3. traintuple + testtuple

    data_sample_1, data_sample_2, data_sample_3, _ = default_dataset.train_data_sample_keys

    spec = factory.create_algo(AlgoCategory.simple, local=client.debug)
    algo = client.add_algo(spec)

    cp_spec = factory.create_compute_plan()

    traintuple_spec_1 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_1]
    )
    cp_spec.add_testtuple(
        metrics=[default_metric_local] if client.debug else [default_metric],
        traintuple_spec=traintuple_spec_1,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )

    traintuple_spec_2 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_2],
        in_models=[traintuple_spec_1]
    )
    cp_spec.add_testtuple(
        metrics=[default_metric_local] if client.debug else [default_metric],
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
        traintuple_spec=traintuple_spec_2
    )

    traintuple_spec_3 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_3],
        in_models=[traintuple_spec_2]
    )
    cp_spec.add_testtuple(
        metrics=[default_metric_local] if client.debug else [default_metric],
        traintuple_spec=traintuple_spec_3,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )

    # Submit compute plan and wait for it to complete
    cp_added = client.add_compute_plan(cp_spec)
    cp = client.wait(cp_added)

    assert cp.status == "PLAN_STATUS_DONE"

    # All the train/test tuples should succeed
    for t in client.list_compute_plan_traintuples(cp.key) + client.list_compute_plan_testtuples(cp.key):
        assert t.status == models.Status.done


@pytest.mark.slow
def test_compute_plan_update(factory, client, default_dataset, default_metric, default_metric_local):
    """A compute plan with 3 traintuples and 3 associated testtuples.

    This is done by sending 3 requests (one create and two updates).
    """

    data_sample_1, data_sample_2, data_sample_3, _ = default_dataset.train_data_sample_keys

    spec = factory.create_algo(AlgoCategory.simple, local=client.debug)
    algo = client.add_algo(spec)

    # Create a compute plan with traintuple + testtuple

    cp_spec = factory.create_compute_plan()
    traintuple_spec_1 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_1]
    )
    cp_spec.add_testtuple(
        metrics=[default_metric_local] if client.debug else [default_metric],
        traintuple_spec=traintuple_spec_1,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )
    cp = client.add_compute_plan(cp_spec, auto_batching=True, batch_size=1)

    # Update compute plan with traintuple + testtuple

    cp_spec = factory.update_compute_plan(cp)
    traintuple_spec_2 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_2],
        in_models=[traintuple_spec_1],
        metadata={"foo": "bar"},
    )
    cp_spec.add_testtuple(
        metrics=[default_metric_local] if client.debug else [default_metric],
        traintuple_spec=traintuple_spec_2,
        metadata={"foo": "bar"},
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )
    cp = client.update_compute_plan(cp_spec, auto_batching=True, batch_size=1)

    # Update compute plan with traintuple

    cp_spec = factory.update_compute_plan(cp)
    traintuple_spec_3 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_3],
        in_models=[traintuple_spec_2]
    )
    cp = client.update_compute_plan(cp_spec)

    # Update compute plan with testtuple

    cp_spec = factory.update_compute_plan(cp)
    cp_spec.add_testtuple(
        metrics=[default_metric_local] if client.debug else [default_metric],
        traintuple_spec=traintuple_spec_3,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )
    cp = client.update_compute_plan(cp_spec)

    # All the train/test tuples should succeed
    cp_added = client.get_compute_plan(cp.key)
    cp = client.wait(cp_added)
    traintuples = client.list_compute_plan_traintuples(cp.key)
    testtuples = client.list_compute_plan_testtuples(cp.key)
    tuples = traintuples + testtuples
    assert len(tuples) == 6
    for t in tuples:
        assert t.status == models.Status.done

    # Check tuples metadata
    assert traintuples[1].metadata == {"foo": "bar"}
    assert testtuples[1].metadata == {"foo": "bar"}


@pytest.mark.slow
@pytest.mark.remote_only
def test_compute_plan_single_client_failure(factory, client, default_dataset, default_metric):
    """In a compute plan with 3 traintuples, failing the root traintuple
    should cancel its descendents and the associated testtuples"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintuple + testtuple
    # 2. traintuple + testtuple
    # 3. traintuple + testtuple
    #
    # Intentionally use an invalid (broken) algo.

    data_sample_1, data_sample_2, data_sample_3, _ = default_dataset.train_data_sample_keys

    spec = factory.create_algo(category=AlgoCategory.simple, py_script=sbt.factory.INVALID_ALGO_SCRIPT,
                               local=client.debug)
    algo = client.add_algo(spec)

    cp_spec = factory.create_compute_plan()

    traintuple_spec_1 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_1]
    )
    cp_spec.add_testtuple(
        metrics=[default_metric],
        traintuple_spec=traintuple_spec_1,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )

    traintuple_spec_2 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_2],
        in_models=[traintuple_spec_1]
    )
    cp_spec.add_testtuple(
        metrics=[default_metric],
        traintuple_spec=traintuple_spec_2,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )

    traintuple_spec_3 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_3],
        in_models=[traintuple_spec_2]
    )
    cp_spec.add_testtuple(
        metrics=[default_metric],
        traintuple_spec=traintuple_spec_3,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )

    # Submit compute plan and wait for it to complete
    cp_added = client.add_compute_plan(cp_spec)
    cp = client.wait(cp_added, raises=False)

    assert cp.status == "PLAN_STATUS_FAILED"
    assert cp.failed_task.category == "TASK_TRAIN"


@pytest.mark.slow
def test_compute_plan_aggregate_composite_traintuples(factory, clients, default_datasets, default_metrics,
                                                      default_metrics_local):
    """
    Compute plan version of the `test_aggregate_composite_traintuples` method from `test_execution.py`
    """
    aggregate_worker = clients[0].node_id
    number_of_rounds = 2

    # register algos on first node
    spec = factory.create_algo(AlgoCategory.composite, local=clients[0].debug)
    composite_algo = clients[0].add_algo(spec)
    spec = factory.create_algo(AlgoCategory.aggregate, local=clients[0].debug)
    aggregate_algo = clients[0].add_algo(spec)

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
                out_trunk_model_permissions=Permissions(public=False, authorized_ids=[c.node_id for c in clients]),
                metadata={'foo': 'bar'},
                **kwargs,
            )
            composite_traintuple_specs.append(spec)

        # create aggregate on its node
        spec = cp_spec.add_aggregatetuple(
            aggregate_algo=aggregate_algo,
            worker=aggregate_worker,
            in_models=composite_traintuple_specs,
            metadata={'foo': 'bar'},
        )

        # save state of round
        previous_aggregatetuple_spec = spec
        previous_composite_traintuple_specs = composite_traintuple_specs

    # last round: create associated testtuple
    for composite_traintuple, dataset, metric, metric_local in zip(
            previous_composite_traintuple_specs, default_datasets, default_metrics, default_metrics_local):
        cp_spec.add_testtuple(
            metrics=[metric_local] if clients[0].debug else [metric],
            dataset=dataset,
            data_samples=dataset.test_data_sample_keys,
            traintuple_spec=composite_traintuple,
        )
    cp_spec.add_testtuple(
        metrics=[metric_local] if clients[0].debug else [metric],
        traintuple_spec=previous_aggregatetuple_spec,
        dataset=default_datasets[0],
        data_samples=default_datasets[0].test_data_sample_keys,
    )

    cp_added = clients[0].add_compute_plan(cp_spec)
    cp = clients[0].wait(cp_added)
    traintuples = clients[0].list_compute_plan_traintuples(cp.key)
    composite_traintuples = clients[0].list_compute_plan_composite_traintuples(cp.key)
    aggregatetuples = clients[0].list_compute_plan_aggregatetuples(cp.key)
    testtuples = clients[0].list_compute_plan_testtuples(cp.key)

    for task in composite_traintuple_specs:
        remote_task = clients[0].get_composite_traintuple(task.id)
        if task.in_head_model_id:
            assert task.in_head_model_id in remote_task.parent_task_keys
        if task.in_trunk_model_id:
            assert task.in_trunk_model_id in remote_task.parent_task_keys

    tuples = traintuples + composite_traintuples + aggregatetuples + testtuples
    for t in tuples:
        assert t.status == models.Status.done, t

    # Check tuples metadata
    for tuple in composite_traintuples + aggregatetuples:
        assert tuple.metadata == {'foo': 'bar'}

    # Check that permissions were correctly set
    for task in composite_traintuples:
        task = clients[0].get_composite_traintuple(task.key)
        trunks = [model for model in task.composite.models if model.category == models.ModelType.simple]
        for trunk in trunks:
            assert len(trunk.permissions.process.authorized_ids) == len(clients)


@pytest.mark.slow
@pytest.mark.remote_only
def test_compute_plan_remove_intermediary_model(factory, client, default_dataset, default_metric):
    """
    Create a simple compute plan with clean_models at true, see it done and
    create a traintuple on a intermediary model. Expect it to fail at the
    execution.
    """
    data_sample_1, data_sample_2, data_sample_3, _ = default_dataset.train_data_sample_keys

    # register algo
    spec = factory.create_algo(AlgoCategory.simple, local=client.debug)
    algo = client.add_algo(spec)

    # create a compute plan with clean_model activate
    cp_spec = factory.create_compute_plan(clean_models=True)

    traintuple_spec_1 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_1]
    )
    cp_spec.add_testtuple(
        metrics=[default_metric],
        traintuple_spec=traintuple_spec_1,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )

    traintuple_spec_2 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_2],
        in_models=[traintuple_spec_1]
    )
    cp_spec.add_testtuple(
        metrics=[default_metric],
        traintuple_spec=traintuple_spec_2,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )

    cp_added = client.add_compute_plan(cp_spec)
    cp = client.wait(cp_added)

    traintuple_spec_3 = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_3],
        traintuples=client.list_compute_plan_traintuples(cp.key)
    )

    with pytest.raises(sbt.errors.FutureFailureError):
        cp_added = client.add_traintuple(traintuple_spec_3)
        client.wait(cp_added)


def test_compute_plan_circular_dependency_failure(factory, client, default_dataset):
    spec = factory.create_algo(AlgoCategory.simple, local=client.debug)
    algo = client.add_algo(spec)

    cp_spec = factory.create_compute_plan()

    traintuple_spec_1 = cp_spec.add_traintuple(
        dataset=default_dataset,
        algo=algo,
        data_samples=default_dataset.train_data_sample_keys
    )

    traintuple_spec_2 = cp_spec.add_traintuple(
        dataset=default_dataset,
        algo=algo,
        data_samples=default_dataset.train_data_sample_keys
    )

    traintuple_spec_1.in_models_ids.append(traintuple_spec_2.id)
    traintuple_spec_2.in_models_ids.append(traintuple_spec_1.id)

    with pytest.raises(substra.exceptions.InvalidRequest) as e:
        client.add_compute_plan(cp_spec)

    assert 'missing dependency among inModels IDs' in str(e.value)


@pytest.mark.slow
@pytest.mark.remote_only
def test_execution_compute_plan_canceled(factory, client, default_dataset):
    # XXX A canceled compute plan can be done if the it is canceled while it last tuples
    #     are executing on the workers. The compute plan status will in this case change
    #     from canceled to done.
    #     To increase our confidence that the compute plan won't be done, we create a
    #     compute plan with a large amount of tuples.
    nb_traintuples = 32

    data_sample_key = default_dataset.train_data_sample_keys[0]

    spec = factory.create_algo(AlgoCategory.simple, local=client.debug)
    algo = client.add_algo(spec)

    cp_spec = factory.create_compute_plan()
    previous_traintuple = None
    for _ in range(nb_traintuples):
        previous_traintuple = cp_spec.add_traintuple(
            algo=algo,
            dataset=default_dataset,
            data_samples=[data_sample_key],
            in_models=[previous_traintuple] if previous_traintuple else None
        )

    cp = client.add_compute_plan(cp_spec)

    # wait the first traintuple to be executed to ensure that the compute plan is launched
    # and tuples are scheduled in the celery workers
    first_traintuple = [t for t in client.list_compute_plan_traintuples(cp.key) if t.rank == 0][0]
    first_traintuple = client.wait(first_traintuple)
    assert first_traintuple.status == models.Status.done

    cp = client.cancel_compute_plan(cp.key)
    assert cp.status == models.ComputePlanStatus.canceled

    cp = client.wait(cp, raises=False)
    assert cp.status == models.ComputePlanStatus.canceled

    # check that the status of the done tuple as not been updated
    first_traintuple = [t for t in client.list_compute_plan_traintuples(cp.key) if t.rank == 0][0]
    assert first_traintuple.status == models.Status.done


@pytest.mark.slow
@pytest.mark.remote_only
def test_compute_plan_no_batching(factory, client, default_dataset):
    data_sample_1, data_sample_2, _, _ = default_dataset.train_data_sample_keys

    spec = factory.create_algo(AlgoCategory.simple, local=client.debug)
    algo = client.add_algo(spec)

    # Create a compute plan
    cp_spec = factory.create_compute_plan()
    traintuple_spec_1 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_1],
    )
    cp_added = client.add_compute_plan(cp_spec, auto_batching=False)
    cp = client.wait(cp_added)

    traintuples = client.list_compute_plan_traintuples(cp.key)
    assert len(traintuples) == 1
    assert all([tuple_.status == models.Status.done for tuple_ in traintuples])

    # Update the compute plan
    cp_spec = factory.update_compute_plan(cp)
    cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_2],
        in_models=[traintuple_spec_1],
        metadata={"foo": "bar"},
    )
    cp_added = client.update_compute_plan(cp_spec, auto_batching=False)
    cp = client.wait(cp_added)

    traintuples = client.list_compute_plan_traintuples(cp.key)
    assert len(traintuples) == 2
    assert all([tuple_.status == models.Status.done for tuple_ in traintuples])


LOCAL_FOLDER_ALGO_SCRIPT = f"""
import json
import substratools as tools
import os
class TestAlgo(tools.Algo):
    def train(self, X, y, models, rank):
        state_file = os.path.join(self.compute_plan_path + "/state.json")
        state = None

        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
        else:
            state = {{ "factor": 0 }}

        state["factor"] += 1

        with open(state_file, "w") as f:
            json.dump(state, f)

        res = {{'value': state["factor"] * 2 }}

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


@pytest.mark.slow
def test_compute_plan_local_folder(factory, client, default_dataset, default_metric_1, default_metric_1_local):
    data_sample_1, data_sample_2, _, _ = default_dataset.train_data_sample_keys

    spec = factory.create_algo(category=AlgoCategory.simple, py_script=LOCAL_FOLDER_ALGO_SCRIPT,
                               local=client.debug)
    algo = client.add_algo(spec)
    cp_spec = factory.create_compute_plan()

    # Traintuple 1
    traintuple_spec_1 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_1],
    )

    # Traintuple 2
    traintuple_spec_2 = cp_spec.add_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=[data_sample_2],
        in_models=[traintuple_spec_1],
    )

    # Testtuple
    cp_spec.add_testtuple(
        metrics=[default_metric_1_local] if client.debug else [default_metric_1],
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
        traintuple_spec=traintuple_spec_2
    )

    cp_added = client.add_compute_plan(cp_spec, auto_batching=False)
    cp = client.wait(cp_added)

    testtuples = client.list_compute_plan_testtuples(cp.key)
    # performance is retrieved only on get, not list
    testtuple = client.get_testtuple(testtuples[0].key)
    assert list(testtuple.test.perfs.values())[0] == 20
