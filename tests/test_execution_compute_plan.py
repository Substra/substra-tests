import pytest

import substratest as sbt


@pytest.mark.skip('may raise MVCC errors')
def test_compute_plan(global_execution_env):
    """Execution of a compute plan containing multiple traintuples:
    - 1 traintuple executed on node 1
    - 1 traintuple executed on node 2
    - 1 traintuple executed on node 1 depending on previous traintuples
    """
    factory, network = global_execution_env
    session_1 = network.sessions[0].copy()
    session_2 = network.sessions[1].copy()

    dataset_1 = session_1.state.datasets[0]
    dataset_2 = session_2.state.datasets[0]
    objective_1 = session_1.state.objective[0]

    spec = factory.create_algo()
    algo_2 = session_2.add_algo(spec)

    # create compute plan
    cp_spec = factory.create_compute_plan(objective=objective_1)

    # TODO add a testtuple in the compute plan

    traintuple_spec_1 = cp_spec.add_traintuple(
        algo=algo_2,
        dataset=dataset_1,
        data_samples=dataset_1.train_data_sample_keys,
    )

    traintuple_spec_2 = cp_spec.add_traintuple(
        algo=algo_2,
        dataset=dataset_2,
        data_samples=dataset_2.train_data_sample_keys,
    )

    cp_spec.add_traintuple(
        algo=algo_2,
        dataset=dataset_1,
        data_samples=dataset_1.train_data_sample_keys,
        in_models_tuples=[traintuple_spec_1, traintuple_spec_2],
    )

    # submit compute plan and wait for it to complete
    cp = session_1.add_compute_plan(cp_spec)

    traintuples = [
        session_1.get_traintuple(key).future().wait()
        for key in cp.traintuple_keys
    ]

    # check all traintuples are done and check they have been executed on the expected
    # node
    for t in traintuples:
        assert t.status == 'done'

    traintuple_1, traintuple_2, traintuple_3 = traintuples

    assert len(traintuple_3.in_models) == 2

    assert traintuple_1.dataset.worker == session_1.node_id
    assert traintuple_2.dataset.worker == session_2.node_id
    assert traintuple_3.dataset.worker == session_1.node_id


def test_compute_plan_single_session_success(global_execution_env):
    """A compute plan with 3 traintuples and 3 associated testtuples"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintuple + testtuple
    # 2. traintuple + testtuple
    # 3. traintuple + testtuple

    factory, network = global_execution_env
    session = network.sessions[0].copy()

    dataset = session.state.datasets[0]
    data_sample_1, data_sample_2, data_sample_3, _ = dataset.train_data_sample_keys
    objective = session.state.objectives[0]

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    cp_spec = factory.create_compute_plan(objective=objective)

    traintuple_spec_1 = cp_spec.add_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=[data_sample_1]
    )
    cp_spec.add_testtuple(traintuple_spec_1)

    traintuple_spec_2 = cp_spec.add_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=[data_sample_2],
        in_models_tuples=[traintuple_spec_1]
    )
    cp_spec.add_testtuple(traintuple_spec_2)

    traintuple_spec_3 = cp_spec.add_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=[data_sample_3],
        in_models_tuples=[traintuple_spec_2]
    )
    cp_spec.add_testtuple(traintuple_spec_3)

    # Submit compute plan and wait for it to complete
    cp = session.add_compute_plan(cp_spec)

    traintuples = [
        session.get_traintuple(key).future().wait()
        for key in cp.traintuple_keys
    ]

    testtuples = [
        session.get_testtuple(key).future().wait()
        for key in cp.testtuple_keys
    ]

    # All the train/test tuples should succeed
    for t in traintuples + testtuples:
        assert t.status == 'done'

    compute_plan = session.get_compute_plan(cp.compute_plan_id)
    assert cp.compute_plan_id == compute_plan.compute_plan_id
    assert set(cp.traintuple_keys) == set(compute_plan.traintuples)
    assert set(cp.testtuple_keys) == set(compute_plan.testtuples)


def test_compute_plan_single_session_failure(global_execution_env):
    """In a compute plan with 3 traintuples, failing the root traintuple should also
    fail its descendents and the associated testtuples"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintuple + testtuple
    # 2. traintuple + testtuple
    # 3. traintuple + testtuple
    #
    # Intentionally use an invalid (broken) algo.

    factory, network = global_execution_env
    session = network.sessions[0].copy()

    dataset = session.state.datasets[0]
    data_sample_1, data_sample_2, data_sample_3, _ = dataset.train_data_sample_keys
    objective = session.state.objectives[0]

    spec = factory.create_algo(py_script=sbt.factory.INVALID_ALGO_SCRIPT)
    algo = session.add_algo(spec)

    cp_spec = factory.create_compute_plan(objective=objective)

    traintuple_spec_1 = cp_spec.add_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=[data_sample_1]
    )
    cp_spec.add_testtuple(traintuple_spec_1)

    traintuple_spec_2 = cp_spec.add_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=[data_sample_2],
        in_models_tuples=[traintuple_spec_1]
    )
    cp_spec.add_testtuple(traintuple_spec_2)

    traintuple_spec_3 = cp_spec.add_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=[data_sample_3],
        in_models_tuples=[traintuple_spec_2]
    )
    cp_spec.add_testtuple(traintuple_spec_3)

    # Submit compute plan and wait for it to complete
    cp = session.add_compute_plan(cp_spec)

    traintuples = [
        session.get_traintuple(key).future().wait(raises=False)
        for key in cp.traintuple_keys
    ]

    testtuples = [
        session.get_testtuple(key).future().wait(raises=False)
        for key in cp.testtuple_keys
    ]

    # All the train/test tuples should be marked as failed
    for t in traintuples + testtuples:
        assert t.status == 'failed'

    compute_plan = session.get_compute_plan(cp.compute_plan_id)
    assert cp.compute_plan_id == compute_plan.compute_plan_id
    assert set(cp.traintuple_keys) == set(compute_plan.traintuples)
    assert set(cp.testtuple_keys) == set(compute_plan.testtuples)


def test_compute_plan_aggregate_composite_traintuples(factory, session_1, session_2):
    """
    Compute plan version of the `test_aggregate_composite_traintuples` method from `test_execution.py`
    """
    aggregate_worker = session_1.node_id
    sessions = [session_1, session_2]
    number_of_rounds = 2

    # register objectives, datasets, and data samples
    datasets = []
    for s in sessions:
        # register one dataset per node
        spec = factory.create_dataset()
        dataset = s.add_dataset(spec)
        datasets.append(dataset)

        # register one data sample per dataset per round of aggregation
        for _ in range(number_of_rounds):
            spec = factory.create_data_sample(test_only=False, datasets=[dataset])
            s.add_data_sample(spec)
    # reload datasets (to ensure they are properly linked with the created data samples)
    datasets = [
        sessions[i].get_dataset(d.key)
        for i, d in enumerate(list(datasets))
    ]
    # register test data on first node
    spec = factory.create_data_sample(test_only=True, datasets=[datasets[0]])
    test_data_sample = sessions[0].add_data_sample(spec)
    # register objective on first node
    spec = factory.create_objective(
        dataset=datasets[0],
        data_samples=[test_data_sample],
    )
    objective = sessions[0].add_objective(spec)

    # register algos on first node
    spec = factory.create_composite_algo()
    composite_algo = sessions[0].add_composite_algo(spec)
    spec = factory.create_aggregate_algo()
    aggregate_algo = sessions[0].add_aggregate_algo(spec)

    # launch execution
    previous_aggregatetuple_spec = None
    previous_composite_traintuple_specs = []

    cp_spec = factory.create_compute_plan(objective=objective)

    for round_ in range(number_of_rounds):
        # create composite traintuple on each node
        composite_traintuple_specs = []
        for index, dataset in enumerate(datasets):
            kwargs = {}
            if previous_aggregatetuple_spec:
                kwargs = {
                    'in_head_model_tuple': previous_composite_traintuple_specs[index],
                    'in_trunk_model_tuple': previous_aggregatetuple_spec,
                }
            spec = cp_spec.add_composite_traintuple(
                composite_algo=composite_algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0 + round_]],
                **kwargs,
            )
            composite_traintuple_specs.append(spec)

        # create aggregate on its node
        spec = cp_spec.add_aggregatetuple(
            aggregate_algo=aggregate_algo,
            worker=aggregate_worker,
            in_models_tuples=composite_traintuple_specs,
        )

        # save state of round
        previous_aggregatetuple_spec = spec
        previous_composite_traintuple_specs = composite_traintuple_specs

    # last round: create associated testtuple
    for composite_traintuple_spec in previous_composite_traintuple_specs:
        cp_spec.add_testtuple(
            traintuple_spec=composite_traintuple_spec,
        )

    session_1.add_compute_plan(cp_spec).future().wait()


def test_compute_plan_circular_dependency_failure(factory, session):
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    data_sample = session.add_data_sample(spec)

    spec = factory.create_objective(dataset=dataset)
    objective = session.add_objective(spec)

    cp_spec = factory.create_compute_plan(objective=objective)

    traintuple_spec_1 = cp_spec.add_traintuple(
        dataset=dataset,
        algo=algo,
        data_samples=[data_sample]
    )

    traintuple_spec_2 = cp_spec.add_traintuple(
        dataset=dataset,
        algo=algo,
        data_samples=[data_sample]
    )

    traintuple_spec_1.in_models_ids.append(traintuple_spec_2.id)
    traintuple_spec_2.in_models_ids.append(traintuple_spec_1.id)

    # TODO make sur the creation is rejected
    cp = session.add_compute_plan(cp_spec)
    assert False