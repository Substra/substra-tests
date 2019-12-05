import pytest
import substra
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
        in_models=[traintuple_spec_1, traintuple_spec_2],
    )

    # submit compute plan and wait for it to complete
    cp = session_1.add_compute_plan(cp_spec).future().wait()

    traintuples = cp.list_traintuple(session_1)

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
        in_models=[traintuple_spec_1]
    )
    cp_spec.add_testtuple(traintuple_spec_2)

    traintuple_spec_3 = cp_spec.add_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=[data_sample_3],
        in_models=[traintuple_spec_2]
    )
    cp_spec.add_testtuple(traintuple_spec_3)

    # Submit compute plan and wait for it to complete
    cp = session.add_compute_plan(cp_spec).future().wait()

    # All the train/test tuples should succeed
    for t in cp.list_traintuple(session) + cp.list_testtuple(session):
        assert t.status == 'done'


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
        in_models=[traintuple_spec_1]
    )
    cp_spec.add_testtuple(traintuple_spec_2)

    traintuple_spec_3 = cp_spec.add_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=[data_sample_3],
        in_models=[traintuple_spec_2]
    )
    cp_spec.add_testtuple(traintuple_spec_3)

    # Submit compute plan and wait for it to complete
    cp = session.add_compute_plan(cp_spec).future().wait()

    traintuples = cp.list_traintuple(session)
    testtuples = cp.list_testtuple(session)

    # All the train/test tuples should be marked as failed
    for t in traintuples + testtuples:
        assert t.status == 'failed'


def test_compute_plan_aggregate_composite_traintuples(global_execution_env):
    """
    Compute plan version of the `test_aggregate_composite_traintuples` method from `test_execution.py`
    """
    factory, network = global_execution_env
    sessions = [s.copy() for s in network.sessions]

    aggregate_worker = sessions[0].node_id
    number_of_rounds = 2

    # register objectives, datasets, and data samples
    datasets = sessions[0].state.datasets + sessions[1].state.datasets
    objective = sessions[0].state.objectives[0]

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
                    'in_head_model': previous_composite_traintuple_specs[index],
                    'in_trunk_model': previous_aggregatetuple_spec,
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
            in_models=composite_traintuple_specs,
        )

        # save state of round
        previous_aggregatetuple_spec = spec
        previous_composite_traintuple_specs = composite_traintuple_specs

    # last round: create associated testtuple
    for composite_traintuple_spec in previous_composite_traintuple_specs:
        cp_spec.add_testtuple(
            traintuple_spec=composite_traintuple_spec,
        )

    cp = sessions[0].add_compute_plan(cp_spec).future().wait()
    tuples = (cp.list_traintuple(sessions[0]) +
              cp.list_composite_traintuple(sessions[0]) +
              cp.list_aggregatetuple(sessions[0]) +
              cp.list_testtuple(sessions[0]))
    for t in tuples:
        assert t.status == 'done'


def test_compute_plan_circular_dependency_failure(global_execution_env):
    factory, network = global_execution_env
    session = network.sessions[0].copy()

    dataset = session.state.datasets[0]
    objective = session.state.objectives[0]

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    cp_spec = factory.create_compute_plan(objective=objective)

    traintuple_spec_1 = cp_spec.add_traintuple(
        dataset=dataset,
        algo=algo,
        data_samples=dataset.train_data_sample_keys
    )

    traintuple_spec_2 = cp_spec.add_traintuple(
        dataset=dataset,
        algo=algo,
        data_samples=dataset.train_data_sample_keys
    )

    traintuple_spec_1.in_models_ids.append(traintuple_spec_2.id)
    traintuple_spec_2.in_models_ids.append(traintuple_spec_1.id)

    with pytest.raises(substra.exceptions.InvalidRequest) as e:
        session.add_compute_plan(cp_spec)

    assert 'missing dependency among inModels IDs' in str(e.value)
