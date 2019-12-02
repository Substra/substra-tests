import pytest

import substratest as sbt


@pytest.mark.skip('may raise MVCC errors')
def test_compute_plan(data_network):
    """Execution of a compute plan containing multiple traintuples:
    - 1 traintuple executed on node 1
    - 1 traintuple executed on node 2
    - 1 traintuple executed on node 1 depending on previous traintuples
    """
    factory, network = data_network
    session_1, session_2 = data_network.sessions

    dataset_1 = session_1.state.datasets[0]
    dataset_2 = session_2.state.datasets[0]
    objective_1 = session_1.state.objective[0]
    train_data_samples_1 = session_1.state.train_data_samples
    train_data_samples_2 = session_2.state.train_data_samples

    spec = factory.create_algo()
    algo_2 = session_2.add_algo(spec)

    # create compute plan
    cp_spec = factory.create_compute_plan(algo=algo_2, objective=objective_1)

    # TODO add a testtuple in the compute plan

    traintuple_spec_1 = cp_spec.add_traintuple(
        dataset=dataset_1,
        data_samples=train_data_samples_1,
    )

    traintuple_spec_2 = cp_spec.add_traintuple(
        dataset=dataset_2,
        data_samples=train_data_samples_2,
    )

    _ = cp_spec.add_traintuple(
        dataset=dataset_1,
        data_samples=train_data_samples_1,
        traintuple_specs=[traintuple_spec_1, traintuple_spec_2],
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


def test_compute_plan_single_session_success(data_network):
    """A compute plan with 3 traintuples and 3 associated testtuples"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintuple + testtuple
    # 2. traintuple + testtuple
    # 3. traintuple + testtuple

    factory, network = data_network
    session = network.sessions[0]

    dataset = session.state.datasets[0]
    data_sample_1, data_sample_2, data_sample_3, _ = session.state.train_data_samples
    objective = session.state.objectives[0]

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    cp_spec = factory.create_compute_plan(algo=algo, objective=objective)

    traintuple_spec_1 = cp_spec.add_traintuple(
        dataset=dataset,
        data_samples=[data_sample_1]
    )
    cp_spec.add_testtuple(traintuple_spec_1)

    traintuple_spec_2 = cp_spec.add_traintuple(
        dataset=dataset,
        data_samples=[data_sample_2],
        traintuple_specs=[traintuple_spec_1]
    )
    cp_spec.add_testtuple(traintuple_spec_2)

    traintuple_spec_3 = cp_spec.add_traintuple(
        dataset=dataset,
        data_samples=[data_sample_3],
        traintuple_specs=[traintuple_spec_2]
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


def test_compute_plan_single_session_failure(data_network):
    """In a compute plan with 3 traintuples, failing the root traintuple should also
    fail its descendents and the associated testtuples"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintuple + testtuple
    # 2. traintuple + testtuple
    # 3. traintuple + testtuple
    #
    # Intentionally use an invalid (broken) algo.

    factory, network = data_network
    session = network.sessions[0]

    dataset = session.state.datasets[0]
    data_sample_1, data_sample_2, data_sample_3, _ = session.state.train_data_samples
    objective = session.state.objectives[0]

    spec = factory.create_algo(py_script=sbt.factory.INVALID_ALGO_SCRIPT)
    algo = session.add_algo(spec)

    cp_spec = factory.create_compute_plan(algo=algo, objective=objective)

    traintuple_spec_1 = cp_spec.add_traintuple(
        dataset=dataset,
        data_samples=[data_sample_1]
    )
    cp_spec.add_testtuple(traintuple_spec_1)

    traintuple_spec_2 = cp_spec.add_traintuple(
        dataset=dataset,
        data_samples=[data_sample_2],
        traintuple_specs=[traintuple_spec_1]
    )
    cp_spec.add_testtuple(traintuple_spec_2)

    traintuple_spec_3 = cp_spec.add_traintuple(
        dataset=dataset,
        data_samples=[data_sample_3],
        traintuple_specs=[traintuple_spec_2]
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
