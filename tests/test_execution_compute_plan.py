import pytest

import substratest as sbt


@pytest.mark.skip('may raise MVCC errors')
def test_compute_plan(factory, session_1, session_2):
    """Execution of a compute plan containing multiple traintuples:
    - 1 traintuple executed on node 1
    - 1 traintuple executed on node 2
    - 1 traintuple executed on node 1 depending on previous traintuples
    """

    # TODO create a fixture for initializing network with a set of nodes!

    # add test data samples / dataset / ojective on node 1
    spec = factory.create_dataset()
    dataset_1 = session_1.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset_1])
    test_data_sample_1 = session_1.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset_1])
    data_sample_11 = session_1.add_data_sample(spec)

    spec = factory.create_objective(
        dataset=dataset_1,
        data_samples=[test_data_sample_1],
    )
    objective_1 = session_1.add_objective(spec)

    # refresh dataset_1 as data samples have been added
    dataset_1 = session_1.get_dataset(dataset_1.key)

    # add train data samples / dataset / algo on node 2
    spec = factory.create_dataset()
    dataset_2 = session_2.add_dataset(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset_2])
    data_sample_21 = session_2.add_data_sample(spec)

    spec = factory.create_algo()
    algo_2 = session_2.add_algo(spec)

    # refresh dataset_2 as data samples have been added
    dataset_2 = session_2.get_dataset(dataset_2.key)

    # create compute plan
    cp_spec = factory.create_compute_plan(algo=algo_2, objective=objective_1)

    # TODO add a testtuple in the compute plan

    traintuple_spec_1 = cp_spec.add_traintuple(
        dataset=dataset_1,
        data_samples=[data_sample_11]
    )

    traintuple_spec_2 = cp_spec.add_traintuple(
        dataset=dataset_2,
        data_samples=[data_sample_21]
    )

    _ = cp_spec.add_traintuple(
        dataset=dataset_1,
        data_samples=[data_sample_11],
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


def test_compute_plan_single_session_success(factory, session):
    """A compute plan with 3 traintuples and 3 associated testtuples"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintuple + testtuple
    # 2. traintuple + testtuple
    # 3. traintuple + testtuple

    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    data_sample_1 = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    data_sample_2 = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    data_sample_3 = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    test_data_sample_1 = session.add_data_sample(spec)

    spec = factory.create_objective(
        dataset=dataset,
        data_samples=[test_data_sample_1],
    )
    objective = session.add_objective(spec)

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
        for key in cp.traintuples
    ]

    testtuples = [
        session.get_testtuple(key).future().wait()
        for key in cp.testtuples
    ]

    # All the train/test tuples should succeed
    for t in traintuples + testtuples:
        assert t.status == 'done'


def test_compute_plan_single_session_failure(factory, session):
    """In a compute plan with 3 traintuples, failing the root traintuple should also
    fail its descendents and the associated testtuples"""

    # Create a compute plan with 3 steps:
    #
    # 1. traintuple + testtuple
    # 2. traintuple + testtuple
    # 3. traintuple + testtuple
    #
    # Intentionally use an invalid (broken) algo.

    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_algo(py_script=sbt.factory.INVALID_ALGO_SCRIPT)
    algo = session.add_algo(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    data_sample_1 = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    data_sample_2 = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    data_sample_3 = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    test_data_sample_1 = session.add_data_sample(spec)

    spec = factory.create_objective(
        dataset=dataset,
        data_samples=[test_data_sample_1],
    )
    objective = session.add_objective(spec)

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
        for key in cp.traintuples
    ]

    testtuples = [
        session.get_testtuple(key).future().wait()
        for key in cp.testtuples
    ]

    # All the train/test tuples should be marked as failed
    for t in traintuples + testtuples:
        assert t.status == 'failed'
