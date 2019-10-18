import substra

import pytest

import substratest as sbt


def test_tuples_execution_on_same_node(factory, session):
    """Execution of a traintuple, a following testtuple and a following traintuple."""
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    test_data_sample = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    train_data_sample = session.add_data_sample(spec)

    spec = factory.create_objective(
        dataset=dataset,
        data_samples=[test_data_sample],
    )
    objective = session.add_objective(spec)

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    # create traintuple
    spec = factory.create_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
    )
    traintuple = session.add_traintuple(spec).future().wait()
    assert traintuple.status == 'done'
    assert traintuple.out_model is not None

    # create testtuple
    # don't create it before to avoid MVCC errors
    spec = factory.create_testtuple(traintuple=traintuple)
    testtuple = session.add_testtuple(spec).future().wait()
    assert testtuple.status == 'done'

    # add a traintuple depending on first traintuple
    spec = factory.create_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
        traintuples=[traintuple],
    )
    traintuple = session.add_traintuple(spec).future().wait()
    assert traintuple.status == 'done'
    assert len(traintuple.in_models) == 1


@pytest.mark.skip('conflict not returned in chaincode')
def test_federated_learning_workflow(factory, session):
    """Test federated learning workflow."""
    # create test environment
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    test_data_sample = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    train_data_sample = session.add_data_sample(spec)

    spec = factory.create_objective(
        dataset=dataset,
        data_samples=[test_data_sample],
    )
    objective = session.add_objective(spec)

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    # create traintuple with rank 0
    spec = factory.create_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
        tag='foo',
        rank=0,
    )
    traintuple_1 = session.add_traintuple(spec).future().wait()
    assert traintuple_1.status == 'done'
    assert traintuple_1.out_model is not None
    assert traintuple_1.tag == 'foo'
    assert traintuple_1.compute_plan_id is not None

    with pytest.raises(substra.exceptions.AlreadyExists):
        session.add_traintuple(spec)

    # create traintuple with rank 1
    spec = factory.create_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
        traintuples=[traintuple_1],
        tag='foo',
        compute_plan_id=traintuple_1.compute_plan_id,
        rank=1,
    )
    traintuple_2 = session.add_traintuple(spec).future().wait()
    assert traintuple_2.status == 'done'
    assert traintuple_2.out_model is not None
    assert traintuple_2.tag == 'foo'
    assert traintuple_2.compute_plan_id == traintuple_1.compute_plan_id

    with pytest.raises(substra.exceptions.AlreadyExists):
        session.add_traintuple(spec)


def test_tuples_execution_on_different_nodes(factory, session_1, session_2):
    """Execution of a traintuplute on node 1 and the following testtuple on node 2."""
    # add test data samples / dataset / ojective on node 1
    spec = factory.create_dataset()
    dataset_1 = session_1.add_dataset(spec)
    spec = factory.create_data_sample(test_only=True, datasets=[dataset_1])
    test_data_sample_1 = session_1.add_data_sample(spec)
    spec = factory.create_objective(
        dataset=dataset_1,
        data_samples=[test_data_sample_1],
    )
    objective_1 = session_1.add_objective(spec)

    # add train data samples / dataset / algo on node 2
    spec = factory.create_dataset()
    dataset_2 = session_2.add_dataset(spec)
    spec = factory.create_data_sample(test_only=False, datasets=[dataset_2])
    train_data_sample_2 = session_2.add_data_sample(spec)
    spec = factory.create_algo()
    algo_2 = session_2.add_algo(spec)

    # add traintuple on node 2; should execute on node 2 (dataset located on node 2)
    spec = factory.create_traintuple(
        algo=algo_2,
        objective=objective_1,
        dataset=dataset_2,
        data_samples=[train_data_sample_2],
    )
    traintuple = session_1.add_traintuple(spec).future().wait()
    assert traintuple.status == 'done'
    assert traintuple.out_model is not None
    assert traintuple.dataset.worker == session_2.node_id

    # add testtuple; should execute on node 1 (objective dataset is located on node 1)
    spec = factory.create_testtuple(traintuple=traintuple)
    testtuple = session_1.add_testtuple(spec).future().wait()
    assert testtuple.status == 'done'
    assert testtuple.dataset.worker == session_1.node_id


def test_traintuple_execution_failure(factory, session):
    """Invalid algo script is causing traintuple failure."""
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    test_data_sample = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    train_data_sample = session.add_data_sample(spec)

    spec = factory.create_objective(
        dataset=dataset,
        data_samples=[test_data_sample],
    )
    objective = session.add_objective(spec)

    invalid_script = sbt.factory.DEFAULT_ALGO_SCRIPT.replace('train', 'naitr')
    spec = factory.create_algo(py_script=invalid_script)
    algo = session.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
    )
    traintuple = session.add_traintuple(spec).future().wait()
    assert traintuple.status == 'failed'
    assert traintuple.out_model is None


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
