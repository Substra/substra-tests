import pytest


@pytest.mark.local_only
@pytest.mark.slow
def test_traintuple_fake_data(factory, client, default_dataset, default_objective):
    """Execution of a traintuple, a following testtuple and a following traintuple."""

    spec = factory.create_algo()
    algo = client.add_algo(spec)

    # create traintuple
    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=list(),
        metadata={"foo": "bar"},
        fake_data=True,
        n_fake_samples=1,
    )
    traintuple = client.add_traintuple(spec).future().wait()
    print(traintuple.log.replace('\n', ''))

    # create testtuple
    spec = factory.create_testtuple(
        objective=default_objective,
        traintuple=traintuple,
        fake_data=True,
        n_fake_samples=1,
    )
    testtuple = client.add_testtuple(spec).future().wait()
    print(testtuple.log)
    assert testtuple.dataset.perf == 3

    # add a traintuple depending on first traintuple
    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=list(),
        traintuples=[traintuple],
        metadata=None,
        fake_data=True,
        n_fake_samples=1,
    )
    traintuple = client.add_traintuple(spec).future().wait()
    print(traintuple.log.replace('\n', ''))

    # create testtuple
    spec = factory.create_testtuple(
        objective=default_objective,
        traintuple=traintuple,
        fake_data=True,
        n_fake_samples=1,
    )
    testtuple = client.add_testtuple(spec).future().wait()
    print(testtuple.log.replace('\n', ''))
    assert testtuple.dataset.perf == 6


@pytest.mark.local_only
@pytest.mark.slow
def test_composite_traintuple_fake_data(factory, client, default_dataset, default_objective):
    """Execution of a traintuple, a following testtuple and a following traintuple."""

    spec = factory.create_composite_algo()
    algo = client.add_composite_algo(spec)

    # first composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=list(),
        fake_data=True,
        n_fake_samples=1,
    )
    composite_traintuple_1 = client.add_composite_traintuple(spec).future().wait()
    print(composite_traintuple_1.log.replace('\n', ''))

    # second composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=list(),
        head_traintuple=composite_traintuple_1,
        trunk_traintuple=composite_traintuple_1,
        fake_data=True,
        n_fake_samples=1,
    )
    composite_traintuple_2 = client.add_composite_traintuple(spec).future().wait()
    print(composite_traintuple_2.log.replace('\n', ''))

    # add a 'composite' testtuple
    spec = factory.create_testtuple(
        objective=default_objective,
        traintuple=composite_traintuple_2,
        fake_data=True,
        n_fake_samples=1,
    )
    testtuple = client.add_testtuple(spec).future().wait()
    print(testtuple.log.replace('\n', ''))
    assert testtuple.dataset.perf == 48


@pytest.mark.local_only
@pytest.mark.slow
def test_compute_plan_fake_data(factory, client, default_dataset, default_objective):
    """Execution of a compute plan containing multiple traintuples:
    - 1 traintuple executed on node 1
    - 1 traintuple executed on node 2
    - 1 traintuple executed on node 1 depending on previous traintuples
    - 1 testtuple executed on node 1 depending on the last traintuple
    """

    spec = factory.create_algo()
    algo_2 = client.add_algo(spec)

    # create compute plan
    cp_spec = factory.create_compute_plan(
        tag='foo',
        metadata={"foo": "bar"},
    )

    traintuple_spec_1 = cp_spec.add_traintuple(
        algo=algo_2,
        dataset=default_dataset,
        data_samples=list(),
        metadata=None,
        fake_data=True,
        n_fake_samples=1,
    )

    traintuple_spec_2 = cp_spec.add_traintuple(
        algo=algo_2,
        dataset=default_dataset,
        data_samples=list(),
        in_models=[traintuple_spec_1],
        metadata={},
        fake_data=True,
        n_fake_samples=1,
    )

    cp_spec.add_testtuple(
        objective=default_objective,
        traintuple_spec=traintuple_spec_2,
        metadata={'foo': 'bar'},
        fake_data=True,
        n_fake_samples=1,
    )

    # submit compute plan and wait for it to complete
    cp = client.add_compute_plan(cp_spec).future().wait()

    # check testtuple perf
    testtuples = cp.list_testtuple()
    testtuple = testtuples[0]
    assert testtuple.dataset.perf == 6
