import pytest

from substra.sdk.models import Status

import substratest as sbt
from . import settings


# CELERY_TASK_MAX_RETRIES uses the remote backend settings
# no check on permissions with the local backend
@pytest.mark.remote_only
@pytest.mark.parametrize('fail_count,status', (
    (settings.CELERY_TASK_MAX_RETRIES, Status.done),
    (settings.CELERY_TASK_MAX_RETRIES + 1, Status.failed),
))
def test_execution_retry_on_fail(fail_count, status, factory, client, default_dataset):
    """Execution of a traintuple which fails on the N first tries, and suceeds on the N+1th try"""

    # This test ensures the compute task retry mechanism works correctly.
    #
    # It executes an algorithm that `raise`s on the first N runs, and then
    # succeeds.
    #
    # /!\ This test should ideally be part of the substra-backend project,
    #     not substra-tests. For the sake of expendiency, we're keeping it
    #     as part of substra-tests for now, but we intend to re-implement
    #     it in substra-backend instead eventually.
    # /!\ This test makes use of the "local" folder to keep track of a counter.
    #     This is a hack to make the algo raise or succeed depending on the retry
    #     count. Ideally, we would use a more elegant solution.
    # /!\ This test doesn't validate that an error in the docker build phase (of
    #     the compute task execution) triggers a retry. Ideally, it's that case that
    #     would be tested, since errors in the docker build are the main use-case
    #     the retry feature was build for.

    retry_algo_snippet_toreplace = """
    tools.algo.execute(TestAlgo())"""

    retry_snippet_replacement = f"""
    counter_path = "/sandbox/local/counter"
    counter = 0
    try:
        with open(counter_path) as f:
            counter = int(f.read())
    except IOError:
        pass # file doesn't exist yet

    # Fail if the counter is below the retry count
    if counter < {fail_count}:
        counter = counter + 1
        with open(counter_path, 'w') as f:
            f.write(str(counter))
        raise Exception("Intentionally keep on failing until we have failed {fail_count} time(s). The algo has now \
            failed " + str(counter) + " time(s).")

    # The counter is greater than the retry count
    tools.algo.execute(TestAlgo())"""

    py_script = sbt.factory.DEFAULT_ALGO_SCRIPT.replace(retry_algo_snippet_toreplace, retry_snippet_replacement)
    spec = factory.create_algo(py_script)
    algo = client.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
        rank=0,  # make sure it's part of a compute plan, so we have access to the /sandbox/local
                 # folder (that's where we store the counter)
    )
    traintuple = client.add_traintuple(spec)
    traintuple = client.wait(traintuple, raises=False)

    # Assuming that, on the backend, CELERY_TASK_MAX_RETRIES is set to 1, the algo
    # should be retried up to 1 time(s) (i.e. max 2 attempts in total)
    # - if it fails less than 2 times, it should be marked as "done"
    # - if it fails 2 times or more, it should be marked as "failed"
    assert traintuple.status == status
