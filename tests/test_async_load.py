import asyncio
import concurrent
import functools
import logging
import time

import pytest

logger = logging.getLogger(__name__)


def timeit(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        r = f(*args, **kwargs)
        te = time.time()
        elaps = (te - ts) * 1000
        logger.info(f'{f.__name__}: {elaps:.2f}ms')
        return r
    return wrapper


@pytest.mark.parametrize('nb_requests', [1, 10, 100, 1000])
def test_async_get(nb_requests, factory, session):
    # add algo
    spec = factory.create_algo()
    algo = session.add_algo(spec)

    # get single algo
    @timeit
    def job():
        logger.info('doing request')
        session.get_algo(algo.key)

    # list algos
    async def worker():
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            loop = asyncio.get_event_loop()
            futures = [loop.run_in_executor(executor, job)
                       for i in range(nb_requests)]
            for _ in await asyncio.gather(*futures):
                pass

    loop = asyncio.get_event_loop()
    loop.run_until_complete(worker())


@pytest.mark.parametrize('nb_requests', [1, 10, 100, 1000])
def test_async_post(nb_requests, factory, session):
    # add algo
    specs = [factory.create_algo() for i in range(nb_requests)]

    # get single algo
    @timeit
    def job(spec):
        logger.info('doing request')
        session.add_algo(spec)

    # list algos
    async def worker():
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            loop = asyncio.get_event_loop()
            futures = [loop.run_in_executor(executor, job, specs[i])
                       for i in range(nb_requests)]
            for _ in await asyncio.gather(*futures):
                pass

    loop = asyncio.get_event_loop()
    loop.run_until_complete(worker())
