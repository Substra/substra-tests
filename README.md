# substra-tests [![Python](https://github.com/Substra/substra-tests/actions/workflows/python.yml/badge.svg)](https://github.com/Substra/substra-tests/actions/workflows/python.yml)

Substra end-to-end (e2e) tests

# Prerequisites

This project requires python 3.6+.

Install tests dependencies:

```
pip3 install --no-cache-dir "git+ssh://git@github.com/Substra/substra.git@main"
pip3 install -r requirements.txt
```

The tests suite requires a Substra network up and running to test the remote backend.
The network can be started with skaffold (Kubernetes) or manually with helm charts.

The substra project is needed for running the tests.
It can be found [here](https://github.com/SubstraFoundation/substra)

You will need to install it thanks to the `pip` binary.

# Run the tests

The tests can run both on the remote backend and the local backend. To run the complete
test suite on both backends:

```bash
make test
```

# Run the tests on the remote backend

The network configuration is described in a yaml file.

A default configuration file is available:
- `values.yaml` (default): for networks started with Kubernetes

To run the tests using the default `values.yaml` file:

```
make test-remote
```

To run the tests using the provided `local-values.yaml` (or a custom config file):

```
SUBSTRA_TESTS_CONFIG_FILEPATH=local-values.yaml make test-remote
```

## Minimal mode

Since tests can take a long time to run, some of them are marked as slow. You can run the "fast" ones with:

```
make test-minimal
```

Note that `test_compute_plan` from `test_execution_compute_plan.py` is not marked as slow even though it takes several
seconds to complete. This is because it covers a very basic use case of the platform and is needed to ensure basic
features aren't broken.

# Run the tests on the local backend

The network configuration is described in a yaml file: `local-backend-values.yaml` and cannot be changed.

To run the tests using on the local backend:

```
make test-local
```

Some tests are skipped in this mode as they need the remote backend to run.

Use `--local` option to run a single test with `pytest`.

```
pytest --local tests/test_execution_compute_plan.py::test_compute_plan_single_client_success
```

# Test design guidelines

When adding or modifying tests, please follow these guidelines:

1. The complete test suite must be independent from the substra network
   - The substra network must be started prior to (and independently from) executing the tests
   - The substra network can be running locally / on the cloud
   - The substra network must be started through kubernetes
1. It should be possible to run the test suite multiple times without restarting the substra network
1. Each test must have a deterministic behavior (must not fail randomly)
1. Each test must be fast to run:
   - Avoid tests that take a lot of time to complete
   - Group related long running tests (when it makes sense)
1. Each test should not complete before all the tasks it created have been executed. This requirement ensures that the next test will be launched with a substra network ready to execute new tasks
1. Tests must not use hardcoded network configuration settings/values. Use settings files instead (e.g. `values.yaml`)
1. Tests should target a substra network with at least 2 organizations
1. By default, a test must pass on the remote and local backend. If the test is specific to one backend, add the corresponding mark.

## Code formatting

You can opt into auto-formatting of code on pre-commit using [Black](https://github.com/psf/black) and [isort](https://github.com/PyCQA/isort).

This relies on hooks managed by [pre-commit](https://pre-commit.com/), which you can set up as follows.

Install [pre-commit](https://pre-commit.com/), then run:

```sh
pre-commit install
```

## How to generate the changelog

The changelog is managed with [towncrier](https://towncrier.readthedocs.io/en/stable/index.html).
To add a new entry in the changelog, add a file in the `changes` folder. The file name should have the following structure:
`<unique_id>.<change_type>`.
The `unique_id` is a unique identifier, we currently use the PR number.
The `change_type` can be of the following types: `added`, `changed`, `removed`, `fixed`.

To generate the changelog (for example during a release), use the following command (you must have the dev dependencies installed):

```
towncrier build --version=<x.y.z>
```

You can use the `--draft` option to see what would be generated without actually writing to the changelog (and without removing the fragments).
