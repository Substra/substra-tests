# substra-tests

Substra end-to-end (e2e) tests

# Prerequisites

Install tests dependencies:

```
pip3 install -r requirements.txt
```

The tests suite requires a Substra network up and running. The network can be started
either with skaffold (Kubernetes), with docker-compose, or manually.

The substra project is needed for running the tests.  
It can be found [here](https://github.com/SubstraFoundation/substra)

You will need to install it thanks to the `pip` binary.

# Run the tests

## Using a network configuration file

This is the mode used by default. Two files are currently available:
- `values.yaml` (default): for networks started with Kubernetes
- `local-values.yaml`: for networks started with docker-compose, or started manually

To run the tests using the default `values.yaml` file:

```
make test
```

To run the tests using the provided `local-values.yaml` (or a custom config file):

```
SUBSTRA_TESTS_CONFIG_FILEPATH=local-values.yaml make test
```

## Using a skaffold file

It is possible to use a `substra-backend` skaffold file as the source for the network configuration:
```
SUBSTRA_TESTS_SKAFFOLD_FILEPATH=$SUBSTRA_SOURCE/substra-backend/skaffold.yaml make test
```

# Test design guidelines

When adding or modifying tests, please follow these guidelines.

The test suite:
- Must be independent from the substra network
  - The substra network is started prior to, and independently from the tests being run
  - The substra network can be running locally / on the cloud
  - The substra network can be started through docker-compose or through kubernetes
- It should be possible to run the test suite multiple times without restarting the substra network
- Must not use hardcoded network configuration settings/values. Use settings files instead (e.g. `values.yaml`)
- Should target a substra network with at least 2 organisations

Each new test:
- Should have a deterministic behavior (not fail randomly)
- Must be fast to run:
  - Avoid tests that take a lot of time to complete
  - Group related long running tests (when it makes sense)
- Should not complete before all the tuples it created have been executed. This requirement ensures that the next test will be launched with a susbtra network ready to execute new tuples
