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

The tests suite:
- Must be easy to configure through settings files to define the substra network to target
- Must be independent from the substra network
  - Substra network must be started separately
  - Substra network can be located locally / on the cloud
  - Substra network can be started through docker-compose or through kubenertes
- Requires a setup with at least 2 organisations
- Should be able to run multiple times without restarting the network

Each new test:
- Should have a deterministic behavior (not fail randomly)
- Must be fast to run:
  - avoid adding latency
  - group related long running tests
- Should not complete before all the tuples it created have been executed: it is required to ensure that the next test will be launched with a susbtra network ready to execute new tuples
