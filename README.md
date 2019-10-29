# substra-tests

Substra end-to-end (e2e) tests

# Prepare

```
pip3 install -r requirements.txt
```

# Run

With skaffold

```
make test
```

With docker-compose

```
SUBSTRAT_CONFIG_FILE=values-docker-compose.yaml make test
```

# Tests Configuration

The tests suite requires a Substra network up and running. The network can be started
either with skaffold, manually or through docker-compose.

To execute the tests, the network configuration must be defined. There are currently
two different ways to specify it:
- in a dedicated configuration yaml file (default mode, using the file *values.yaml*)
- from a backend skaffold yaml file

The configuration YAML path can be overriden using the `SUBSTRAT_CONFIG_FILEPATH` environment
variable.

To use a skaffold file, the environment variable `SUBSTRAT_SKAFFOLD_FILEPATH` must be set. For instance:

```
SUBSTRAT_SKAFFOLD_FILEPATH=../substra-backend/skaffold.yaml make test
```
