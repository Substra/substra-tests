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
