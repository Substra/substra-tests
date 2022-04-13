override CWD := $(shell basename $(shell pwd))
DOCKER_IMG := $(CWD)
DOCKER_TAG := latest
PARALLELISM := 5

.PHONY: pyclean test test-minimal install docker

pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

test: test-remote test-local

test-remote: pyclean
	pytest tests -rs -v --durations=0 -m "not workflows" -n $(PARALLELISM)

test-minimal: pyclean
	pytest tests -rs -v --durations=0 -m "not slow and not workflows" -n $(PARALLELISM)

test-local: test-subprocess test-docker

test-docker: pyclean
	DEBUG_SPAWNER=docker pytest tests -rs -v --durations=0 -m "not workflows" --local

test-subprocess: pyclean
	DEBUG_SPAWNER=subprocess pytest tests -rs -v --durations=0 -m "not workflows and not subprocess_skip" --local

test-workflows: pyclean
	pytest tests -v --durations=0 -m "workflows"

test-ci: test-remote test-workflows

test-all: test-local test-ci

install:
	pip3 install -r requirements.txt

docker:
	docker build -f docker/substra-tests/Dockerfile .	-t $(DOCKER_IMG):$(DOCKER_TAG)
