override CWD := $(shell basename $(shell pwd))
DOCKER_IMG := $(CWD)
DOCKER_TAG := latest
PARALLELISM := 5

.PHONY: pyclean test test-minimal install docker

pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

test: test-remote test-local

test-remote: test-remote-sdk test-remote-workflows

test-remote-sdk: pyclean
	pytest tests -rs -v --durations=0 -m "not workflows" -n $(PARALLELISM) --log-level=INFO

test-remote-workflows: pyclean
	pytest tests -v --durations=0 -m "workflows" --log-level=INFO

test-minimal: pyclean
	pytest tests -rs -v --durations=0 -m "not slow and not workflows" -n $(PARALLELISM) --log-level=INFO

test-local: test-subprocess test-docker test-subprocess-workflows

test-docker: pyclean
	pytest tests -rs -v --durations=0 -m "not workflows" --mode=docker

test-subprocess: pyclean
	pytest tests -rs -v --durations=0 -m "not workflows and not subprocess_skip" --mode=subprocess --log-level=INFO

test-subprocess-workflows: pyclean
	pytest tests -v --durations=0 -m "workflows" --mode=subprocess --log-level=INFO

test-all: test-local test-remote

install:
	pip3 install -r requirements.txt

docker:
	docker build -f docker/substra-tests/Dockerfile .	-t $(DOCKER_IMG):$(DOCKER_TAG)
