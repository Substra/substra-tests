override CWD := $(shell basename $(shell pwd))
DOCKER_IMG := $(CWD)
DOCKER_TAG := latest
SUBSTRA_GIT_REPO := https://github.com/SubstraFoundation/substra.git
SUBSTRA_GIT_REF := master
PARALLELISM := 5

.PHONY: pyclean test test-minimal install docker

pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

test: test-remote test-local

test-remote: pyclean
	pytest tests -rs -v --durations=0 -n $(PARALLELISM)

test-minimal: pyclean
	pytest tests -rs -v --durations=0 -m "not slow" -n $(PARALLELISM)

test-local: pyclean
	pytest tests -rs -v --durations=0 --local

install:
	pip3 install -r requirements.txt

# Usage:
#   make docker
#   make docker DOCKER_TAG=master SUBSTRA_GIT_REF=master
#   make docker DOCKER_TAG=1.0    SUBSTRA_GIT_REF=v1.0
#   make docker DOCKER_TAG=pr_123 SUBSTRA_GIT_REF=refs/pull/123/head
#   make docker DOCKER_TAG=commit SUBSTRA_GIT_REF=da39a3ee5e6b4b0d3255bfef95601890afd80709
docker:
	docker build -f docker/substra-tests/Dockerfile .	-t $(DOCKER_IMG):$(DOCKER_TAG) \
		--build-arg SUBSTRA_GIT_REPO=$(SUBSTRA_GIT_REPO) \
		--build-arg SUBSTRA_GIT_REF=$(SUBSTRA_GIT_REF)
