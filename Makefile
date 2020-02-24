DOCKER_TAG := latest
DOCKER_TAG_SUBSTRA := latest

.PHONY: pyclean test test-minimal install docker

pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

test: pyclean
	pytest tests -rs -v --durations=0

test-minimal: pyclean
	pytest tests -rs -v --durations=0 -m "not slow"

install:
	pip3 install -r requirements.txt

docker:
	docker build -f docker/Dockerfile . -t substra-tests:$(DOCKER_TAG) --build-arg SUBSTRA_VERSION=$(DOCKER_TAG_SUBSTRA)