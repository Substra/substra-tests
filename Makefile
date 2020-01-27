pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

test: pyclean
	pytest tests -rs -v --durations=0

fast-test: pyclean
	pytest tests -rs -v --durations=0 -m "not slow"

install:
	pip3 install -r requirements.txt
