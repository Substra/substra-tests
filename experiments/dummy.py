import random


def data_sample_generator():
    encoding = 'utf-8'
    count = -1

    while True:
        count += 1
        rdm = random.random()

        content = f'0,{count}'.encode(encoding)
        content = f'# random={rdm} \n'.encode(encoding) + content
        yield content


OPENER_SCRIPT = """
import json
import substratools as tools
class TestOpener(tools.Opener):
    def get_X(self, folders):
        return folders
    def get_y(self, folders):
        return folders
    def fake_X(self):
        return 'fakeX'
    def fake_y(self):
        return 'fakey'
    def get_predictions(self, path):
        with open(path) as f:
            return json.load(f)
    def save_predictions(self, y_pred, path):
        with open(path, 'w') as f:
            return json.dump(y_pred, f)
"""

METRICS_SCRIPT = """
import json
import substratools as tools
class TestMetrics(tools.Metrics):
    def score(self, y_true, y_pred):
        return 101
if __name__ == '__main__':
    tools.metrics.execute(TestMetrics())
"""

ALGO_SCRIPT = f"""
import json
import substratools as tools
class TestAlgo(tools.Algo):
    def train(self, X, y, models, rank):
        return [0, 42], [0, 1]
    def predict(self, X, model):
        return [0, 99]
    def load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)
if __name__ == '__main__':
    tools.algo.execute(TestAlgo())
"""

AGGREGATE_ALGO_SCRIPT = f"""
import json
import substratools as tools
class TestAggregateAlgo(tools.AggregateAlgo):
    def aggregate(self, models, rank):
        return [0, 66]
    def load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)
if __name__ == '__main__':
    tools.algo.execute(TestAggregateAlgo())
"""

COMPOSITE_ALGO_SCRIPT = f"""
import json
import substratools as tools
class TestCompositeAlgo(tools.CompositeAlgo):
    def train(self, X, y, head_model, trunk_model, rank):
        return [0, 42], [0, 1], [0, 2]
    def predict(self, X, head_model, trunk_model):
        return [0, 99]
    def load_head_model(self, path):
        return self._load_model(path)
    def save_head_model(self, model, path):
        return self._save_model(model, path)
    def load_trunk_model(self, path):
        return self._load_model(path)
    def save_trunk_model(self, model, path):
        return self._save_model(model, path)
    def _load_model(self, path):
        with open(path) as f:
            return json.load(f)
    def _save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)
if __name__ == '__main__':
    tools.algo.execute(TestCompositeAlgo())
"""

INVALID_ALGO_SCRIPT = ALGO_SCRIPT.replace('train', 'naitr')

DEFAULT_SUBSTRATOOLS_VERSION = '0.4.0'

METRICS_DOCKERFILE = f"""
FROM substrafoundation/substra-tools:{DEFAULT_SUBSTRATOOLS_VERSION}
COPY metrics.py .
ENTRYPOINT ["python3", "metrics.py"]
"""

ALGO_DOCKERFILE = f"""
FROM substrafoundation/substra-tools:{DEFAULT_SUBSTRATOOLS_VERSION}
COPY algo.py .
ENTRYPOINT ["python3", "algo.py"]
"""
