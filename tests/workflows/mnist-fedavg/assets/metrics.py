import numpy as np
import substratools as tools
from sklearn.metrics import accuracy_score


@tools.register
def score(inputs, outputs, task_properties):
    y_true = inputs["datasamples"]["y"]
    y_pred = _load_predictions(inputs["predictions"])
    perf = accuracy_score(y_true=y_true, y_pred=y_pred)
    tools.save_performance(perf, outputs["performance"])


def _load_predictions(path):
    """Load predictions."""
    return np.load(path)


if __name__ == "__main__":
    tools.execute(score)
