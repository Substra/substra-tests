import substratools as tools
from sklearn.metrics import accuracy_score


class OrganizationMetrics(tools.Metrics):
    def score(self, y_true, y_pred):
        return accuracy_score(y_true=y_true, y_pred=y_pred)


if __name__ == "__main__":
    tools.metrics.execute(OrganizationMetrics())
