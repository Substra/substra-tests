import substratools as tools


class NodeMetrics(tools.Metrics):
    def score(self, y_true, y_pred):
        return (y_true == y_pred).mean()


if __name__ == "__main__":
    tools.metrics.execute(NodeMetrics())
