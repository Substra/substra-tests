import os
import shutil

import torch
import torch.nn.functional as F

import substratools as tools

_INPUT_SAMPLE_SIZE = 21632
_OUT_SAMPLE_SIZE = 10
_NB_CHANNELS = 32


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, _NB_CHANNELS, kernel_size=3)
        self.fc = torch.nn.Linear(_INPUT_SAMPLE_SIZE, _OUT_SAMPLE_SIZE)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, _INPUT_SAMPLE_SIZE)
        x = self.fc(x)
        return x


class ModelAggregator(tools.AggregateAlgo):
    """
    Algo that aggregates models by simply averaging them as in FedAvg
    """

    def aggregate(self, inmodels, rank):
        # get layers
        model = inmodels[0]
        model_state_dict = model.state_dict()

        # average weights
        for layer in model_state_dict:
            weights = []
            for _model in inmodels:
                weights.append(_model.state_dict()[layer])
            weights = torch.stack(weights)
            model_state_dict[layer] = torch.mean(weights, dim=0)

        model.load_state_dict(model_state_dict)

        return model

    def load_model(self, path):
        return torch.load(path)

    def save_model(self, model, path):
        torch.save(model, path + '.h5')
        shutil.move(path + '.h5', path)
        assert os.path.isfile(path)

    def load_head_model(self, path):
        return self.load_model(path)

    def save_head_model(self, model, path):
        self.save_model(model, path)

    def load_trunk_model(self, path):
        return self.load_model(path)

    def save_trunk_model(self, model, path):
        self.save_model(model, path)


if __name__ == '__main__':
    tools.algo.execute(ModelAggregator())