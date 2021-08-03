import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np

import substratools as tools

_INPUT_SAMPLE_SIZE = 21632
_OUT_SAMPLE_SIZE = 10
_NB_CHANNELS = 32
_SEED = 42
_BATCH_SIZE = 32
_NUM_UPDATES = 50


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


def _fit(model, X, y, batch_size, num_updates, rank):
    assert len(X) <= batch_size * num_updates
    tensor_x = torch.FloatTensor(X)  # transform to torch tensor
    tensor_y = torch.LongTensor(y)
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

    # shuffle samples at the beginning of each epoch
    torch.manual_seed(rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer__lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=optimizer__lr)

    idx = 0
    while idx < num_updates:
        for inputs, labels in dataloader:
            if idx >= num_updates:
                # stop the epoch in the middle if the number of updates has been reached
                break
            idx += 1

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


class ModelComp(tools.CompositeAlgo):

    def train(self, X, y, head_model, trunk_model, rank):
        torch.manual_seed(_SEED)  # initialize model weights
        torch.use_deterministic_algorithms(True)
        head_model = head_model or torch.nn.Module()
        trunk_model = trunk_model or Network()

        _fit(
            trunk_model,
            X,
            y,
            batch_size=_BATCH_SIZE,
            num_updates=_NUM_UPDATES,
            rank=rank,
        )

        return head_model, trunk_model

    def predict(self, X, head_model, trunk_model):

        X = torch.FloatTensor(X)
        trunk_model.eval()
        # add the context manager to reduce computation overhead
        with torch.no_grad():
            y_pred = trunk_model(X)

        y_pred = y_pred.data.cpu().numpy()
        return np.argmax(y_pred, axis=1)

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
    tools.algo.execute(ModelComp())
