#!/usr/bin/env python
import os
import shutil

import numpy as np
import substratools as tools


class NodeOpener(tools.Opener):
    def get_X(self, folders):
        """Load data."""
        assert len(folders) == 1, "Supporting only one data sample for the whole dataset"
        path = os.path.join(folders[0], "x.npy")
        return np.load(path)

    def get_y(self, folders):
        """Load labels."""
        assert len(folders) == 1, "Supporting only one data sample for the whole dataset"
        path = os.path.join(folders[0], "y.npy")
        return np.load(path)

    def save_predictions(self, y_pred, path):
        """Save predictions `y_pred` into a npy file."""
        np.save(path, y_pred)
        shutil.move(path + ".npy", path)

    def get_predictions(self, path):
        """Load predictions."""
        return np.load(path)

    def fake_X(self, n_samples=None):
        """Generate false data."""
        raise NotImplementedError

    def fake_y(self, n_samples=None):
        """Generate false labels."""
        raise NotImplementedError
