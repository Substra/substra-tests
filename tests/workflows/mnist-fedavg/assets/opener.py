#!/usr/bin/env python
import os

import numpy as np
import substratools as tools


class OrganizationOpener(tools.Opener):
    def get_data(self, folders):
        """Load data."""
        assert len(folders) == 1, "Supporting only one data sample for the whole dataset"
        X_path = os.path.join(folders[0], "x.npy")
        y_path = os.path.join(folders[0], "y.npy")
        return np.load(X_path), np.load(y_path)

    def fake_data(self, n_samples=None):
        """Generate false data."""
        raise NotImplementedError
