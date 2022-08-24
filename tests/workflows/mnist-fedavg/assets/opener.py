#!/usr/bin/env python
import os

import numpy as np
import substratools as tools


class OrganizationOpener(tools.Opener):
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

    def fake_X(self, n_samples=None):
        """Generate false data."""
        raise NotImplementedError

    def fake_y(self, n_samples=None):
        """Generate false labels."""
        raise NotImplementedError
