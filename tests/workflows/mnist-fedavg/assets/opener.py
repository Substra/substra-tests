#!/usr/bin/env python
import csv
import os

import numpy as np
import substratools as tools


class NodeOpener(tools.Opener):
    def get_X(self, folders):
        """Load data."""
        # Find npy files
        paths = []
        for folder in folders:
            if isinstance(folder, bytes):
                folder = folder.decode()
            if isinstance(folder, np.str):
                folder = str(folder)
            paths += [
                os.path.join(folder, f) for f in os.listdir(folder) if f[-4:] == ".npy"
            ]

        # Load data into a dictionnary
        X_list = []
        for path in paths:
            X_list.append(np.float32(np.load(path)))

        X_res = np.stack(X_list, axis=0)
        return X_res

    def get_y(self, folders):
        """Load labels."""
        # Find csv files
        paths = []
        for folder in folders:
            if isinstance(folder, bytes):
                folder = folder.decode()
            if isinstance(folder, np.str):
                folder = str(folder)
            paths += [
                os.path.join(folder, f) for f in os.listdir(folder) if f[-4:] == ".csv"
            ]

        # Load labels into a dictionnary
        y_list = []
        for path in paths:
            with open(path, "r") as csv_file:
                reader = csv.DictReader(csv_file, delimiter=";")
                for row in reader:
                    y_list.append(row["label"])

        return np.array(y_list)[:].astype("int64")

    def save_predictions(self, y_pred, path):
        """Save predictions `y_pred` into a csv file."""
        np.savetxt(path, y_pred)

    def get_predictions(self, path):
        """Load predictions saved into a csv file."""
        return np.genfromtxt(path, delimiter=',')

    def fake_X(self, n_samples=None):
        """Generate false data."""
        raise NotImplementedError

    def fake_y(self, n_samples=None):
        """Generate false labels."""
        raise NotImplementedError
