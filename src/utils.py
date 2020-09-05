import pandas as pd
import numpy as np

def train_test_split(X, y, train_ratio=0.8):
  size = X.shape[0]
  mix_ids = np.random.permutation(size)
  train_size = int(size * train_ratio)
  train_ids, test_ids = mix_ids[:train_size], mix_ids[train_size:]
  X_train, y_train = X[train_ids], y[train_ids]
  X_test, y_test = X[test_ids], y[test_ids]
  return X_train, y_train, X_test, y_test

def normalize(X):
  X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
  X = (X - X_min) / (X_max - X_min)
  X[np.isnan(X)] = 0
  return X
