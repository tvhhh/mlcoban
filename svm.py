import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

class SVM:
  def __init__(self, X, y):
    self.X = X
    self.N = X.shape[0]   # num_points
    self.d = X.shape[1]   # num_features
    self.y = y
    self.y[np.where(y == 0)] = -1
    self.W = 0
    self.b = 0
  
  def fit(self):
    V = self.y.reshape(-1, 1) * self.X
    P = matrix(V.dot(V.T).astype(np.float))
    q = matrix(-np.ones((self.N, 1)))
    G = matrix(-np.eye(self.N))
    h = matrix(np.zeros((self.N, 1)))
    A = matrix(self.y.reshape(1, -1).astype(np.float))
    b = matrix(np.zeros((1, 1)))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    l = np.array(sol['x'])
    S = np.where(l > 1e-9)[0]
    self.W = np.sum((l[S, 0] * self.y[S]).reshape(-1, 1) * self.X[S], axis=0).T
    self.b = np.mean(self.y[S] - self.X[S].dot(self.W))
    return self.W, self.b
  
  def pred(self, X):
    res = X.dot(self.W) + self.b
    res[np.where(res >= 0)] = 1
    res[np.where(res < 0)] = 0
    return res
  
  def test(self, X_test, y_test):
    y_pred = self.pred(X_test)
    accuracy = np.sum(y_pred == y_test) / X_test.shape[0]
    print("Accuracy: {}".format(accuracy))
    return accuracy
