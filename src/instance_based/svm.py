import os
import sys
path = os.path.join(os.getcwd(), '../')
sys.path.append(path)
import numpy as np
from cvxopt import matrix, solvers
from super import Super

class SVM:
  def __init__(self, X, y):
    self.X = X
    self.N = X.shape[0]   # num_points
    self.d = X.shape[1]   # num_features
    self.y = y
    self.y[np.where(y == 0)] = -1
    self.W = np.random.randn(self.d)
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
    print("Accuracy: {:.2f}%".format(accuracy*100))
    return accuracy


class SoftMarginSVM(Super):
  def __init__(self, X, y, l=0.01):
    super().__init__(X, y, l)
    self.y[np.where(y == 0)] = -1
  
  def loss(self):
    z = 1 - self.y * (self.X.dot(self.W))
    return np.mean(np.maximum(0, z)) + self.l/2*self.W.T.dot(self.W)
  
  def grad(self, X, y):
    h = 1 - y * (X.dot(self.W))
    loss_set = np.where(h >= 0)
    grad_W = np.sum(-y[loss_set].reshape(-1, 1) * X[loss_set], axis=0).T/X.shape[0] + self.l*self.W
    return grad_W

  def pred(self, X):
    res = X.dot(self.W)
    res[np.where(res >= 0)] = 1
    res[np.where(res < 0)] = 0
    return res
