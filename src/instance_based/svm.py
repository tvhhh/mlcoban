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


class SoftMarginSVM:
  def __init__(self, X, y, l=0.1):
    self.X = X
    self.N = X.shape[0]   # num_points
    self.d = X.shape[1]   # num_features
    self.y = y
    self.y[np.where(y == 0)] = -1
    self.W = 0
    self.b = 0
    self.l = l
  
  def loss(self):
    z = 1 - self.y * (self.X.dot(self.W) + self.b)
    return np.mean(np.maximum(0, z)) + self.l/2*self.W.T.dot(self.W)
  
  def grad(self, X, y):
    h = 1 - y * (X.dot(self.W) + self.b)
    loss_set = np.where(h >= 0)
    grad_W = np.sum(-y[loss_set].reshape(-1, 1) * X[loss_set], axis=0).T/X.shape[0] + self.l*self.W
    grad_b = np.sum(-y[loss_set])/X.shape[0]
    return grad_W, grad_b
  
  def fit(self, lr=0.01, nepoches=50, batch_size=10, hist=False):
    self.W = np.random.randn(self.d)
    self.b = np.random.randn()
    W_old = self.W.copy()
    b_old = self.b
    ep = 0
    nbatches = int(np.ceil(float(self.N) / batch_size))
    loss_hist = [self.loss()]
    while ep < nepoches:
      ep += 1
      mix_ids = np.random.permutation(self.N)
      for i in range(nbatches):
        batch_ids = mix_ids[i*batch_size : min((i+1)*batch_size, self.N)]
        X_batch, y_batch = self.X[batch_ids], self.y[batch_ids]
        grad_W, grad_b = self.grad(X_batch, y_batch)
        self.W = self.W - lr*grad_W
        self.b -= lr*grad_b
      loss = self.loss()
      loss_hist.append(loss)
      if hist:
        print("Epoch: {}\tLoss: {}".format(ep, loss))
      if np.linalg.norm(self.W - W_old) == 0 and self.b - b_old == 0:
        break
      W_old = self.W.copy()
      b_old = self.b
    return self.W, self.b, loss_hist

  def plot(self, loss_hist):
    eps = range(len(loss_hist))
    plt.figure(figsize=(10, 8))
    plt.plot(eps, loss_hist)
    plt.title("Soft-margin SVM loss")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.show()

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
