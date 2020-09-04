import numpy as np
import matplotlib.pyplot as plt

class MulticlassSVM:
  def __init__(self, X, y, l=0):
    self.X = X.copy()
    self.X = np.concatenate((self.X, np.ones((X.shape[0], 1))), axis=1)
    self.N = self.X.shape[0]  # num_points
    self.d = self.X.shape[1]  # num_features
    self.y = y
    self.C = np.max(y)+1      # num_labels
    self.W = 0
    self.l = l
  
  def prob(self, X):
    return X.dot(self.W)
  
  def loss(self):
    Z = self.prob(self.X)
    Zy = Z[np.arange(self.N), self.y].reshape(-1, 1)
    h = np.maximum(0, Z - Zy + 1)
    h[np.arange(self.N), self.y] = 0
    return np.sum(h)/self.N + self.l/2*np.sum(self.W*self.W)
  
  def grad(self, X, y):
    Z = self.prob(X)
    Zy = Z[np.arange(X.shape[0]), y].reshape(-1, 1)
    h = np.maximum(0, Z - Zy + 1)
    h[np.arange(X.shape[0]), y] = 0
    F = (h > 0).astype(np.int)
    F[np.arange(X.shape[0]), y] = -np.sum(F, axis=1)
    return X.T.dot(F)/X.shape[0] + self.l*self.W
  
  def fit(self, lr=0.01, nepoches=50, batch_size=10, hist=False):
    self.W = np.random.randn(self.d, self.C)
    W_old = self.W.copy()
    ep = 0
    nbatches = int(np.ceil(float(self.N) / batch_size))
    loss_hist = [self.loss()]
    while ep < nepoches:
      ep += 1
      mix_ids = np.random.permutation(self.N)
      for i in range(nbatches):
        batch_ids = mix_ids[i*batch_size : min((i+1)*batch_size, self.N)]
        X_batch, y_batch = self.X[batch_ids], self.y[batch_ids]
        self.W -= lr*self.grad(X_batch, y_batch)
      loss = self.loss()
      loss_hist.append(loss)
      if hist:
        print("Epoch: {}\tLoss: {}".format(ep, loss))
      if np.linalg.norm(self.W - W_old) == 0:
        break
      W_old = self.W.copy()
    return self.W, loss_hist
  
  def plot(self, loss_hist):
    eps = range(len(loss_hist))
    plt.figure(figsize=(10, 8))
    plt.plot(eps, loss_hist)
    plt.title("Multiclass SVM loss")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.show()
  
  def pred(self, X):
    Z = self.prob(X)
    return np.argmax(Z, axis=1)
  
  def test(self, X_test, y_test):
    X = X_test.copy()
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    y_pred = self.pred(X)
    accuracy = np.sum(y_pred == y_test) / X_test.shape[0]
    print("Accuracy: {}".format(accuracy))
    return accuracy
