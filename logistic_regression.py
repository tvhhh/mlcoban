import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
  s = np.array(z, dtype=np.float)
  return 1 / (1 + np.exp(-s))

class LogisticRegression:
  def __init__(self, X, y, l=0):
    self.X = X
    self.N = X.shape[0]   # num_points
    self.d = X.shape[1]   # num_features
    self.y = y
    self.W = 0
    self.l = l
  
  def prob(self, X):
    return sigmoid(X.dot(self.W))
  
  def loss(self):
    a = self.prob(self.X)
    return -np.mean(self.y*np.log(a) + (1-self.y)*np.log(1-a) - self.l/2 * self.W.T.dot(self.W))
  
  def grad(self, X, y):
    a = self.prob(X)
    return X.T.dot(a - y) / X.shape[0] + self.l*self.W
  
  def fit(self, lr=0.01, nepoches=50, batch_size=10, bp=1e-5, hist=False):
    self.W = np.random.randn(self.d)
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
        self.W = self.W - lr*self.grad(X_batch, y_batch)
      loss = self.loss()
      loss_hist.append(loss)
      if hist:
        print("Epoch: {}\tLoss: {}".format(ep, loss))
      if np.linalg.norm(self.W - W_old)/self.W.size < bp:
        break
      W_old = self.W.copy()
    return self.W, loss_hist
  
  def plot(self, loss_hist):
    eps = range(len(loss_hist))
    plt.figure(figsize=(10, 8))
    plt.plot(eps, loss_hist)
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.show()
  
  def pred(self, X, threshold=0.5):
    res = self.prob(X)
    res[np.where(res >= threshold)] = 1
    res[np.where(res < 1)] = 0
    return res
  
  def test(self, X_test, y_test, threshold=0.5):
    y_pred = self.pred(X_test, threshold)
    accuracy = np.sum(y_pred == y_test) / X_test.shape[0]
    print("Accuracy: {}".format(accuracy))
    return accuracy
