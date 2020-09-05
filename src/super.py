import numpy as np
import matplotlib.pyplot as plt

class Super:
  def __init__(self, X, y, l=0):
    self.X = X.copy()
    self.X = np.concatenate((self.X, np.ones((X.shape[0], 1))), axis=1)
    self.N = self.X.shape[0]  # num_points
    self.d = self.X.shape[1]  # num_features
    self.y_label = y
    self.C = np.max(y) + 1    # num_labels
    if self.C == 2:
      self.y = y.copy()
      self.W = np.random.randn(self.d)
    else:
      self.y = np.zeros((self.N, self.C))
      self.y[np.arange(self.N), self.y_label] = 1
      self.W = np.random.randn(self.d, self.C)
    self.l = l
  
  def prob(self, X):
    raise NotImplementedError()
  
  def loss(self):
    raise NotImplementedError()

  def grad(self, X, y):
    raise NotImplementedError()

  def fit(self, lr=0.01, nepoches=50, batch_size=10, print_every=0, plot=False):
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
      if print_every > 0 and ep % print_every == 0:
        print("Epoch {}/{}\tLoss: {}".format(ep, nepoches, loss))
      if np.linalg.norm(self.W - W_old)/self.W.size < 1e-6:
        break
      W_old = self.W.copy()
    if plot:
      eps = range(len(loss_hist))
      plt.figure(figsize=(10, 8))
      plt.plot(eps, loss_hist)
      plt.title("Loss history")
      plt.xlabel("Epoches")
      plt.ylabel("Loss")
      plt.show()
    return self.W, loss_hist
  
  def pred(self, X):
    raise NotImplementedError()

  def test(self, X_test, y_test):
    X = X_test.copy()
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    y_pred = self.pred(X)
    accuracy = np.sum(y_pred == y_test) / X_test.shape[0]
    print("Accuracy: {:.2f}%".format(accuracy*100))
    return accuracy
