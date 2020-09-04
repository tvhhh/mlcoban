import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
  e_z = np.exp(z - np.max(z, axis=1).reshape(-1, 1))
  return e_z / np.sum(e_z, axis=1).reshape(-1, 1)

class SoftmaxRegression:
  def __init__(self, X, y):
    self.X = X
    self.N = X.shape[0]   # num_points
    self.d = X.shape[1]   # num_features
    self.y_label = y
    self.C = np.max(y)+1  # num_labels
    self.y = np.zeros((self.N, self.C))
    self.y[np.arange(self.N), self.y_label] = 1
    self.W = 0
  
  def prob(self, X):
    return softmax(X.dot(self.W))
  
  def loss(self):
    A = self.prob(self.X)
    ids = range(self.N)
    return -np.mean(np.log(A[ids, self.y_label]))
  
  def grad(self, X, y):
    A = self.prob(X)
    E = A - y
    return X.T.dot(E) / X.shape[0]
  
  def fit(self, lr=0.01, nepoches=50, batch_size=10, bp=1e-5, hist=False):
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
      if np.linalg.norm(self.W - W_old)/self.W.size < bp:
        break
      W_old = self.W.copy()
    return self.W, loss_hist
  
  def plot(self, loss_hist):
    eps = range(len(loss_hist))
    plt.figure(figsize=(10, 8))
    plt.plot(eps, loss_hist)
    plt.title("Softmax Regression loss")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.show()
  
  def pred(self, X):
    return np.argmax(X.dot(self.W), axis=1)
  
  def test(self, X_test, y_test):
    y_pred = self.pred(X_test)
    accuracy = np.sum(y_pred == y_test) / X_test.shape[0]
    print("Accuracy: {}".format(accuracy))
    return accuracy
