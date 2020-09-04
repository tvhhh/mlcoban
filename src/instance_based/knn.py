import numpy as np

def pairwise_dist(Z, X):
  X2 = np.sum(X*X, axis=1)
  Z2 = np.sum(Z*Z, axis=1)
  return X2.reshape(1, -1) + Z2.reshape(-1, 1) - 2*Z.dot(X.T)

class KNN:
  def __init__(self, X, y, k=5, weights='uniform'):
    self.X = X
    self.N = X.shape[0]   # num_points
    self.d = X.shape[1]   # num_features
    self.y = y
    self.C = np.max(y)+1  # num_labels
    self.k = k
    self.weights = weights
    self.W = 0
  
  def pred(self, X_test):
    dist = pairwise_dist(X_test, self.X)
    k_nearest = np.argsort(dist, axis=1)[:, :self.k]
    dist = np.take_along_axis(dist, k_nearest, axis=1)
    label = self.y[k_nearest]
    y = np.zeros((X_test.shape[0], self.C))
    if self.weights == 'distance':
      self.W = 1 / np.sqrt(dist)
    elif self.weights == 'uniform':
      self.W = np.ones(dist.shape)
    elif self.weights == 'normal':
      self.W = np.exp(-dist)
    for i in range(self.k):
      label_k = label[:, i]
      W_k = self.W[:, i]
      y[np.arange(X_test.shape[0]), label_k] += W_k
    return np.argmax(y, axis=1)
  
  def test(self, X_test, y_test):
    y_pred = self.pred(X_test)
    accuracy = np.sum(y_pred == y_test) / X_test.shape[0]
    print("Accuracy: {}".format(accuracy))
    return accuracy
