import numpy as np

def pairwise_dist(Z, X):
  X2 = np.sum(X*X, axis=1)
  Z2 = np.sum(Z*Z, axis=1)
  return X2.reshape(1, -1) + Z2.reshape(-1, 1) - 2*Z.dot(X.T)

class KNN:
  def __init__(self, X, y):
    self.X = X
    self.N = X.shape[0]     # num_points
    self.d = X.shape[1]     # num_features
    self.y = y
    self.C = np.max(y) + 1  # num_labels
  
  def get_weights(self, dist, weights):
    if weights == 'distance':
      W = 1 / np.sqrt(dist)
    elif weights == 'uniform':
      W = np.ones(dist.shape)
    elif weights == 'normal':
      W = np.exp(-dist)
    else:
      W = np.random.randn(dist.shape)
    return W
  
  def pred(self, X_test, k, weights):
    dist = pairwise_dist(X_test, self.X)
    k_nearest = np.argsort(dist, axis=1)[:, :k]
    dist = np.take_along_axis(dist, k_nearest, axis=1)
    label = self.y[k_nearest]
    y = np.zeros((X_test.shape[0], self.C))
    W = self.get_weights(dist, weights)
    for i in range(k):
      label_k = label[:, i]
      W_k = W[:, i]
      y[np.arange(X_test.shape[0]), label_k] += W_k
    return np.argmax(y, axis=1)
  
  def test(self, X_test, y_test, k=5, weights='uniform'):
    y_pred = self.pred(X_test, k, weights)
    accuracy = np.sum(y_pred == y_test) / X_test.shape[0]
    print("Accuracy: {:.2f}%".format(accuracy*100))
    return accuracy
