import os
import sys
path = os.path.join(os.getcwd(), '../')
sys.path.append(path)
import numpy as np
from super import Super

class MulticlassSVM(Super):
  def __init__(self, X, y, l=0.01):
    super().__init__(X, y, l)
    self.y = y

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
  
  def pred(self, X):
    Z = self.prob(X)
    return np.argmax(Z, axis=1)
