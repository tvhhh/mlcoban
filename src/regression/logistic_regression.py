import os
import sys
path = os.path.join(os.getcwd(), '../')
sys.path.append(path)
import numpy as np
from super import Super

def sigmoid(z):
  s = np.array(z, dtype=np.float)
  return 1 / (1 + np.exp(-s))

class LogisticRegression(Super):
  def prob(self, X):
    return sigmoid(X.dot(self.W))
  
  def loss(self):
    a = self.prob(self.X)
    return -np.mean(self.y*np.log(a) + (1-self.y)*np.log(1-a) - self.l/2 * self.W.T.dot(self.W))
  
  def grad(self, X, y):
    a = self.prob(X)
    return X.T.dot(a - y) / X.shape[0] + self.l*self.W
  
  def pred(self, X):
    res = self.prob(X)
    res[np.where(res >= 0.5)] = 1
    res[np.where(res < 0.5)] = 0
    return res
