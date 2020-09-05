import os
import sys
path = os.path.join(os.getcwd(), '../')
sys.path.append(path)
import numpy as np
from super import Super

def softmax(z):
  e_z = np.exp(z - np.max(z, axis=1).reshape(-1, 1))
  return e_z / np.sum(e_z, axis=1).reshape(-1, 1)

class SoftmaxRegression(Super):  
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
  
  def pred(self, X):
    return np.argmax(X.dot(self.W), axis=1)
