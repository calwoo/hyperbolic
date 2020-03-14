import numpy as np 
import scipy.sparse

from multiprocessing import Pool
from multiprocessing.sharedctypes import Array
from ctypes import c_double

n = 10 # num features
m = 20000 # num train examples

X = scipy.sparse.random(m, n, density=0.2).toarray()
real_w = np.random.uniform(0, 1, size=(n, 1))

X = X / X.max()
y = np.dot(X, real_w)

coef_shared = Array(c_double,
    (np.random.normal(size=(n, 1)) * 1.0 / np.sqrt(n)).flat,
    lock=False)

w = np.frombuffer(coef_shared)
w = w.reshape((n, 1))

# gradient update - hogwild!
learning_rate = 0.001
def mse_gradient_step(X_y_tuple):
    global w
    X, y = X_y_tuple

    err = y.reshape((len(y), 1)) - np.dot(X, w)
    grad = - 2.0 * np.dot(np.transpose(X), err) / X.shape[0]

    for index in np.where(np.abs(grad) > 0.01)[0]:
        coef_shared[index] -= learning_rate * grad[index, 0]

batch_size = 1
examples = [None] * int(X.shape[0] / float(batch_size))

for k in range(int(X.shape[0] / float(batch_size))):
    Xx = X[k*batch_size : (k+1)*batch_size,:].reshape((batch_size,X.shape[1]))
    yy = y[k*batch_size : (k+1)*batch_size].reshape((batch_size,1))
    examples[k] = (Xx, yy) 

p = Pool(5)
p.map(mse_gradient_step, examples)

print("Loss function on the training set: ", np.mean(np.abs(y - np.dot(X, w))))
print("Difference from the real weight vector: ", np.abs(real_w - w).sum())