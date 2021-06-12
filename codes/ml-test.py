import numpy as np 
import matplotlib.pyplot as plt 

xs = np.array([1, 2, 4])
ys = np.array([2, 1, 2])

one = np.ones(xs.shape[0])
X = np.stack((one, xs)).T

W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), ys)
print(W)

x = np.linspace(1,4)
y_predict = W[0] + W[1]*x 
plt.scatter(xs, ys)
plt.plot(x, y_predict, c='red')
plt.show()