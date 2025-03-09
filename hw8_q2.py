import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import odeint
import numpy as np
from numpy import linalg as LA
from scipy import signal
from scipy.linalg import fractional_matrix_power
from scipy import sparse
import cvxpy as cp
import scipy as sp

np.random.seed(0)
n = 5  # dimension
m = 100  # number of distance samples
X = np.random.normal(size=(n, m))
Y = np.random.normal(size=(n, m))
P = np.random.normal(size=(n, n))
A_true = P @ P.T + np.eye(n)
sqrtA_true = sp.linalg.sqrtm(A_true)
d_true = np.linalg.norm(sqrtA_true @ (X - Y), axis=0)
# exact distances
d = np.maximum(np.zeros(m), d_true + 0.2 * np.random.normal(size=(m,)))
# add noise and make nonnegative

A = cp.Variable((n, n))
cc = cp.Variable(1, nonneg=True)
f0 = 0
for i in range(m):
    x_i = X[:, i]
    y_i = Y[:, i]
    d_i = d[i]
    z_i = x_i - y_i
    f0 = f0 + z_i.T @ A @ z_i + d_i ** 2 - 2 * d_i * cp.sqrt(z_i.T @ A @ z_i)
constraints = []
problem = cp.Problem(cp.Minimize(f0), constraints)
problem.solve(solver=cp.CLARABEL)
opt_val = problem.value
A_val = A.value
ss=2