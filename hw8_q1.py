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

np.random.seed(10)

K = 5  # number of classes
n = 20  # feature dimension
mTrain = 200  # number of training samples
mTest = 100  # number of test samples
# Generate synthetic data
A_true = np.random.normal(size=(K, n))
b_true = np.random.normal(size=(K, 1))
v = 0.2 * np.random.normal(size=(K, mTrain + mTest))  # noise
data = np.random.normal(size=(n, mTrain + mTest))
label = np.argmax(A_true @ data + np.tile(b_true, (1, mTrain + mTest)) +
                  v, axis=0)
# Training data
x = data[:, :mTrain]
y = label[:mTrain]
# Test data
xtest = data[:, mTrain:]
ytest = label[mTrain:]
# Define a range of mu values (regularization parameters) on a log-scale
mus = np.logspace(-1, 2, 10)
N = np.size(mus)
errorTrain = []
errorTest = []


def cvx_opt(
        mu_i: np.ndarray
) -> list:
    ## cp variables
    A = cp.Variable((K, n))
    b = cp.Variable(K)

    # Objective
    f0 = 1 * mu_i * cp.square(cp.norm(A, 'fro'))

    for i in range(mTrain):
        x_i = x[:, i]
        F_i = A @ x_i + b  # construct F
        y_i = y[i]
        f_yi = F_i[y_i]
        F_removed = cp.hstack([F_i[j] for j in range(K) if j != y_i])
        f_k = cp.max(F_removed)
        f0 = f0 + cp.pos(1 + f_k - f_yi)

    # Constraint
    constraints = [cp.sum(b) == 0]

    problem = cp.Problem(cp.Minimize(f0), constraints)
    problem.solve(solver=cp.CLARABEL)
    A_val = A.value
    b_val = b.value
    opt_val = problem.value
    print(problem.value)

    return A_val, b_val, opt_val


A_list = np.zeros((N, K, n))
b_list = np.zeros((N, K))
for i in range(N):
    mu_i = mus[i]
    [A_val, b_val, opt_val] = cvx_opt(mu_i)
    A_list[i, :, :] = A_val
    b_list[i, :] = b_val

## Do the validation check
y_hat_train = np.zeros((N, mTrain))
error_rate = np.zeros(N)
error_list = np.zeros((N, mTrain))
for i in range(N):
    error_i = np.zeros(mTrain)
    for j in range(mTrain):
        F = A_list[i, :, :] @ x[:, j] + b_list[i, :]
        y_hat_train[i, j] = round(np.max(F))
        if y_hat_train[i, j] >= 4:
            y_hat_train[i, j] = 4
        if y_hat_train[i, j] == y[j]:
            error_i[j] = 1
    error_list[i, :] = error_i

ss = 5
