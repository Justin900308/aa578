import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.linalg as la
import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
from scipy import sparse
import cvxpy as cp

# n = 20
# np.random.seed(0)
#
# ## for part a,b
# # random variables
# p = np.random.rand(n)
# a = np.random.rand(n)
# d = np.random.rand(n)
# c = np.random.rand(n, n)
#
# # optimization variables
# x = cp.Variable(n)
# y = cp.Variable((n, n))
#
# # cost function
# f0_a = p @ x
# f0_b = p @ x - cp.trace(np.transpose(c) @ y)
#
# # constraints for part a
# constraints_a = []
# constraints_a.append(cp.sum(x) == 0)
# for i in range(n):
#     constraints_a.append(-a[i] <= x[i])
#     constraints_a.append(x[i] <= d[i])
#     # for j in range(n):
#     #     constraints_a.append(x[i] == (cp.sum(y[i, :]) - cp.sum(y[:, i])))
#     #     constraints_a.append(y[i, j] >= 0)
#
# # constraints for part b
# constraints_b = []
# constraints_b.append(cp.sum(x) == 0)
# for i in range(n):
#     constraints_b.append(-a[i] <= x[i])
#     constraints_b.append(x[i] <= d[i])
#     for j in range(n):
#         constraints_b.append(x[i] == (cp.sum(y[i, :]) - cp.sum(y[:, i])))
#         constraints_b.append(y[i, j] >= 0)
#
# # solve part a
# problem_a = cp.Problem(cp.Maximize(f0_a), constraints_a)
# problem_a.solve(solver=cp.CLARABEL)
# obj_val_a = problem_a.value
#
# # solve part b
# problem_b = cp.Problem(cp.Maximize(f0_b), constraints_b)
# problem_b.solve(solver=cp.CLARABEL)
# obj_val_b = problem_b.value
#
# print(obj_val_a, obj_val_b)
#
# ## part c
#
# # random variables
# c = np.random.rand(n, n)
#
# # cp variables
# p = cp.Variable(n)
# ub = cp.Variable(1)
# lb = cp.Variable(1)
# f0_c = ub - lb
# constraints_c = []
# for i in range(n):
#     constraints_c.append(ub <= p[i])
#     constraints_c.append(lb >= p[i])
#     for j in range(n):
#         constraints_c.append((p[j] - p[i]) <= c[i, j])
# # solve part b
# problem_c = cp.Problem(cp.Maximize(f0_c), constraints_c)
# problem_c.solve(solver=cp.CLARABEL)
# ub_val = ub.value
# lb_val = lb.value
# obj_val_c = problem_c.value
#
# print(obj_val_c)

## p4
n = 2
m = 100
r = 1
N = 21
np.random.seed(3)
x = np.random.randn(n, m)

x[:, 39] = 10 * np.random.randn(n)
x[:, 64] = 10 * np.random.randn(n)
x[:, 32] = 10 * np.random.randn(n)

r_list = np.linspace(0, 0.8 * m, N)


def cvx_opt(
        r_i: np.ndarray
) -> list:
    A = cp.Variable((n, n), PSD=True)
    sigma = cp.Variable((n, n), PSD=True)
    b = cp.Variable(n)
    constraints_4 = [A >> 0]
    f0_4 = - cp.log_det(A)
    # f0_4 = f0_4 + 0*cp.maximum(0, cp.norm(A @ x[:, 39] + b, 2) - 1)
    for i in range(m):
        f0_4 = f0_4 + r_i * cp.maximum(0, cp.norm(A @ x[:, i] + b, 2) - 1) / m
    ## First exclude outliers
    for i in range(m):
        if i == 39:
            continue
        elif i == 64:
            continue
        elif i == 32:
            continue
        else:
            constraints_4.append(cp.norm(A @ x[:, i] + b, 2) <= 1)

    problem_4 = cp.Problem(cp.Minimize(f0_4), constraints_4)
    problem_4.solve(solver=cp.CLARABEL)
    opt_val = problem_4.value
    A_val = A.value
    b_val = b.value

    return [opt_val, A_val, b_val]


A_total = np.zeros((N, n, n))
b_total = np.zeros((N, n))
opt_val_list = np.zeros(N)
vol_list = np.zeros(N)
dist_list = np.zeros(N)
for i in range(N):
    [opt_val_i, A_val_i, b_val_i] = cvx_opt(r_list[i])

    opt_val_list[i] = opt_val_i
    A_total[i, :, :] = A_val_i
    b_total[i, :] = b_val_i
    vol_list[i] = np.pi * LA.det(LA.inv(A_val_i))
    dist_i = 0
    for j in range(m):
        # print(LA.norm(A_val_i @ x[:, j] + b_val_i, 2) - 1)
        dist_i = dist_i + np.maximum(0, LA.norm(A_val_i @ x[:, j] + b_val_i, 2) - 1)
    dist_list[i] = dist_i
    print('optimal value:   ', opt_val_list[i], 'det(A_inv):    ', vol_list[i], 'dist_sum:  ', dist_list[i])
plt.subplot(2, 1, 1)
plt.plot(r_list, vol_list)
plt.xlabel('r(penalty for distance)')
plt.ylabel('volume')
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(dist_list, vol_list)
plt.xlabel('total distance')
plt.ylabel('volume')
plt.grid()
plt.show()

## plot some selected cases
theta = np.linspace(0, 2 * np.pi, 301)
for j in range(m):
    plt.plot(x[0, j], x[1, j], 'b.')
for i in range(N):
    A_i = A_total[i, :, :]
    b_i = b_total[i, :]
    if i % 2 == 0:
        c = - LA.inv(A_i) @ b_i
        u = np.zeros((2, 301))
        for j in range(301):
            u[:, j] = c + LA.inv(A_i) @ np.array([np.sin(theta[j]), np.cos(theta[j])])
            plt.plot(u[0, j], u[1, j], 'g.', markersize=2)
        plt.plot(c[0], c[1], 'r.')
print(x[:, 39], x[:, 64], x[:, 32])
plt.show()
print(x[:, 39], x[:, 64], x[:, 32])
