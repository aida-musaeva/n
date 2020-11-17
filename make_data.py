import numpy as np
import random as r
import math
import sympy as sp
import scipy.optimize
from scipy import optimize
from numpy import *
import time
import pandas as pd
import csv



start_time = time.time()
np.set_printoptions(precision=2)


def make_sample(N):
    a = np.array([-0.288, 0.628, 0.000141, 3.83])
    c = -28.305
    a0 = -0.000656
    a = np.array([-0.288, 0.628, 0.000141, 3.83])

    X = np.random.randint(0, 2, N)
    print(X)
    if X.any() == 0:
        X[r.randint(0, N - 1)] = 1
    print(X)
    average_profit = r.randint(10000, 90000)
    dist = r.randint(400, 12000)
    k = [[r.uniform(dist / 800, 20), r.randint(0, 1), average_profit, math.log(dist, math.e)] for j in range(N)]
    cost = np.array([2.6 * 53038 * r.uniform(dist / 800, min(k[it][0], dist / 800 * 3 / 2)) / 149 for it in range(N)])
    print(k)
    print(cost)
    print('SS')
    return X, k, cost


def f(x, a0=-0.000656, c=-28.305):
    global X
    N = X.shape[0]
    cost = np.array([4100 for i in range(X.shape[0])])
    a = np.array([-0.288, 0.628, 0.000141, 3.83])  # Вектор параметров
    ak = [np.dot(a, k[i]) for i in range(N)]
    N = X.shape[0]
    f = zeros([N])
    s = 1
    for i in range(N):
        if X[i] == 1:
            s += exp(a0 * x[i] + ak[i] + c)
    for i in range(N):
        f[i] = (1 - exp(a0 * x[i] + ak[i] + c) / s) * (cost[i] - x[i]) - 1 / a0
        if X[i] == 0:
            f[i] = 0
    return f


def M(price, a0=-0.000656, c=-28.305):
    global X
    a = np.array([-0.288, 0.628, 0.000141, 3.83])  # Вектор параметров
    ak = [np.dot(a, k[i]) for i in range(X.shape[0])]
    N = X.shape[0]
    m = np.zeros([N])
    s = 1
    for i in range(N):
        if X[i] == 1:
            s += exp(a0 * price[i] + ak[i] + c)
    for i in range(N):
        m[i] = exp(a0 * price[i] + ak[i] + c) / s
        if X[i] == 0:
            m[i] = 0
    return m


N = 5
k_data = []
w_data = []
for it1 in range(100):
    X, k, cost = make_sample(N)
    x0 = np.zeros([N])
    for it2 in range(N):
        x0[it2] = (cost[it2] + 1000) * X[it2]

    sol = optimize.root(f, x0, method='krylov')
    P_opt = sol.x
    m = M(P_opt)
    w = (P_opt - cost) * m
    print(P_opt)  # Цены
    print(m)  # Доли
    print(w)
    for i in range(N):
        if X[i] == 0:
            k[i] = [0 for it in range(4)]

    with open("k_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(k)
    with open("w_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([w])

k_data = pd.read_csv('k_data.csv', sep=';', engine='python')
w_data = pd.read_csv('w_data.csv', sep=';', engine='python')
print(k_data)
print(w_data)
print("~~~~~ %s seconds ~~~~~" % (time.time() - start_time))