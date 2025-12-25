#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

x = []
time_dede = []
time_cvxpy = []

with open("timing_unweighted_sum.txt", "r") as f:
    for line in f:
        parts = line.split()
        n = float(parts[0])
        t1 = float(parts[1])
        t2 = float(parts[2])

        x.append(n)
        time_dede.append(t1/3600)
        time_cvxpy.append(t2/3600)

x = np.array(x)
time_dede = np.array(time_dede)
time_cvxpy = np.array(time_cvxpy)

TAIL_START = 1500 
mask = x >= TAIL_START

x_tail = x[mask]
dede_tail = time_dede[mask]
cvxpy_tail = time_cvxpy[mask]

def power_law_fit(n, t):
    log_n = np.log(n)
    log_t = np.log(t)
    alpha, log_C = np.polyfit(log_n, log_t, 1)
    C = np.exp(log_C)
    return C, alpha

C_dede, alpha_dede = power_law_fit(x_tail, dede_tail)
C_cvxpy, alpha_cvxpy = power_law_fit(x_tail, cvxpy_tail)

# Fitted curves
x_fit = np.linspace(x.min(), x.max(), 500)
fit_dede = C_dede * x_fit**alpha_dede
fit_cvxpy = C_cvxpy * x_fit**alpha_cvxpy

plt.figure()
plt.plot(x, time_dede, "o", label="DeDe (10 cores)")
plt.plot(x, time_cvxpy, "o", label="CVXPY")

plt.plot(x_fit, fit_dede, "-", label=f"DeDe fit: n^{alpha_dede:.2f}")
plt.plot(x_fit, fit_cvxpy, "-", label=f"CVXPY fit: n^{alpha_cvxpy:.2f}")

plt.xlabel("n")
plt.ylabel("Time (hours)")
plt.title("Time to solve nxn maximal sum")
plt.legend()
plt.savefig("timing_unweighted_sum.png", dpi=300, bbox_inches="tight")
