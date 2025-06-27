import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

df = pd.read_csv("solve_times.csv")

N = df["N"].values
dede_time = df["DeDe_time"].values
cvxpy_time = df["CVXPY_time"].values

mask = N % 500 == 0

N_sampled = N[mask]
dede_sampled = dede_time[mask]
cvxpy_sampled = cvxpy_time[mask]

log_N = np.log(N_sampled)
log_dede = np.log(dede_sampled)
log_cvxpy = np.log(cvxpy_sampled)

dede_fit = linregress(log_N, log_dede)
cvxpy_fit = linregress(log_N, log_cvxpy)

k_dede = dede_fit.slope
a_dede = np.exp(dede_fit.intercept)

k_cvxpy = cvxpy_fit.slope
a_cvxpy = np.exp(cvxpy_fit.intercept)

print(f"DeDe fit: time ≈ {a_dede:.3e} * N^{k_dede:.3f}")
print(f"CVXPY fit: time ≈ {a_cvxpy:.3e} * N^{k_cvxpy:.3f}")

fitted_dede = a_dede * N**k_dede
fitted_cvxpy = a_cvxpy * N**k_cvxpy

# plot solve time and fitted curve for both DeDe and CVXPY
plt.figure(figsize=(10, 6))
plt.plot(N, dede_time, label="DeDe Solve Time (actual)", color="blue")
plt.plot(N, cvxpy_time, label="CVXPY Solve Time (actual)", color="red")
plt.plot(N, fitted_dede, '--', color="blue", label=f"DeDe Fit: O(N^{k_dede:.2f})")
plt.plot(N, fitted_cvxpy, '--', color="red", label=f"CVXPY Fit: O(N^{k_cvxpy:.2f})")

plt.xlabel("Problem Size (N)")
plt.ylabel("Solve Time (seconds)")
plt.title("Solve Time vs Problem Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("time_regular.png")
plt.show()

# plot relative error

df["relative_error"] = np.abs(df["DeDe_result"] - df["CVXPY_result"]) / np.abs(df["CVXPY_result"])

plt.figure(figsize=(10, 6))
plt.plot(df["N"], df["relative_error"], label="Relative Error", color="purple")
plt.xlabel("Problem Size (N)")
plt.ylabel("Relative Error")
plt.title("Relative Error of Objective Value: DeDe vs CVXPY")
plt.yscale("log")
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.savefig("rel_err_regular.png")
plt.show()
