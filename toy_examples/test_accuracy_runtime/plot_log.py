import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("solve_times.csv")

plt.figure(figsize=(10, 6))
plt.plot(df["N"], df["DeDe_time"], label="DeDe Solve Time", color="blue")
plt.plot(df["N"], df["CVXPY_time"], label="CVXPY Solve Time", color="red")
plt.xlabel("Problem Size (N)")
plt.ylabel("Solve Time (seconds)")
plt.title("Solve Time vs Problem Size (Log Scale)")
plt.yscale("log") 
plt.xlim(left=0)
plt.ylim(bottom=0.001) 
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.savefig("time_log.png")
plt.show()

