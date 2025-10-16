#Name-Eklavya Chauhan
#Roll No.-2311067
#Date-17/11/2023
#Assignment-12

import math
from mylib import Simpsons
from mylib import GaussianQuadrature

# --------------------------
# Problem 1
# --------------------------
f1 = lambda x: (x**2) / (1 + x**4) # Function to integrate
a1, b1 = -1, 1 # Integration limits
true_val1 = 0.487495494 # Known true value for comparison

results1 = {} # Store results for different N
for n in range(2, 17):
    try:
        GQ1 = GaussianQuadrature(f1, a1, b1, n)
        results1[n] = GQ1.integrate()
    except ValueError:
        continue

# --------------------------
# Problem 2
# --------------------------
f2 = lambda x: math.sqrt(1 + x**4)
a2, b2 = 0, 1
true_val2 = 1.089429413

# Simpson’s rule
Simpson_obj = Simpsons(f2, a2, b2, N=20, outfile="temp.txt") # Using N=20 for Simpson's rule
simpson_val = Simpson_obj.integrate() # Perform integration

results2 = {} # Store results for different N
for n in range(2, 17):
    try:
        GQ2 = GaussianQuadrature(f2, a2, b2, n)
        results2[n] = GQ2.integrate()
    except ValueError:
        continue

# --------------------------
# Write summary and comparison
# --------------------------
with open("Integration_results.txt", "w") as f:
    f.write("===== SUMMARY OF RESULTS =====\n\n")

    # Problem 1
    f.write("Problem 1: ∫[-1,1] x² / (1 + x⁴) dx\n")
    f.write(f"True value ≈ {true_val1}\n")
    for n, val in results1.items():
        err = abs(val - true_val1)
        f.write(f"  N = {n:<2}  =>  Result = {val:.9f},  Error = {err:.2e}\n")
    f.write("\n")

    # Problem 2
    f.write("Problem 2: ∫[0,1] √(1 + x⁴) dx\n")
    f.write(f"True value ≈ {true_val2}\n")
    f.write(f"Simpson's Rule (N=20): {simpson_val:.9f}, Error = {abs(simpson_val - true_val2):.2e}\n\n")

    f.write("Gaussian Quadrature results:\n")
    f.write(" N  |    Result    |    Error\n")
    f.write("-"*36 + "\n")
    for n, val in results2.items():
        err = abs(val - true_val2)
        f.write(f"{n:<3} | {format(val, '.9f'):>12} | {format(err, '.2e'):>12}\n")
    f.write("\n")

    # Comparison Table
    f.write("===== COMPARISON TABLE =====\n")
    f.write(f"{'Method':<25}{'Result':<20}{'Error'}\n")
    f.write("-"*60 + "\n")
    f.write(f"{'Simpson (N=20)':<25}{format(simpson_val, '.9f'):<20}{format(abs(simpson_val - true_val2), '.2e')}\n")
    for n, val in results2.items():
        err = abs(val - true_val2)
        f.write(f"{('Gaussian N='+str(n)):<25}{format(val, '.9f'):<20}{format(err, '.2e')}\n")

    # Accuracy Summary
    acc1 = next((n for n, v in results1.items() if abs(v - true_val1) < 1e-9), None)
    acc2 = next((n for n, v in results2.items() if abs(v - true_val2) < 1e-9), None)
    acc_simpson = abs(simpson_val - true_val2) < 1e-9

    f.write("\n===== ACCURACY NOTE =====\n")
    f.write(f"For Integral 1, 9-decimal accuracy reached at N = {acc1}\n" if acc1 else
            "For Integral 1, 9-decimal accuracy not reached within N ≤ 16\n")
    f.write(f"For Integral 2 (Gaussian), 9-decimal accuracy reached at N = {acc2}\n" if acc2 else
            "For Integral 2 (Gaussian), 9-decimal accuracy not reached within N ≤ 16\n")
    f.write(f"For Integral 2 (Simpson, N=20), 9-decimal accuracy reached? {'Yes' if acc_simpson else 'No'}\n")

print("Results written to 'Integration_results.txt'")
