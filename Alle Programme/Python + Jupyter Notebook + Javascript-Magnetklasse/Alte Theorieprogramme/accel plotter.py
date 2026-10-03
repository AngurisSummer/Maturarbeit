import matplotlib.pyplot as plt

# Data extracted from the new table
amt_magnets = [1, 2, 3, 4, 5, 6, 7, 8]

average_accel = [
    0.123915366,
    0.177557351,
    0.211726539,
    0.270992627,
    0.291358248,
    0.351838259,
    0.402998,
    0.3940776
]

st_dev = [
    0.088393961,
    0.065543115,
    0.133972652,
    0.140193251,
    0.096880147,
    0.111238021,
    0.038761895,
    0.017363004
]

theory_test_2 = [
    0.109,
    0.176,
    0.222,
    0.258,
    0.292,
    0.325,
    0.352,
    0.370
]

theory_err = [
    0.018596057,
    0.029971666,
    0.03780517,
    0.043850591,
    0.049793836,
    0.055396495,
    0.059858186,
    0.063076734
]

# Plotting
plt.figure(figsize=(10, 6))

# 2. Plot Experimental data SECOND
plt.errorbar(amt_magnets, theory_test_2, yerr=theory_err, fmt='o', label='Average Acceleration (Theory)', capsize=5, color='red', zorder=2)

# 2. Plot Experimental data SECOND
plt.errorbar(amt_magnets, average_accel, yerr=st_dev, fmt='o', label='Average Acceleration (Exp.)', capsize=5, color='blue', zorder=2)

# Adding labels and titles
plt.xlabel('Number of Magnets')
plt.ylabel('Acceleration m/s2')
plt.title('Acceleration vs. Number of Magnets (Theory Comparison)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()