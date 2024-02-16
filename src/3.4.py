import numpy as np
import matplotlib.pyplot as plt

def benford_law_digit_probability(n, d, sigma=1.0):
    probability = np.sum(np.log10(1 + 1 / (10 * k + d)) if (10 * k + d) != 0 else 0 for k in range(int(10 ** (n-2)), int(10 ** (n-1))))
    return probability * 100  # Convert to percentage

def calculate_mean_variance(n):
    mean_values = []
    variance_values = []
    for d in range(10):
        probability = benford_law_digit_probability(n, d)
        mean_values.append(d * probability)
        variance_values.append(d**2 * probability)

    mean_n = np.sum(mean_values) / 100
    variance_n = np.sum(variance_values) / 100 - mean_n**2

    if n > 1:
        mean_n_minus_1, variance_n_minus_1 = calculate_mean_variance(n-1)
        mean_avg = 0.5 * (mean_n + mean_n_minus_1)
        variance_avg = 0.5 * (variance_n + variance_n_minus_1)
        return mean_avg, variance_avg

    return mean_n, variance_n

table_data = []
for n in range(1, 9):
    mean, variance = calculate_mean_variance(n)
    table_data.append([f'n={n}', variance, mean])

fig, ax = plt.subplots()

data = np.array([[entry[1], entry[2]] for entry in table_data], dtype=float)
im = ax.imshow(data, cmap='coolwarm')

ax.set_xticks(np.arange(2))
ax.set_yticks(np.arange(len(table_data)))
ax.set_xticklabels(['Variance', 'Mean'])
ax.set_yticklabels([entry[0] for entry in table_data])

# Loop over data dimensions and create text annotations.
for i in range(len(table_data)):
    for j in range(2):
        text = ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)

cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('Value')

plt.show()
