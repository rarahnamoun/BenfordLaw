import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def benford_law_digit_probability(n, d, sigma=1.0):
    probability = sigma * np.sum(np.log10(1 + 1 / (10 * k + d)) if (10 * k + d) != 0 else 0 for k in range(int(10 ** (n-2)), int(10 ** (n-1))))
    return probability * 100  # Convert to percentage

n_values = range(1, 6)
d_values = range(10)
sigma_value = 1.0

table_data = []

for n in n_values:
    row = [benford_law_digit_probability(n, d, sigma=sigma_value) for d in d_values]
    table_data.append(row)

table_df = pd.DataFrame(table_data, columns=[f'Digit {d}' for d in d_values], index=[f'Position {n}' for n in n_values])

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(table_df, cmap='viridis', annot=True, fmt='.2f', linewidths=.5, cbar_kws={'label': 'Probability (%)'})
plt.title('Generalized Benford\'s Law: Digit Probabilities in Positions')
plt.show()
