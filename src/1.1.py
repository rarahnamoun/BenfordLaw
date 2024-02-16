import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hypergeom, binom

N = 100
M = 20
n_values = [10, 50]

plt.figure(figsize=(10, 6))

for n in n_values:
    k_values = np.arange(0, 20)
    hypergeo_pmf = hypergeom.pmf(k_values, N, M, n)
    binom_pmf = binom.pmf(k_values, n, M/N)

    plt.bar(k_values - 0.2 + (n_values.index(n) * 0.4), hypergeo_pmf, width=0.4, label=f'Hypergeometric (n={n})', alpha=0.7)
    plt.bar(k_values + 0.2 + (n_values.index(n) * 0.4), binom_pmf, width=0.4, label=f'Binomial (n={n})', alpha=0.7)

plt.xlabel('Number of Successes (k)')
plt.ylabel('Probability (P(k))')
plt.title('Comparison of Hypergeometric and Binomial Distributions for Different n values')
plt.legend()

plt.show()
