import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zipf


def generate_benford_numbers(size):
    digits = np.arange(1, 10)
    probabilities = np.log10(1 + 1 / digits)
    probabilities /= probabilities.sum()
    benford_numbers = np.random.choice(digits, size=size, p=probabilities)
    return benford_numbers


def fit_benford_zipf_laws(data):
    observed_counts = np.histogram(data, bins=np.arange(0.5, 10.5, 1))[0]

    benford_probs = np.log10(1 + 1 / np.arange(1, 10))
    benford_probs /= benford_probs.sum()

    zipf_alpha = 1.0  # You may adjust the alpha parameter for Zipf's Law
    zipf_probs = zipf.pmf(np.arange(1, 10), zipf_alpha)
    zipf_probs /= zipf_probs.sum()

    return observed_counts, benford_probs, zipf_probs


def plot_laws_fit(observed_counts, benford_probs, zipf_probs):
    digits = np.arange(1, 10)

    plt.bar(digits - 0.2, observed_counts / observed_counts.sum(), width=0.4, color='black', label='Zipf\'s Law')
    plt.bar(digits + 0.2, benford_probs, width=0.4, color='blue', alpha=0.7, label='Benford\'s Law')
   

    plt.title('Fitting Benford\'s and Zipf\'s Laws to Observed Distribution')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.xticks(digits)
    plt.legend()
    plt.show()


# Generate Benford's Law numbers
size = 1000
benford_numbers = generate_benford_numbers(size)

# Fit Benford's and Zipf's Laws
observed_counts, benford_probs, zipf_probs = fit_benford_zipf_laws(benford_numbers)

# Plot the fit
plot_laws_fit(observed_counts, benford_probs, zipf_probs)
