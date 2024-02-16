import numpy as np
import matplotlib.pyplot as plt


def generate_benford_numbers(size):
    digits = np.arange(1, 10)
    probabilities = np.log10(1 + 1 / digits)
    probabilities /= probabilities.sum()
    benford_numbers = np.random.choice(digits, size=size, p=probabilities)
    return benford_numbers


def plot_average_histogram(iterations, size, bins):
    average_hist = np.zeros(len(bins) - 1)

    for _ in range(iterations):
        benford_numbers = generate_benford_numbers(size)
        hist, _ = np.histogram(benford_numbers, bins=bins, density=True)
        average_hist += hist

    average_hist /= iterations

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    plt.bar(bin_centers, average_hist, color='black', width=1, edgecolor='white')

    plt.title(f'Average Histogram of Numbers Following Benford\'s Law ({iterations} Iterations)')
    plt.xlabel('Digit')
    plt.ylabel('Average Probability')
    plt.xticks(np.arange(1, 10))
    plt.show()


iterations = 100
size = 1000
bins = np.arange(0.5, 10.5, 1)

plot_average_histogram(iterations, size, bins)
