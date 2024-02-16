import numpy as np
import matplotlib.pyplot as plt
from scipy import constants


def is_numeric_constant(value):
    return isinstance(value, (int, float, np.integer, np.floating))


def get_most_significant_digits(constant_values):
    return [int(str(abs(value))[0]) for value in constant_values if is_numeric_constant(value)]


def benford_law_distribution():
    digits = np.arange(1, 10)
    benford_probs = np.log10(1 + 1 / digits)
    benford_probs /= benford_probs.sum()
    return benford_probs


def print_digit_counts(data):
    digit_counts = np.histogram(data, bins=np.arange(0.5, 10.5, 1))[0]
    digits = np.arange(1, 10)

    print("Digit Counts:")
    for digit, count in zip(digits, digit_counts):
        print(f"Digit {digit}: {count}")


def plot_distribution(data, label):
    digit_counts = np.histogram(data, bins=np.arange(0.5, 10.5, 1))[0]
    digit_probs = digit_counts / digit_counts.sum()

    digits = np.arange(1, 10)

    plt.bar(digits, digit_probs, color='blue', alpha=0.7, label=label)
    plt.plot(digits, benford_law_distribution(), color='red', marker='o', linestyle='dashed', label='Benford\'s Law')

    plt.title('Most Significant Digits Distribution vs Benford\'s Law')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.xticks(digits)
    plt.legend()
    plt.show()


all_constants = [getattr(constants, constant) for constant in dir(constants) if
                 is_numeric_constant(getattr(constants, constant))]
larger_dataset = np.tile(all_constants, 1000)
all_digits = get_most_significant_digits(larger_dataset)
print_digit_counts(all_digits)
plot_distribution(all_digits, 'Most Significant Digits of All Numeric Constants')
