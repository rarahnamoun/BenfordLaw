import numpy as np
import matplotlib.pyplot as plt
from math import factorial


def generate_factorial_numbers(size):
    return [factorial(i) for i in range(1, size + 1)]


def generate_powers_of_two(size):
    return [2 ** i for i in range(1, size + 1)]


def get_most_significant_digits(numbers):
    return [int(str(abs(value))[0]) for value in numbers]


def benford_law_distribution():
    digits = np.arange(1, 10)
    benford_probs = np.log10(1 + 1 / digits)
    benford_probs /= benford_probs.sum()
    return benford_probs


def print_digit_counts(data, label):
    digit_counts = np.histogram(data, bins=np.arange(0.5, 10.5, 1))[0]
    digits = np.arange(1, 10)

    print(f"{label} Digit Counts:")
    for digit, count in zip(digits, digit_counts):
        print(f"Digit {digit}: {count}")


def plot_distribution(data, label, color):
    digit_counts = np.histogram(data, bins=np.arange(0.5, 10.5, 1))[0]
    digit_probs = digit_counts / digit_counts.sum()

    digits = np.arange(1, 10)

    plt.bar(digits, digit_probs, color=color, alpha=0.7, label=label)
    plt.plot(digits, benford_law_distribution(), color='red', marker='o', linestyle='dashed', label='Benford\'s Law')

    plt.title(f'{label} Distribution vs Benford\'s Law')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.xticks(digits)
    plt.legend()


# Generate datasets
size = 1000
factorial_numbers = generate_factorial_numbers(size)
powers_of_two_numbers = generate_powers_of_two(size)

# Analyze the distribution of most significant digits
factorial_digits = get_most_significant_digits(factorial_numbers)
powers_of_two_digits = get_most_significant_digits(powers_of_two_numbers)

# Print digit counts for analysis
print_digit_counts(factorial_digits, 'Factorial Numbers')
print_digit_counts(powers_of_two_digits, 'Powers of Two Numbers')

# Plot both distributions in one picture
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plot_distribution(factorial_digits, 'Factorial Numbers', 'black')

plt.subplot(1, 2, 2)
plot_distribution(powers_of_two_digits, 'Powers of Two Numbers', 'black')

plt.show()
