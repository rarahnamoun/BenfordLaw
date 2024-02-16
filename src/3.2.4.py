import numpy as np
from scipy.stats import chi2
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

def weibull_distribution_pdf(x, c):
    return (c / 2) * (x / 2) ** (c / 2 - 1) * np.exp(-(x / 2) ** (c / 2))

def weibull_distribution(size, c):
    return np.random.weibull(c, size) * 2  # Scale parameter to match the distribution

def get_most_significant_digits(numbers):
    return [int(str(abs(value))[0]) for value in numbers if value != 0]

def benford_law_distribution():
    digits = np.arange(1, 10)
    benford_probs = np.log10(1 + 1 / digits)
    benford_probs /= benford_probs.sum()
    return benford_probs

def fitness_function(params):
    weibull_numbers = weibull_distribution(size, params[0])
    weibull_digits = get_most_significant_digits(weibull_numbers)

    expected_benford_probs = benford_law_distribution()
    observed_probs = np.histogram(weibull_digits, bins=np.arange(0.5, 10.5, 1))[0] / len(weibull_digits)

    chi_squared_stat = np.sum((observed_probs - expected_benford_probs) ** 2 / expected_benford_probs)

    return chi_squared_stat

varbound = np.array([[1.0, 5.0]])

size = 1000

algorithm_param = {'max_num_iteration': 100, 'population_size': 10, 'mutation_probability': 0.1, 'elit_ratio': 0.01,
                   'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform', 'max_iteration_without_improv': None}
model = ga(function=fitness_function, dimension=1, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)

model.run()

optimized_params = model.output_dict['variable'][0]
print("Optimized Parameter (c):", optimized_params)

optimized_weibull_numbers = weibull_distribution(size, optimized_params)
optimized_weibull_digits = get_most_significant_digits(optimized_weibull_numbers)

print("\nOptimized Weibull Distribution Digit Counts:")
digit_counts = np.histogram(optimized_weibull_digits, bins=np.arange(0.5, 10.5, 1))[0]
digits = np.arange(1, 10)
for digit, count in zip(digits, digit_counts):
    print(f"Digit {digit}: {count}")

plt.figure(figsize=(12, 6))
x_values = np.linspace(0, max(optimized_weibull_numbers), 100)
pdf_values = weibull_distribution_pdf(x_values, optimized_params)

plt.subplot(1, 2, 1)
plt.plot(x_values, pdf_values, color='blue', label='Optimized Weibull PDF')
plt.title(f'Optimized Weibull PDF (c={optimized_params})')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()

plt.subplot(1, 2, 2)
digit_counts = np.histogram(optimized_weibull_digits, bins=np.arange(0.5, 10.5, 1))[0]
digit_probs = digit_counts / digit_counts.sum()

plt.bar(digits, digit_probs, color='purple', alpha=0.7, label='Optimized Weibull Distribution')
plt.plot(digits, benford_law_distribution(), color='red', marker='o', linestyle='dashed', label='Benford\'s Law')

plt.title('Optimized Weibull Distribution vs Benford\'s Law')
plt.xlabel('Digit')
plt.ylabel('Probability')
plt.xticks(digits)
plt.legend()

plt.show()
