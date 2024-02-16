import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

def log_normal_distribution_pdf(x, sigma):
    return 1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) ** 2) / (2 * sigma ** 2))

def log_normal_distribution(size, sigma):
    return np.random.lognormal(sigma=sigma, size=size)

def get_most_significant_digits(numbers):
    return [int(str(abs(value))[0]) for value in numbers if value != 0]

def benford_law_distribution():
    digits = np.arange(1, 10)
    benford_probs = np.log10(1 + 1 / digits)
    benford_probs /= benford_probs.sum()
    return benford_probs

def fitness_function(params):
    log_normal_numbers = log_normal_distribution(size, np.exp(params[0]))
    log_normal_digits = get_most_significant_digits(log_normal_numbers)

    expected_benford_probs = benford_law_distribution()
    observed_probs = np.histogram(log_normal_digits, bins=np.arange(0.5, 10.5, 1))[0] / len(log_normal_digits)

    chi_squared_stat = np.sum((observed_probs - expected_benford_probs) ** 2 / expected_benford_probs)

    return chi_squared_stat

varbound = np.array([[np.log(0.1), np.log(1.0)]])
size = 1000
num_iterations = 1
optimized_params_list = []

algorithm_param = {'max_num_iteration': 100, 'population_size': 10, 'mutation_probability': 0.1, 'elit_ratio': 0.01,
                   'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform', 'max_iteration_without_improv': None}

x_values = np.linspace(0.1, 1.0, 100)
pdf_values_benford = benford_law_distribution()

for _ in range(num_iterations):
    model = ga(function=fitness_function, dimension=1, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
    model.run()
    optimized_params = model.output_dict['variable'][0]
    optimized_params_list.append(np.exp(optimized_params))

    log_normal_numbers = log_normal_distribution(size, np.exp(optimized_params))
    log_normal_digits = get_most_significant_digits(log_normal_numbers)
    observed_probs = np.histogram(log_normal_digits, bins=np.arange(0.5, 10.5, 1))[0] / len(log_normal_digits)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x_values, log_normal_distribution_pdf(x_values, np.exp(optimized_params)), color='blue', label='Optimized Log-Normal PDF')
    plt.title(f'Optimized Log-Normal PDF (sigma={np.exp(optimized_params)})')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(range(1, 10), observed_probs, color='purple', alpha=0.7, label='Observed Log-Normal Distribution')
    plt.plot(range(1, 10), pdf_values_benford, color='red', marker='o', linestyle='dashed', label='Benford\'s Law')

    plt.title('Observed Log-Normal Distribution vs Benford\'s Law')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.xticks(range(1, 10))
    plt.legend()

    plt.show()

average_optimized_param = np.mean(optimized_params_list)
print(f"\nAverage Optimized Log-Normal Variance (sigma) over {num_iterations} iteration: {average_optimized_param}")
