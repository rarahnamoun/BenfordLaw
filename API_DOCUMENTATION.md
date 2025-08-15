# Benford Distribution Analysis - API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Modules](#core-modules)
   - [Main Module (`main.py`)](#main-module-mainpy)
   - [Statistical Analysis Modules](#statistical-analysis-modules)
   - [Benford's Law Analysis Modules](#benfords-law-analysis-modules)
   - [Network Analysis Module](#network-analysis-module)
   - [Genetic Algorithm Optimization Modules](#genetic-algorithm-optimization-modules)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)

## Overview

This project explores the application of genetic algorithms (GAs) to determine optimal parameters for fitting the Benford distribution to various probability distributions (Weibull, lognormal, and Pareto). It also assesses the J-Divergence of complex network models concerning their conformity to Benford's Law and investigates the probability density function (PDF) of Benford's Law.

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- numpy
- matplotlib
- scipy
- pandas
- seaborn
- networkx
- geneticalgorithm

## Core Modules

### Main Module (`main.py`)

#### Functions

##### `generate_benford_numbers(size)`

Generates random numbers following Benford's Law distribution.

**Parameters:**
- `size` (int): Number of random numbers to generate

**Returns:**
- `numpy.ndarray`: Array of integers (1-9) following Benford's Law distribution

**Example:**
```python
import numpy as np
from main import generate_benford_numbers

# Generate 1000 numbers following Benford's Law
benford_nums = generate_benford_numbers(1000)
print(f"Generated {len(benford_nums)} numbers")
print(f"First 10 numbers: {benford_nums[:10]}")
```

##### `plot_average_histogram(iterations, size, bins)`

Creates and displays an average histogram of numbers following Benford's Law over multiple iterations.

**Parameters:**
- `iterations` (int): Number of iterations to average over
- `size` (int): Number of samples per iteration
- `bins` (numpy.ndarray): Bin edges for histogram

**Returns:**
- None (displays plot)

**Example:**
```python
import numpy as np
from main import plot_average_histogram

# Create average histogram over 100 iterations
iterations = 100
size = 1000
bins = np.arange(0.5, 10.5, 1)
plot_average_histogram(iterations, size, bins)
```

### Statistical Analysis Modules

#### Module `1.1.py` - Hypergeometric vs Binomial Distribution Comparison

**Purpose:** Compares hypergeometric and binomial distributions for different sample sizes.

**Key Features:**
- Visualizes probability mass functions
- Compares distributions for multiple sample sizes
- Uses scipy.stats for statistical calculations

**Usage:**
```python
# Run the comparison analysis
python src/1.1.py
```

#### Module `2.py` - Text Analysis and Zipf's Law

**Purpose:** Analyzes word frequency in text files and visualizes Zipf's Law.

**Key Features:**
- Word frequency counting from text files
- Zipf's Law visualization
- Log-log plotting for power law analysis

**Dependencies:**
- Requires `Text-2.txt` file in the src directory

**Usage:**
```python
# Run text analysis
python src/2.py
```

#### Module `2.1.py` - Zipf's Law Distribution Analysis

**Purpose:** Demonstrates Zipf's Law distribution with predefined data.

**Key Features:**
- Zipf distribution parameter fitting
- Log-log plot visualization
- Statistical distribution analysis

**Usage:**
```python
# Run Zipf's Law analysis
python src/2.1.py
```

### Benford's Law Analysis Modules

#### Module `3.1.py` - Benford's vs Zipf's Law Comparison

##### Functions

##### `generate_benford_numbers(size)`

Generates random numbers following Benford's Law distribution.

**Parameters:**
- `size` (int): Number of random numbers to generate

**Returns:**
- `numpy.ndarray`: Array of integers following Benford's distribution

##### `fit_benford_zipf_laws(data)`

Fits both Benford's and Zipf's laws to observed data.

**Parameters:**
- `data` (array-like): Input data to fit

**Returns:**
- `tuple`: (observed_counts, benford_probs, zipf_probs)
  - `observed_counts` (numpy.ndarray): Histogram counts of observed data
  - `benford_probs` (numpy.ndarray): Benford's Law probabilities
  - `zipf_probs` (numpy.ndarray): Zipf's Law probabilities

##### `plot_laws_fit(observed_counts, benford_probs, zipf_probs)`

Visualizes the fit of Benford's and Zipf's laws to observed data.

**Parameters:**
- `observed_counts` (numpy.ndarray): Observed frequency counts
- `benford_probs` (numpy.ndarray): Benford's Law probabilities
- `zipf_probs` (numpy.ndarray): Zipf's Law probabilities

**Returns:**
- None (displays plot)

**Example:**
```python
from src.py31 import generate_benford_numbers, fit_benford_zipf_laws, plot_laws_fit

# Generate and analyze data
data = generate_benford_numbers(1000)
observed, benford, zipf = fit_benford_zipf_laws(data)
plot_laws_fit(observed, benford, zipf)
```

#### Module `3.2.py` - Physical Constants Analysis

##### Functions

##### `is_numeric_constant(value)`

Checks if a value is a numeric constant.

**Parameters:**
- `value` (any): Value to check

**Returns:**
- `bool`: True if value is numeric, False otherwise

##### `get_most_significant_digits(constant_values)`

Extracts the most significant digits from a list of numeric values.

**Parameters:**
- `constant_values` (list): List of numeric values

**Returns:**
- `list`: List of most significant digits (1-9)

##### `benford_law_distribution()`

Calculates the theoretical Benford's Law probability distribution.

**Returns:**
- `numpy.ndarray`: Normalized Benford's Law probabilities for digits 1-9

##### `print_digit_counts(data)`

Prints the count of each digit (1-9) in the dataset.

**Parameters:**
- `data` (array-like): Input data containing digits

**Returns:**
- None (prints to console)

##### `plot_distribution(data, label)`

Plots the distribution of most significant digits compared to Benford's Law.

**Parameters:**
- `data` (array-like): Input digit data
- `label` (str): Label for the plot legend

**Returns:**
- None (displays plot)

**Example:**
```python
from scipy import constants
from src.py32 import get_most_significant_digits, plot_distribution

# Analyze physical constants
all_constants = [getattr(constants, constant) for constant in dir(constants) 
                if is_numeric_constant(getattr(constants, constant))]
digits = get_most_significant_digits(all_constants)
plot_distribution(digits, 'Physical Constants')
```

#### Module `3.3.py` - Generalized Benford's Law Heatmap

##### Functions

##### `benford_law_digit_probability(n, d, sigma=1.0)`

Calculates the probability of digit `d` appearing in position `n` according to generalized Benford's Law.

**Parameters:**
- `n` (int): Position in the number (1-based)
- `d` (int): Digit (0-9)
- `sigma` (float, optional): Scaling parameter (default: 1.0)

**Returns:**
- `float`: Probability percentage for the digit in the specified position

**Example:**
```python
from src.py33 import benford_law_digit_probability

# Calculate probability of digit 1 in first position
prob = benford_law_digit_probability(1, 1)
print(f"Probability of digit 1 in position 1: {prob:.2f}%")
```

**Key Features:**
- Generates heatmap visualization of digit probabilities
- Supports multiple positions and digits
- Uses seaborn for advanced visualization

#### Module `3.4.py` - Mean and Variance Analysis

##### Functions

##### `benford_law_digit_probability(n, d, sigma=1.0)`

Calculates digit probability for generalized Benford's Law (same as 3.3.py).

##### `calculate_mean_variance(n)`

Calculates the mean and variance for digit probabilities at position `n`.

**Parameters:**
- `n` (int): Position in the number

**Returns:**
- `tuple`: (mean, variance) for the specified position

**Example:**
```python
from src.py34 import calculate_mean_variance

# Calculate statistics for first position
mean, variance = calculate_mean_variance(1)
print(f"Position 1 - Mean: {mean:.3f}, Variance: {variance:.3f}")
```

### Network Analysis Module

#### Module `3.5.py` - Complex Network Analysis

##### Functions

##### `benford_pmf(x)`

Calculates the Benford's Law probability mass function.

**Parameters:**
- `x` (numeric): Input value

**Returns:**
- `float`: Benford probability for the given value

##### `plot_degree_and_benford_histogram(graph, graph_type)`

Plots degree distribution histogram compared to Benford's Law.

**Parameters:**
- `graph` (networkx.Graph): Input network graph
- `graph_type` (str): Type of graph for labeling

**Returns:**
- `numpy.ndarray`: Differences between actual and Benford distributions

##### `calculate_and_plot(graph_type, num_nodes, num_edges, seed=None)`

Generates and analyzes different types of complex networks.

**Parameters:**
- `graph_type` (str): Type of graph ('scale-free', 'random', 'small-world')
- `num_nodes` (int): Number of nodes in the graph
- `num_edges` (int): Number of edges (or degree parameter)
- `seed` (int, optional): Random seed for reproducibility

**Returns:**
- None (displays plot and prints statistics)

**Example:**
```python
from src.py35 import calculate_and_plot

# Analyze different network types
calculate_and_plot('scale-free', 100, 2)
calculate_and_plot('random', 100, 2)
calculate_and_plot('small-world', 100, 2)
```

**Supported Network Types:**
- **Scale-free networks**: Generated using Barabási-Albert model
- **Random networks**: Generated using Erdős-Rényi model
- **Small-world networks**: Generated using Watts-Strogatz model

### Genetic Algorithm Optimization Modules

#### Module `3.2.2.py` - Factorial and Powers Analysis

##### Functions

##### `generate_factorial_numbers(size)`

Generates factorial numbers up to a specified size.

**Parameters:**
- `size` (int): Number of factorial values to generate

**Returns:**
- `list`: List of factorial numbers [1!, 2!, ..., size!]

##### `generate_powers_of_two(size)`

Generates powers of two up to a specified size.

**Parameters:**
- `size` (int): Number of powers to generate

**Returns:**
- `list`: List of powers of two [2^1, 2^2, ..., 2^size]

##### `get_most_significant_digits(numbers)`

Extracts most significant digits from a list of numbers.

**Parameters:**
- `numbers` (list): List of numeric values

**Returns:**
- `list`: List of most significant digits

**Example:**
```python
from src.py322 import generate_factorial_numbers, get_most_significant_digits

# Analyze factorial numbers
factorials = generate_factorial_numbers(20)
digits = get_most_significant_digits(factorials)
print(f"Most significant digits of factorials: {digits}")
```

#### Module `3.2.3.py` - Pareto Distribution Optimization

##### Functions

##### `pareto_distribution_pdf(x, a)`

Calculates the probability density function of Pareto distribution.

**Parameters:**
- `x` (numeric): Input value
- `a` (float): Shape parameter

**Returns:**
- `float`: PDF value

##### `pareto_distribution(size, a)`

Generates random samples from Pareto distribution.

**Parameters:**
- `size` (int): Number of samples
- `a` (float): Shape parameter

**Returns:**
- `numpy.ndarray`: Random samples from Pareto distribution

##### `fitness_function(params)`

Fitness function for genetic algorithm optimization.

**Parameters:**
- `params` (array-like): Parameters to optimize

**Returns:**
- `float`: Chi-squared statistic (lower is better)

**Example:**
```python
from geneticalgorithm import geneticalgorithm as ga
from src.py323 import fitness_function

# Set up genetic algorithm
algorithm_param = {
    'max_num_iteration': 100,
    'population_size': 50,
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}

varbound = np.array([[0.1, 5.0]])  # Parameter bounds
model = ga(function=fitness_function, dimension=1, 
           variable_type='real', variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)
model.run()
```

#### Module `3.2.4.py` - Weibull Distribution Optimization

Similar structure to Pareto optimization but for Weibull distribution parameters.

##### Key Functions:
- `weibull_distribution_pdf(x, a, b)`: Weibull PDF calculation
- `weibull_distribution(size, a, b)`: Random sample generation
- `fitness_function(params)`: GA fitness function for Weibull parameters

#### Module `3.2.5.py` - Lognormal Distribution Optimization

Similar structure for lognormal distribution optimization.

##### Key Functions:
- `lognormal_distribution_pdf(x, mu, sigma)`: Lognormal PDF calculation
- `lognormal_distribution(size, mu, sigma)`: Random sample generation
- `fitness_function(params)`: GA fitness function for lognormal parameters

#### Module `3.2.5.2.py` - Enhanced Lognormal Analysis

Extended version of lognormal analysis with additional features and visualization.

## Usage Examples

### Basic Benford's Law Analysis

```python
import numpy as np
from src.main import generate_benford_numbers, plot_average_histogram

# Generate Benford numbers and visualize
numbers = generate_benford_numbers(1000)
bins = np.arange(0.5, 10.5, 1)
plot_average_histogram(100, 1000, bins)
```

### Network Analysis Workflow

```python
from src.py35 import calculate_and_plot

# Analyze different network topologies
networks = ['scale-free', 'random', 'small-world']
for network_type in networks:
    calculate_and_plot(network_type, 100, 2, seed=42)
```

### Genetic Algorithm Optimization

```python
from geneticalgorithm import geneticalgorithm as ga
import numpy as np

# Example for Pareto distribution optimization
def setup_ga_optimization(fitness_func, param_bounds):
    algorithm_param = {
        'max_num_iteration': 100,
        'population_size': 50,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform'
    }
    
    model = ga(function=fitness_func, 
               dimension=len(param_bounds),
               variable_type='real', 
               variable_boundaries=param_bounds,
               algorithm_parameters=algorithm_param)
    
    return model

# Usage
from src.py323 import fitness_function
bounds = np.array([[0.1, 5.0]])  # Pareto shape parameter bounds
model = setup_ga_optimization(fitness_function, bounds)
model.run()
optimal_params = model.output_dict['variable']
```

## Best Practices

### Performance Optimization
1. **Use vectorized operations**: All modules use NumPy vectorization for better performance
2. **Set random seeds**: Use `np.random.seed()` for reproducible results
3. **Batch processing**: Process multiple iterations efficiently using the averaging functions

### Data Analysis
1. **Validate input data**: Ensure numeric data is properly formatted
2. **Handle edge cases**: Check for zero or negative values in logarithmic calculations
3. **Statistical significance**: Use appropriate sample sizes for meaningful results

### Visualization
1. **Consistent styling**: All plots use consistent color schemes and formatting
2. **Clear labeling**: Include descriptive titles, axis labels, and legends
3. **Interactive analysis**: Combine multiple visualizations for comprehensive analysis

### Genetic Algorithm Tuning
1. **Parameter bounds**: Set appropriate bounds for optimization parameters
2. **Population size**: Use adequate population sizes for complex optimization problems
3. **Convergence criteria**: Monitor convergence and adjust iteration limits accordingly

## Error Handling

Common issues and solutions:

1. **Import errors**: Ensure all required packages are installed
2. **File not found**: Check that `Text-2.txt` exists in the src directory for text analysis
3. **Memory issues**: Reduce sample sizes for large-scale analysis
4. **Convergence problems**: Adjust GA parameters or increase iteration limits