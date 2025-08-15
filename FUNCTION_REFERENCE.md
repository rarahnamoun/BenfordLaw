# Function Reference Guide

## Table of Contents

1. [Core Functions](#core-functions)
2. [Distribution Analysis Functions](#distribution-analysis-functions)
3. [Benford's Law Functions](#benfords-law-functions)
4. [Network Analysis Functions](#network-analysis-functions)
5. [Genetic Algorithm Functions](#genetic-algorithm-functions)
6. [Utility Functions](#utility-functions)
7. [Visualization Functions](#visualization-functions)

## Core Functions

### `main.py`

#### `generate_benford_numbers(size)`

Generates random numbers following Benford's Law distribution.

**Signature:**
```python
def generate_benford_numbers(size: int) -> numpy.ndarray
```

**Parameters:**
- `size` (int): Number of random numbers to generate

**Returns:**
- `numpy.ndarray`: Array of integers (1-9) following Benford's Law distribution

**Algorithm:**
1. Creates array of digits 1-9
2. Calculates Benford probabilities: log₁₀(1 + 1/d) for digit d
3. Normalizes probabilities to sum to 1
4. Uses numpy.random.choice with calculated probabilities

**Example:**
```python
numbers = generate_benford_numbers(1000)
# Returns array like [1, 2, 1, 3, 1, 4, ...]
```

**Mathematical Foundation:**
Benford's Law states that the probability of digit d appearing as the first digit is:
P(d) = log₁₀(1 + 1/d)

---

#### `plot_average_histogram(iterations, size, bins)`

Creates and displays an average histogram of numbers following Benford's Law over multiple iterations.

**Signature:**
```python
def plot_average_histogram(iterations: int, size: int, bins: numpy.ndarray) -> None
```

**Parameters:**
- `iterations` (int): Number of iterations to average over
- `size` (int): Number of samples per iteration
- `bins` (numpy.ndarray): Bin edges for histogram

**Returns:**
- None (displays plot using matplotlib)

**Algorithm:**
1. Initialize average histogram array
2. For each iteration:
   - Generate Benford numbers
   - Create histogram
   - Add to running average
3. Divide by number of iterations
4. Create bar plot with proper formatting

**Example:**
```python
bins = np.arange(0.5, 10.5, 1)
plot_average_histogram(100, 1000, bins)
```

---

## Distribution Analysis Functions

### `1.1.py` - Hypergeometric and Binomial Comparison

This module demonstrates comparison between hypergeometric and binomial distributions.

**Key Parameters:**
- `N = 100`: Total population size
- `M = 20`: Number of success items in population
- `n_values = [10, 50]`: Sample sizes to compare

**Functions:**
- Uses `scipy.stats.hypergeom.pmf()` for hypergeometric PMF
- Uses `scipy.stats.binom.pmf()` for binomial PMF
- Creates side-by-side bar plots for comparison

---

### `2.py` - Text Analysis and Word Frequency

#### Text Processing Function

**Description:** Analyzes word frequency in text files and demonstrates Zipf's Law.

**Algorithm:**
1. Opens and reads text file (`Text-2.txt`)
2. Splits text into words
3. Counts word frequencies using dictionary
4. Creates rank-frequency data
5. Plots both linear and log-log visualizations

**Data Structure:**
```python
wordcount = {}  # Dictionary storing word: frequency pairs
df = pd.DataFrame({'x': ranks, 'y': frequencies})
```

---

### `2.1.py` - Zipf's Law Distribution

#### Zipf Distribution Fitting

**Parameters:**
- `C`: Normalization constant (set to maximum count)
- `s = 2.0`: Zipf exponent parameter

**Mathematical Formula:**
```
P(rank) = C / rank^s
```

**Visualization:**
- Log-log plot showing power law relationship
- Red line represents theoretical Zipf distribution

---

## Benford's Law Functions

### `3.1.py` - Benford vs Zipf Comparison

#### `generate_benford_numbers(size)`

**Signature:**
```python
def generate_benford_numbers(size: int) -> numpy.ndarray
```
Same as main.py implementation.

---

#### `fit_benford_zipf_laws(data)`

Fits both Benford's and Zipf's laws to observed data.

**Signature:**
```python
def fit_benford_zipf_laws(data: array_like) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
```

**Parameters:**
- `data` (array-like): Input data to fit

**Returns:**
- `tuple`: (observed_counts, benford_probs, zipf_probs)
  - `observed_counts` (numpy.ndarray): Histogram counts of observed data
  - `benford_probs` (numpy.ndarray): Benford's Law probabilities
  - `zipf_probs` (numpy.ndarray): Zipf's Law probabilities

**Algorithm:**
1. Create histogram of input data
2. Calculate Benford probabilities
3. Calculate Zipf probabilities with α=1.0
4. Normalize all probability distributions

---

#### `plot_laws_fit(observed_counts, benford_probs, zipf_probs)`

Visualizes the fit of laws to observed data.

**Signature:**
```python
def plot_laws_fit(observed_counts: numpy.ndarray, benford_probs: numpy.ndarray, zipf_probs: numpy.ndarray) -> None
```

**Visualization Features:**
- Side-by-side bar plots
- Different colors for each distribution
- Proper axis labels and legend
- Digit-wise comparison (1-9)

---

### `3.2.py` - Physical Constants Analysis

#### `is_numeric_constant(value)`

Type checking function for numeric constants.

**Signature:**
```python
def is_numeric_constant(value: any) -> bool
```

**Parameters:**
- `value` (any): Value to check

**Returns:**
- `bool`: True if value is numeric (int, float, np.integer, np.floating)

**Implementation:**
```python
return isinstance(value, (int, float, np.integer, np.floating))
```

---

#### `get_most_significant_digits(constant_values)`

Extracts first digits from numeric values.

**Signature:**
```python
def get_most_significant_digits(constant_values: list) -> list[int]
```

**Parameters:**
- `constant_values` (list): List of numeric values

**Returns:**
- `list[int]`: List of most significant digits (1-9)

**Algorithm:**
1. Filter numeric constants
2. Convert to string representation
3. Extract first character
4. Convert back to integer

**Edge Cases:**
- Handles negative values using abs()
- Filters out non-numeric values
- Handles scientific notation

---

#### `benford_law_distribution()`

Calculates theoretical Benford's Law distribution.

**Signature:**
```python
def benford_law_distribution() -> numpy.ndarray
```

**Returns:**
- `numpy.ndarray`: Normalized probabilities for digits 1-9

**Formula:**
```python
benford_probs = np.log10(1 + 1 / digits)
```

---

#### `print_digit_counts(data)`

Console output function for digit frequency analysis.

**Signature:**
```python
def print_digit_counts(data: array_like) -> None
```

**Output Format:**
```
Digit Counts:
Digit 1: 301
Digit 2: 176
...
```

---

#### `plot_distribution(data, label)`

Visualization function comparing observed vs theoretical distributions.

**Signature:**
```python
def plot_distribution(data: array_like, label: str) -> None
```

**Features:**
- Blue bars for observed data
- Red line with markers for Benford's Law
- Comprehensive labels and legend

---

### `3.3.py` - Generalized Benford's Law

#### `benford_law_digit_probability(n, d, sigma=1.0)`

Calculates generalized Benford's Law probabilities.

**Signature:**
```python
def benford_law_digit_probability(n: int, d: int, sigma: float = 1.0) -> float
```

**Parameters:**
- `n` (int): Position in the number (1-based indexing)
- `d` (int): Digit (0-9)
- `sigma` (float): Scaling parameter (default: 1.0)

**Returns:**
- `float`: Probability percentage for digit d in position n

**Mathematical Formula:**
```
P(n,d) = σ × Σ[k=10^(n-2) to 10^(n-1)-1] log₁₀(1 + 1/(10k + d))
```

**Implementation Details:**
- Handles edge case where denominator is zero
- Returns probability as percentage (× 100)
- Supports multiple digit positions

**Example:**
```python
prob = benford_law_digit_probability(1, 1)  # P(digit 1 in position 1)
# Returns ~30.1% (Benford's Law for digit 1)
```

---

### `3.4.py` - Mean and Variance Analysis

#### `calculate_mean_variance(n)`

Statistical analysis of digit probabilities by position.

**Signature:**
```python
def calculate_mean_variance(n: int) -> tuple[float, float]
```

**Parameters:**
- `n` (int): Position in the number

**Returns:**
- `tuple[float, float]`: (mean, variance) for the specified position

**Algorithm:**
1. Calculate probability for each digit (0-9) at position n
2. Compute weighted mean: Σ(digit × probability)
3. Compute variance: Σ(digit² × probability) - mean²
4. Apply recursive averaging for positions > 1

**Recursive Component:**
For n > 1, averages current position with previous position:
```python
mean_avg = 0.5 * (mean_n + mean_n_minus_1)
variance_avg = 0.5 * (variance_n + variance_n_minus_1)
```

---

## Network Analysis Functions

### `3.5.py` - Complex Network Analysis

#### `benford_pmf(x)`

Benford's Law probability mass function.

**Signature:**
```python
def benford_pmf(x: numeric) -> float
```

**Parameters:**
- `x` (numeric): Input value (typically degree or rank)

**Returns:**
- `float`: Benford probability for the given value

**Formula:**
```python
return np.log10(1 + 1 / x)
```

---

#### `plot_degree_and_benford_histogram(graph, graph_type)`

Network degree analysis with Benford comparison.

**Signature:**
```python
def plot_degree_and_benford_histogram(graph: networkx.Graph, graph_type: str) -> numpy.ndarray
```

**Parameters:**
- `graph` (networkx.Graph): Input network graph
- `graph_type` (str): Type description for labeling

**Returns:**
- `numpy.ndarray`: Differences between actual and Benford distributions

**Algorithm:**
1. Extract degree sequence from graph
2. Create degree distribution histogram
3. Calculate theoretical Benford probabilities
4. Rescale Benford probabilities to match degree range
5. Create comparative visualization
6. Return difference array for analysis

**Visualization Components:**
- Actual degree histogram (bars)
- Theoretical Benford distribution (bars)
- Proper axis labels and legend

---

#### `calculate_and_plot(graph_type, num_nodes, num_edges, seed=None)`

Comprehensive network generation and analysis.

**Signature:**
```python
def calculate_and_plot(graph_type: str, num_nodes: int, num_edges: int, seed: int = None) -> None
```

**Parameters:**
- `graph_type` (str): Network type ('scale-free', 'random', 'small-world')
- `num_nodes` (int): Number of nodes in the graph
- `num_edges` (int): Number of edges or degree parameter
- `seed` (int, optional): Random seed for reproducibility

**Supported Network Types:**

**Scale-free networks:**
- Generated using Barabási-Albert model
- `nx.barabasi_albert_graph(num_nodes, num_edges, seed=seed)`
- Preferential attachment mechanism
- Power-law degree distribution

**Random networks:**
- Generated using Erdős-Rényi model
- `nx.erdos_renyi_graph(num_nodes, 0.03, seed=seed)`
- Random edge placement
- Poisson degree distribution

**Small-world networks:**
- Generated using Watts-Strogatz model
- `nx.watts_strogatz_graph(num_nodes, num_edges, p=0.3, seed=seed)`
- Regular lattice with random rewiring
- High clustering, short path lengths

**Analysis Output:**
- Network statistics (nodes, edges, degrees)
- Benford conformity analysis
- Sum of differences calculation
- Comprehensive visualization

---

## Genetic Algorithm Functions

### `3.2.2.py` - Factorial and Powers Analysis

#### `generate_factorial_numbers(size)`

Factorial sequence generator.

**Signature:**
```python
def generate_factorial_numbers(size: int) -> list[int]
```

**Parameters:**
- `size` (int): Number of factorial values to generate

**Returns:**
- `list[int]`: Factorial sequence [1!, 2!, ..., size!]

**Implementation:**
```python
from math import factorial
return [factorial(i) for i in range(1, size + 1)]
```

**Mathematical Properties:**
- Rapid growth: n! = n × (n-1)!
- First few values: 1, 2, 6, 24, 120, 720, ...
- Often follows Benford's Law for large ranges

---

#### `generate_powers_of_two(size)`

Powers of two sequence generator.

**Signature:**
```python
def generate_powers_of_two(size: int) -> list[int]
```

**Parameters:**
- `size` (int): Number of powers to generate

**Returns:**
- `list[int]`: Powers sequence [2¹, 2², ..., 2^size]

**Implementation:**
```python
return [2 ** i for i in range(1, size + 1)]
```

**Mathematical Properties:**
- Exponential growth: 2ⁿ
- Known to conform well to Benford's Law
- Used as test case for distribution analysis

---

### `3.2.3.py` - Pareto Distribution Optimization

#### `pareto_distribution_pdf(x, a)`

Pareto distribution probability density function.

**Signature:**
```python
def pareto_distribution_pdf(x: numeric, a: float) -> float
```

**Parameters:**
- `x` (numeric): Input value (x ≥ 1)
- `a` (float): Shape parameter (a > 0)

**Returns:**
- `float`: PDF value

**Mathematical Formula:**
```
f(x; a) = a / x^(a+1)
```

**Properties:**
- Heavy-tailed distribution
- Scale-invariant
- Often follows Benford's Law

---

#### `pareto_distribution(size, a)`

Pareto random sample generator.

**Signature:**
```python
def pareto_distribution(size: int, a: float) -> numpy.ndarray
```

**Parameters:**
- `size` (int): Number of samples
- `a` (float): Shape parameter

**Returns:**
- `numpy.ndarray`: Random samples from Pareto distribution

**Implementation:**
```python
return np.random.pareto(a, size)
```

---

#### `fitness_function(params)`

Genetic algorithm fitness function for Pareto optimization.

**Signature:**
```python
def fitness_function(params: array_like) -> float
```

**Parameters:**
- `params` (array-like): Parameters to optimize (Pareto shape parameter)

**Returns:**
- `float`: Chi-squared statistic (lower values indicate better fit)

**Algorithm:**
1. Extract shape parameter from params array
2. Generate Pareto samples with given parameter
3. Extract first digits from samples
4. Calculate observed probability distribution
5. Compare with theoretical Benford distribution using chi-squared test
6. Return chi-squared statistic as fitness metric

**Optimization Goal:**
Minimize chi-squared statistic to find Pareto parameters that best conform to Benford's Law.

---

### `3.2.4.py` - Weibull Distribution Optimization

#### `weibull_distribution_pdf(x, a, b)`

Weibull distribution probability density function.

**Signature:**
```python
def weibull_distribution_pdf(x: numeric, a: float, b: float) -> float
```

**Parameters:**
- `x` (numeric): Input value (x ≥ 0)
- `a` (float): Shape parameter (a > 0)
- `b` (float): Scale parameter (b > 0)

**Mathematical Formula:**
```
f(x; a, b) = (a/b) × (x/b)^(a-1) × exp(-(x/b)^a)
```

---

#### `weibull_distribution(size, a, b)`

Weibull random sample generator.

**Parameters:**
- `size` (int): Number of samples
- `a` (float): Shape parameter
- `b` (float): Scale parameter

**Implementation:**
Uses numpy's Weibull generator with parameter transformation.

---

### `3.2.5.py` - Lognormal Distribution Optimization

#### `lognormal_distribution_pdf(x, mu, sigma)`

Lognormal distribution probability density function.

**Signature:**
```python
def lognormal_distribution_pdf(x: numeric, mu: float, sigma: float) -> float
```

**Parameters:**
- `x` (numeric): Input value (x > 0)
- `mu` (float): Mean of underlying normal distribution
- `sigma` (float): Standard deviation of underlying normal distribution

**Mathematical Formula:**
```
f(x; μ, σ) = (1/(x × σ × √(2π))) × exp(-((ln(x) - μ)² / (2σ²)))
```

---

#### `lognormal_distribution(size, mu, sigma)`

Lognormal random sample generator.

**Implementation:**
```python
return np.random.lognormal(mu, sigma, size)
```

---

## Utility Functions

### Common Patterns Across Modules

#### Chi-squared Test Implementation

Standard pattern used across genetic algorithm modules:

```python
def calculate_chi_squared(observed_probs, expected_probs):
    """
    Calculate chi-squared statistic for goodness of fit
    
    Formula: χ² = Σ((observed - expected)² / expected)
    """
    return np.sum((observed_probs - expected_probs) ** 2 / expected_probs)
```

#### First Digit Extraction

Common utility pattern:

```python
def extract_first_digits(data):
    """Extract first digits from numeric data"""
    return [int(str(abs(val))[0]) for val in data if val > 0]
```

#### Probability Distribution Calculation

Standard normalization:

```python
def calculate_probabilities(counts):
    """Convert counts to probability distribution"""
    return counts / counts.sum()
```

---

## Visualization Functions

### Common Matplotlib Patterns

#### Bar Plot Comparison

Standard pattern for distribution comparison:

```python
def create_comparison_plot(observed, theoretical, labels):
    x = np.arange(1, 10)
    width = 0.35
    
    plt.bar(x - width/2, observed, width, label=labels[0], alpha=0.7)
    plt.bar(x + width/2, theoretical, width, label=labels[1], alpha=0.7)
    
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.xticks(x)
    plt.legend()
    plt.show()
```

#### Heatmap Visualization

Used in modules like 3.3.py:

```python
import seaborn as sns

def create_heatmap(data, labels):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, cmap='viridis', annot=True, fmt='.2f', 
                linewidths=.5, cbar_kws={'label': 'Probability (%)'})
    plt.title('Probability Heatmap')
    plt.show()
```

#### Network Visualization

Pattern for network analysis:

```python
def plot_network_analysis(graph, degrees, analysis_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Degree distribution
    axes[0,0].hist(degrees, bins=50)
    axes[0,0].set_title('Degree Distribution')
    
    # Benford comparison
    # ... additional plots
    
    plt.tight_layout()
    plt.show()
```

---

## Performance Considerations

### Memory Optimization

For large datasets:

```python
def process_large_dataset(data, chunk_size=10000):
    """Process data in chunks to manage memory usage"""
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        result = process_chunk(chunk)
        results.append(result)
    return combine_results(results)
```

### Vectorization

Prefer NumPy vectorized operations:

```python
# Efficient
digits = np.array([int(str(x)[0]) for x in data])
probabilities = np.histogram(digits, bins=np.arange(0.5, 10.5, 1))[0]

# Less efficient
probabilities = []
for digit in range(1, 10):
    count = sum(1 for d in digits if d == digit)
    probabilities.append(count)
```

### Random Seed Management

For reproducible results:

```python
def set_random_seed(seed=42):
    """Set random seed for reproducible results"""
    np.random.seed(seed)
    # Add other random state setters as needed
```

---

## Error Handling Patterns

### Input Validation

```python
def validate_input(data, min_size=100):
    """Validate input data for analysis"""
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("Data must be list or numpy array")
    
    if len(data) < min_size:
        raise ValueError(f"Data size {len(data)} below minimum {min_size}")
    
    numeric_data = [x for x in data if isinstance(x, (int, float)) and x > 0]
    if len(numeric_data) < min_size:
        raise ValueError("Insufficient valid numeric data")
    
    return numeric_data
```

### Graceful Degradation

```python
def safe_calculation(func, *args, **kwargs):
    """Safely execute calculation with fallback"""
    try:
        return func(*args, **kwargs)
    except (ValueError, ZeroDivisionError) as e:
        logger.warning(f"Calculation failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

This comprehensive function reference provides detailed documentation for all public APIs, functions, and components in the Benford Distribution Analysis project, enabling users to effectively utilize and extend the codebase.