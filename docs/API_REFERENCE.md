# Benford Distribution Analysis – API Reference

This reference lists all public functions and executable modules contained in the `src/` directory.  For each function you will find a concise description, its parameters, return values, and a runnable example.  Where appropriate, the example can be copied directly into a Python REPL or script once the project’s dependencies are installed.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Module Overview](#module-overview)
3. [Function Reference](#function-reference)
    * [`src/main.py`](#srcmainpy)
    * [`src/3.1.py`](#src31py)
    * [`src/3.2.py`](#src32py)
    * [`src/3.2.2.py`](#src322py)
    * [`src/3.2.3.py`](#src323py)
    * [`src/3.2.4.py`](#src324py)
    * [`src/3.2.5.py`](#src325py)
    * [`src/3.2.5.2.py`](#src3252py)
    * [`src/3.3.py`](#src33py)
    * [`src/3.4.py`](#src34py)
    * [`src/3.5.py`](#src35py)
    * [`src/3.5.2.py`](#src352py)
4. [Executable / Notebook-Style Scripts](#executable--notebook-style-scripts)

---

## Getting Started

```bash
# 1. (optional) create & activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. install core dependencies
pip install -r requirements.txt

# 3. run one of the example scripts
python src/main.py
```

Python ≥ 3.9 is recommended.  The project relies on NumPy, SciPy, Matplotlib, NetworkX, Pandas, Seaborn, and `geneticalgorithm`.  A pinned list is provided in `requirements.txt`.

---

## Module Overview

| Module | Purpose |
| ------ | ------- |
| `main.py` | Generates Benford-distributed digits, then plots the average histogram over multiple iterations. |
| `1.1.py` | Compares hypergeometric vs. binomial distributions for selected sample sizes. |
| `2.py` | Demonstrates Zipf-like word-frequency behaviour in a text corpus and visualises the result on both linear and log-log axes. |
| `3.1.py` | Generates Benford numbers, fits both Benford’s and Zipf’s laws to the observed data, and visualises the fit. |
| `3.2.py` | Analyses the leading digits of physical constants and compares the empirical distribution against Benford’s law. |
| `3.2.2.py` | Compares factorial numbers and powers-of-two against Benford’s law. |
| `3.2.3.py` | Uses a genetic algorithm to tune a Pareto distribution so that its leading-digit distribution matches Benford’s law. |
| `3.2.4.py` | As above, but for a Weibull distribution. |
| `3.2.5.py` | As above, but for a log-normal distribution. |
| `3.2.5.2.py` | Alternative implementation of the log-normal optimisation experiment. |
| `3.3.py` | Computes the generalised Benford probability of each digit appearing at a given position. |
| `3.4.py` | Calculates the mean and variance of the generalised Benford distribution for different digit positions. |
| `3.5.py` | Evaluates graph-degree distributions (scale-free vs. Erdős–Rényi) against Benford’s law. |
| `3.5.2.py` | Quantifies the conformity of a graph’s degree sequence to Benford’s law using J-divergence and Jensen–Shannon divergence. |

---

## Function Reference

Below, *parameters* are given as `name: type (default) – description`.  Unless noted, functions return `None`.

### src/main.py

#### `generate_benford_numbers(size)`
* **size**: `int` – Number of random digits to sample.
* **Returns**: `numpy.ndarray[int]` of length *size* containing digits 1-9 drawn from Benford’s law.

```python
from src.main import generate_benford_numbers

numbers = generate_benford_numbers(1000)
print(numbers[:10])  # e.g. [1 9 2 1 1 3 1 2 1 1]
```

#### `plot_average_histogram(iterations, size, bins)`
* **iterations**: `int` – How many independent Monte-Carlo draws to average over.
* **size**: `int` – Sample size per iteration.
* **bins**: `numpy.ndarray` – Bin edges for `numpy.histogram`; pass `np.arange(0.5, 10.5, 1)` for digit bins.

Displays a Matplotlib bar plot of the averaged histogram.

Example:

```python
import numpy as np
from src.main import plot_average_histogram

plot_average_histogram(iterations=100, size=1000, bins=np.arange(0.5, 10.5, 1))
```

---

### src/3.1.py

#### `generate_benford_numbers(size)`
Identical to `src.main.generate_benford_numbers` but defined locally for this module.

#### `fit_benford_zipf_laws(data)`
* **data**: `array-like[int]` – Sequence of digits 1-9.
* **Returns**: `(observed_counts, benford_probs, zipf_probs)` where
  * `observed_counts`: `numpy.ndarray[int]` – raw counts per digit;
  * `benford_probs`: `numpy.ndarray[float]` – theoretical Benford probabilities;
  * `zipf_probs`: `numpy.ndarray[float]` – theoretical Zipf probabilities with α = 1.

#### `plot_laws_fit(observed_counts, benford_probs, zipf_probs)`
Draws a grouped bar chart comparing the empirical distribution to Benford and Zipf.

Example Workflow:

```python
from src import _3_1 as exp31  # import src/3.1.py as a module (rename to a valid identifier first!)

digits = exp31.generate_benford_numbers(1000)
obs, ben, zipf = exp31.fit_benford_zipf_laws(digits)
exp31.plot_laws_fit(obs, ben, zipf)
```

*(Tip: because `3.1.py` is not a valid Python identifier, rename to `exp31.py` or import using `import importlib.util`.)*

---

### src/3.2.py

#### `is_numeric_constant(value)`
Returns `True` if *value* is an `int`, `float`, or NumPy scalar type.

#### `get_most_significant_digits(constant_values)`
Extracts the most-significant (leading) digit from every numeric value in *constant_values*.

#### `benford_law_distribution()`
Returns the normalised Benford probability mass function (length 9 `numpy.ndarray`).

#### `print_digit_counts(data)`
Pretty-prints the count of each digit to stdout.

#### `plot_distribution(data, label)`
Plots the leading-digit distribution of *data* alongside Benford’s law.

Example:

```python
import numpy as np
from src import _3_2 as exp32

sample = np.random.lognormal(size=1000)
digits = exp32.get_most_significant_digits(sample)
exp32.plot_distribution(digits, label="Log-normal sample")
```

---

### src/3.2.2.py

*Functions specific to factorials and powers-of-two.*

| Function | Description |
| -------- | ----------- |
| `generate_factorial_numbers(size)` | Returns `[1!, 2!, …, size!]`. |
| `generate_powers_of_two(size)` | Returns `[2¹, 2², …, 2^size]`. |
| `get_most_significant_digits(numbers)` | Shared utility. |
| `benford_law_distribution()` | Shared utility. |
| `print_digit_counts(data, label)` | Print helper. |
| `plot_distribution(data, label, color)` | Visualises distribution vs. Benford. |

Example:

```python
from src import _3_2_2 as exp322

facts = exp322.generate_factorial_numbers(50)
exp322.plot_distribution(exp322.get_most_significant_digits(facts), "Factorials", color="green")
```

---

### src/3.2.3.py — Pareto Optimisation

| Function | Purpose |
| -------- | ------- |
| `pareto_distribution_pdf(x, a)` | Analytical PDF of a Pareto distribution. |
| `pareto_distribution(size, a)` | Random variate generator. |
| `get_most_significant_digits(numbers)` | Utility. |
| `benford_law_distribution()` | Utility. |
| `fitness_function(params)` | Objective passed to the GA; minimises χ² distance to Benford. |

Usage snippet (after ensuring `geneticalgorithm` is installed):

```python
from src import _3_2_3 as exp323

chi2 = exp323.fitness_function([1.5])
print(chi2)
```

---

### src/3.2.4.py — Weibull Optimisation

Same structure as the Pareto variant but operating on a Weibull distribution.  Key public helpers:

* `weibull_distribution_pdf(x, c)`
* `weibull_distribution(size, c)`
* `fitness_function(params)`

---

### src/3.2.5.py / 3.2.5.2.py — Log-Normal Optimisation

Both files optimise the σ parameter of a log-normal distribution.  Main helpers:

* `log_normal_distribution_pdf(x, sigma)`
* `log_normal_distribution(size, sigma)`
* `fitness_function(params)`

---

### src/3.3.py — Generalised Benford

* `benford_law_digit_probability(n, d, sigma=1.0)` – Probability (in %) that digit *d* appears at position *n* under the generalised Benford distribution with scale *σ*.

Example:

```python
from src import _3_3 as exp33
print(exp33.benford_law_digit_probability(2, 5))  # probability of "5" as the 2nd digit
```

---

### src/3.4.py — Mean & Variance of Generalised Benford

Public helpers:

* `benford_law_digit_probability(n, d, sigma=1.0)` – identical to the implementation in `3.3.py`.
* `calculate_mean_variance(n)` – Returns two lists: mean and variance of the digit probabilities for digit positions *n* = 1…*n*.

---

### src/3.5.py — Graph Degree vs. Benford

| Function | Purpose |
| -------- | ------- |
| `benford_pmf(x)` | Benford PMF evaluated at *x*. |
| `plot_degree_and_benford_histogram(graph, graph_type)` | Side-by-side histogram comparing a graph’s degree distribution with Benford’s law. |
| `calculate_and_plot(graph_type, num_nodes, num_edges, seed=None)` | Generates a graph (scale-free or random), then calls the above plotting helper. |

Example:

```python
from src import _3_5 as exp35
exp35.calculate_and_plot("scale-free", num_nodes=1000, num_edges=3, seed=42)
```

---

### src/3.5.2.py — Divergence Metrics

Additional metrics for quantifying deviation from Benford:

* `observed_degree_distribution(graph)` – Returns empirical PMF of a graph’s degree sequence.
* `j_divergence(p, q)` – Symmetric divergence based on Kullback–Leibler.
* `jensen_shannon_divergence(p, q)` – Square-root of Jensen–Shannon divergence.
* `calculate_and_plot(graph_type, num_nodes, num_edges, seed=None)` – Convenience wrapper that prints divergences and visualises distributions.

---

## Executable / Notebook-Style Scripts

Some files (e.g. `1.1.py`, `2.py`, and most `3.*.py`) are intended to be run as standalone scripts.  Each script sets up its own parameters under the `if __name__ == "__main__":` guard (or simply in global scope).  To reproduce the figures in the report, simply execute:

```bash
python src/1.1.py   # distribution comparison
python src/2.py     # word-frequency analysis
python src/3.2.4.py # Weibull GA optimisation
```

---

## Contributing / Extending

1. Add unit tests under `tests/` (pytest recommended).
2. Prefer pure functions; avoid hard-coding parameters in global scope when possible.
3. Adhere to PEP-8 style and add docstrings following the NumPy convention.

---

Happy analysing!