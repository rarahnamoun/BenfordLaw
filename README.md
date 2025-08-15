# Benford Distribution Analysis with Genetic Algorithms and Complex Network Models

This repository contains comprehensive code and documentation for a project that explores the application of genetic algorithms (GAs) to determine optimal parameters for fitting the Benford distribution to Weibull, lognormal, and Pareto distributions. Additionally, it assesses the J-Divergence of complex network models concerning their conformity to Benford's Law and investigates the probability density function (PDF) of Benford's Law.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic Benford analysis
cd src && python main.py

# Verify installation
python -c "import numpy, matplotlib, scipy, pandas, seaborn, networkx; print('✓ All dependencies installed!')"
```

## Project Structure

```
├── src/                    # Source code modules
│   ├── main.py            # Basic Benford number generation
│   ├── 1.1.py             # Hypergeometric vs Binomial comparison
│   ├── 2.py, 2.1.py       # Text analysis and Zipf's Law
│   ├── 3.1.py             # Benford vs Zipf comparison
│   ├── 3.2.py             # Physical constants analysis
│   ├── 3.3.py, 3.4.py     # Generalized Benford's Law
│   ├── 3.5.py             # Complex network analysis
│   └── 3.2.*.py           # Genetic algorithm optimizations
├── API_DOCUMENTATION.md   # Comprehensive API reference
├── USAGE_EXAMPLES.md      # Detailed usage examples and tutorials
├── FUNCTION_REFERENCE.md  # Complete function documentation
├── QUICK_START.md         # 10-minute quick start guide
├── requirements.txt       # Python dependencies
└── report.pdf            # Detailed methodology and results

```

## Documentation

| Document | Purpose |
|----------|---------|
| **[QUICK_START.md](QUICK_START.md)** | Get up and running in 10 minutes |
| **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** | Comprehensive API reference with examples |
| **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** | Detailed tutorials and advanced workflows |
| **[FUNCTION_REFERENCE.md](FUNCTION_REFERENCE.md)** | Complete function documentation |

## Core Features

### 🔢 Benford's Law Analysis
- Generate numbers following Benford's Law
- Analyze real-world datasets for conformity
- Statistical validation using chi-squared tests
- Comprehensive visualization tools

### 🧬 Genetic Algorithm Optimization
- Optimize Pareto distribution parameters
- Optimize Weibull distribution parameters  
- Optimize Lognormal distribution parameters
- Find parameters that best fit Benford's Law

### 🌐 Network Analysis
- Analyze degree distributions in complex networks
- Support for scale-free, random, and small-world networks
- Compare network properties to Benford's Law
- Statistical analysis of network conformity

### 📊 Statistical Analysis
- Chi-squared goodness-of-fit tests
- Distribution comparison tools
- Physical constants analysis
- Text analysis and Zipf's Law demonstration

## Example Usage

### Basic Analysis
```python
import sys
sys.path.append('src')
from main import generate_benford_numbers
import numpy as np

# Generate and analyze Benford numbers
numbers = generate_benford_numbers(1000)
print(f"Sample: {numbers[:10]}")
```

### Quick Dataset Analysis
```python
# Analyze your own data
data = [2**i for i in range(1, 100)]  # Powers of 2
first_digits = [int(str(abs(val))[0]) for val in data if val > 0]

# Compare to Benford's Law
observed = np.histogram(first_digits, bins=np.arange(0.5, 10.5, 1))[0]
benford = np.log10(1 + 1 / np.arange(1, 10))
chi_squared = np.sum((observed/observed.sum() - benford/benford.sum()) ** 2 / (benford/benford.sum()))
print(f"Chi-squared: {chi_squared:.3f} ({'Conforms' if chi_squared < 15.507 else 'Does not conform'})")
```

### Network Analysis
```python
exec(open('src/3.5.py').read())  # Analyze different network types
```

### Genetic Algorithm Optimization
```python
exec(open('src/3.2.3.py').read())  # Optimize Pareto parameters
```

## Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `generate_benford_numbers(size)` | main.py | Generate random Benford numbers |
| `plot_average_histogram(iterations, size, bins)` | main.py | Average histogram visualization |
| `calculate_and_plot(graph_type, nodes, edges)` | 3.5.py | Network analysis |
| `fitness_function(params)` | 3.2.*.py | GA optimization fitness |

## Installation Requirements

- Python 3.7+
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- seaborn >= 0.11.0
- networkx >= 2.6.0
- geneticalgorithm >= 1.0.2

## Research Applications

This toolkit is valuable for:
- **Fraud Detection**: Detecting anomalies in financial data
- **Data Quality Assessment**: Validating naturally occurring datasets  
- **Network Science**: Analyzing real-world network properties
- **Statistical Modeling**: Optimizing distribution parameters
- **Educational Research**: Teaching statistical distributions and optimization

## Contributing

The codebase is designed for extensibility. Key extension points:
- Add new distribution types in genetic algorithm modules
- Implement additional network models in network analysis
- Create custom fitness functions for optimization
- Add new statistical tests and validation methods

## Citation

If you use this code in your research, please reference:
```
Benford Distribution Analysis with Genetic Algorithms and Complex Network Models
[Include appropriate citation details]
```

## License

[Include license information]

---

**Get Started**: Check out [QUICK_START.md](QUICK_START.md) to begin analyzing your data with Benford's Law in just 10 minutes!
