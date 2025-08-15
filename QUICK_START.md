# Quick Start Guide

## Overview
This guide will get you up and running with the Benford Distribution Analysis project in under 10 minutes.

## Prerequisites
- Python 3.7 or higher
- pip package manager

## Installation (2 minutes)

1. **Clone or download the project**
   ```bash
   # If you have the project files
   cd benford-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import numpy, matplotlib, scipy, pandas, seaborn, networkx; print('✓ All dependencies installed successfully!')"
   ```

## Your First Analysis (3 minutes)

### 1. Basic Benford's Law Visualization
```bash
cd src
python main.py
```
This will generate and display a histogram showing how randomly generated numbers follow Benford's Law.

### 2. Analyze Physical Constants
```python
# Run this in Python
import sys
sys.path.append('src')
exec(open('src/3.2.py').read())
```
This analyzes physical constants and compares their first digits to Benford's Law.

### 3. Network Analysis
```python
# Run this in Python
exec(open('src/3.5.py').read())
```
This generates different types of networks and analyzes their degree distributions.

## Common Tasks

### Generate Benford Numbers
```python
import sys
sys.path.append('src')
from main import generate_benford_numbers

# Generate 1000 numbers following Benford's Law
numbers = generate_benford_numbers(1000)
print(f"First 10 numbers: {numbers[:10]}")
print(f"Distribution: {np.bincount(numbers)[1:]}")
```

### Quick Distribution Analysis
```python
import numpy as np
import matplotlib.pyplot as plt

# Your data
data = [2**i for i in range(1, 100)]  # Powers of 2

# Extract first digits
first_digits = [int(str(abs(val))[0]) for val in data if val > 0]

# Calculate observed probabilities
digit_counts = np.histogram(first_digits, bins=np.arange(0.5, 10.5, 1))[0]
observed_probs = digit_counts / digit_counts.sum()

# Benford's Law probabilities
benford_probs = np.log10(1 + 1 / np.arange(1, 10))
benford_probs /= benford_probs.sum()

# Plot comparison
x = np.arange(1, 10)
plt.bar(x - 0.2, observed_probs, 0.4, label='Your Data', alpha=0.7)
plt.bar(x + 0.2, benford_probs, 0.4, label="Benford's Law", alpha=0.7)
plt.xlabel('First Digit')
plt.ylabel('Probability')
plt.legend()
plt.title('Benford\'s Law Analysis')
plt.show()

# Chi-squared test
chi_squared = np.sum((observed_probs - benford_probs) ** 2 / benford_probs)
print(f"Chi-squared statistic: {chi_squared:.4f}")
print(f"Conforms to Benford's Law: {'Yes' if chi_squared < 15.507 else 'No'}")
```

### Genetic Algorithm Optimization
```python
# Run Pareto distribution optimization
exec(open('src/3.2.3.py').read())
```

## Example Datasets to Try

### 1. Fibonacci Numbers
```python
def fibonacci(n):
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

fib_numbers = fibonacci(100)
# Analyze with the quick analysis code above
```

### 2. Population Data
```python
# Example city populations (replace with real data)
populations = [8_400_000, 3_900_000, 2_700_000, 2_300_000, 1_600_000, 
               883_000, 695_000, 632_000, 467_000, 382_000]
# Analyze with the quick analysis code above
```

### 3. Financial Data
```python
# Example stock prices or financial figures
financial_data = [156.23, 234.56, 89.12, 445.67, 23.45, 678.90, 
                  123.45, 567.89, 345.67, 789.01]
# Analyze with the quick analysis code above
```

## Understanding Results

### Chi-squared Statistic
- **< 15.507**: Data likely follows Benford's Law (at α = 0.05)
- **≥ 15.507**: Data does not follow Benford's Law

### Visual Inspection
- **Good fit**: Observed bars closely match Benford's Law bars
- **Poor fit**: Large differences between observed and theoretical

### Common Patterns
- **Powers and exponentials**: Usually follow Benford's Law well
- **Uniform random numbers**: Do not follow Benford's Law
- **Real-world datasets**: Often follow Benford's Law partially

## Troubleshooting

### Import Errors
```bash
# Install missing packages individually
pip install numpy matplotlib scipy pandas seaborn networkx geneticalgorithm
```

### Module Not Found
```python
# Make sure to add src to Python path
import sys
sys.path.append('src')
```

### File Not Found (Text-2.txt)
- The text analysis modules require `Text-2.txt` in the src directory
- You can skip these modules or provide your own text file

### Memory Issues
- Reduce sample sizes for large datasets
- Use smaller population sizes for genetic algorithms

## Next Steps

1. **Read the full documentation**: Check `API_DOCUMENTATION.md` for comprehensive function reference
2. **Try the examples**: Explore `USAGE_EXAMPLES.md` for advanced workflows  
3. **Function reference**: See `FUNCTION_REFERENCE.md` for detailed API documentation
4. **Experiment**: Try your own datasets and see how they conform to Benford's Law

## Quick Reference

### Key Functions
- `generate_benford_numbers(size)`: Generate random Benford numbers
- `plot_average_histogram(iterations, size, bins)`: Average histogram visualization
- `benford_law_distribution()`: Theoretical Benford probabilities
- `calculate_and_plot(graph_type, nodes, edges)`: Network analysis

### Key Files
- `main.py`: Basic Benford number generation
- `3.2.py`: Physical constants analysis  
- `3.5.py`: Network analysis
- `3.2.3.py`: Genetic algorithm optimization (Pareto)
- `3.2.4.py`: Genetic algorithm optimization (Weibull)
- `3.2.5.py`: Genetic algorithm optimization (Lognormal)

### Statistical Interpretation
- **Chi-squared < 15.507**: Follows Benford's Law (p < 0.05)
- **Sample size**: Recommend at least 100 data points
- **First digit only**: Analysis focuses on leading digits 1-9

That's it! You're now ready to explore Benford's Law with your own data.