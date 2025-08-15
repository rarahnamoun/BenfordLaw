# Benford Distribution Analysis - Usage Examples and Tutorial

## Table of Contents
1. [Quick Start](#quick-start)
2. [Basic Examples](#basic-examples)
3. [Advanced Analysis Workflows](#advanced-analysis-workflows)
4. [Genetic Algorithm Optimization](#genetic-algorithm-optimization)
5. [Network Analysis](#network-analysis)
6. [Integration Examples](#integration-examples)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation and Setup

```bash
# Clone the repository (if applicable)
# cd benford-analysis

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, matplotlib, scipy, pandas, seaborn, networkx; print('All dependencies installed successfully!')"
```

### Running Your First Analysis

```python
# Navigate to the src directory
cd src

# Run basic Benford's Law visualization
python main.py
```

## Basic Examples

### 1. Generate and Analyze Benford Numbers

```python
import sys
sys.path.append('src')  # Add src to Python path

import numpy as np
import matplotlib.pyplot as plt
from main import generate_benford_numbers, plot_average_histogram

# Example 1: Generate Benford numbers
print("Example 1: Generating Benford's Law numbers")
numbers = generate_benford_numbers(1000)
print(f"Generated {len(numbers)} numbers")
print(f"First 20 numbers: {numbers[:20]}")
print(f"Distribution of digits: {np.bincount(numbers)[1:]}")

# Example 2: Create averaged histogram
print("\nExample 2: Creating averaged histogram")
bins = np.arange(0.5, 10.5, 1)
plot_average_histogram(50, 1000, bins)
```

### 2. Compare Distributions

```python
# Compare Hypergeometric vs Binomial distributions
exec(open('src/1.1.py').read())

# Analyze text with Zipf's Law
exec(open('src/2.1.py').read())
```

### 3. Physical Constants Analysis

```python
import sys
sys.path.append('src')

from scipy import constants
import numpy as np

# Import functions from 3.2.py
exec(open('src/3.2.py').read())

# Analyze a subset of physical constants
selected_constants = [
    constants.c,           # Speed of light
    constants.h,           # Planck constant
    constants.k,           # Boltzmann constant
    constants.e,           # Elementary charge
    constants.m_e,         # Electron mass
    constants.m_p,         # Proton mass
    constants.N_A,         # Avogadro constant
    constants.R,           # Gas constant
    constants.g,           # Standard gravity
    constants.epsilon_0    # Vacuum permittivity
]

digits = get_most_significant_digits(selected_constants * 100)  # Replicate for better statistics
plot_distribution(digits, 'Selected Physical Constants')
```

## Advanced Analysis Workflows

### 1. Comprehensive Benford's Law Analysis Pipeline

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

def comprehensive_benford_analysis(data_source, data_label):
    """
    Comprehensive Benford's Law analysis pipeline
    
    Parameters:
    -----------
    data_source : callable or array-like
        Function to generate data or pre-existing data
    data_label : str
        Label for the analysis
    """
    
    # Step 1: Generate or prepare data
    if callable(data_source):
        data = data_source()
    else:
        data = data_source
    
    # Step 2: Extract most significant digits
    if hasattr(data[0], '__iter__'):  # If data contains arrays/lists
        all_digits = []
        for item in data:
            digits = [int(str(abs(val))[0]) for val in item if val != 0]
            all_digits.extend(digits)
    else:
        all_digits = [int(str(abs(val))[0]) for val in data if val != 0]
    
    # Step 3: Calculate observed distribution
    digit_counts = np.histogram(all_digits, bins=np.arange(0.5, 10.5, 1))[0]
    observed_probs = digit_counts / digit_counts.sum()
    
    # Step 4: Calculate theoretical Benford distribution
    digits = np.arange(1, 10)
    benford_probs = np.log10(1 + 1 / digits)
    benford_probs /= benford_probs.sum()
    
    # Step 5: Statistical comparison
    chi_squared = np.sum((observed_probs - benford_probs) ** 2 / benford_probs)
    
    # Step 6: Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot comparison
    x = np.arange(1, 10)
    width = 0.35
    ax1.bar(x - width/2, observed_probs, width, label=f'Observed ({data_label})', alpha=0.7)
    ax1.bar(x + width/2, benford_probs, width, label="Benford's Law", alpha=0.7)
    ax1.set_xlabel('Digit')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Distribution Comparison: {data_label}')
    ax1.legend()
    ax1.set_xticks(x)
    
    # Residuals plot
    residuals = observed_probs - benford_probs
    ax2.bar(x, residuals, color='red', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Residual (Observed - Benford)')
    ax2.set_title('Residuals Analysis')
    ax2.set_xticks(x)
    
    plt.tight_layout()
    plt.show()
    
    # Step 7: Report results
    print(f"\n{data_label} Analysis Results:")
    print(f"Sample size: {len(all_digits)} digits")
    print(f"Chi-squared statistic: {chi_squared:.4f}")
    print(f"Degrees of freedom: 8")
    print(f"Critical value (α=0.05): 15.507")
    print(f"Conforms to Benford's Law: {'Yes' if chi_squared < 15.507 else 'No'}")
    
    return {
        'observed_probs': observed_probs,
        'benford_probs': benford_probs,
        'chi_squared': chi_squared,
        'sample_size': len(all_digits)
    }

# Example usage: Analyze factorial numbers
from math import factorial

def generate_factorials():
    return [factorial(i) for i in range(1, 101)]

results = comprehensive_benford_analysis(generate_factorials, "Factorial Numbers (1! to 100!)")
```

### 2. Multi-Distribution Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def compare_distributions_to_benford():
    """Compare multiple probability distributions to Benford's Law"""
    
    # Parameters
    sample_size = 10000
    
    # Generate data from different distributions
    distributions = {
        'Exponential (λ=1)': np.random.exponential(1, sample_size),
        'Lognormal (μ=0, σ=1)': np.random.lognormal(0, 1, sample_size),
        'Pareto (a=1.16)': np.random.pareto(1.16, sample_size) + 1,
        'Uniform (1, 1000)': np.random.uniform(1, 1000, sample_size),
        'Normal (μ=100, σ=50)': np.abs(np.random.normal(100, 50, sample_size))
    }
    
    # Calculate Benford's Law probabilities
    benford_probs = np.log10(1 + 1 / np.arange(1, 10))
    benford_probs /= benford_probs.sum()
    
    # Analyze each distribution
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(distributions.items()):
        # Extract first digits
        first_digits = [int(str(abs(val))[0]) for val in data if val > 0 and not np.isnan(val)]
        
        # Calculate observed probabilities
        digit_counts = np.histogram(first_digits, bins=np.arange(0.5, 10.5, 1))[0]
        observed_probs = digit_counts / digit_counts.sum()
        
        # Chi-squared test
        chi_squared = np.sum((observed_probs - benford_probs) ** 2 / benford_probs)
        
        # Plot
        x = np.arange(1, 10)
        axes[i].bar(x - 0.2, observed_probs, 0.4, label=name, alpha=0.7)
        axes[i].bar(x + 0.2, benford_probs, 0.4, label="Benford's Law", alpha=0.7)
        axes[i].set_title(f'{name}\nχ² = {chi_squared:.3f}')
        axes[i].set_xlabel('First Digit')
        axes[i].set_ylabel('Probability')
        axes[i].legend()
        axes[i].set_xticks(x)
        
        results[name] = {'chi_squared': chi_squared, 'sample_size': len(first_digits)}
    
    # Summary plot
    axes[5].bar(range(len(results)), [r['chi_squared'] for r in results.values()])
    axes[5].axhline(y=15.507, color='red', linestyle='--', label='Critical value (α=0.05)')
    axes[5].set_xticks(range(len(results)))
    axes[5].set_xticklabels(list(results.keys()), rotation=45, ha='right')
    axes[5].set_ylabel('Chi-squared statistic')
    axes[5].set_title('Conformity to Benford\'s Law Summary')
    axes[5].legend()
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run the comparison
comparison_results = compare_distributions_to_benford()
```

## Genetic Algorithm Optimization

### 1. Basic GA Setup for Distribution Fitting

```python
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import sys
sys.path.append('src')

def setup_genetic_algorithm(fitness_function, parameter_bounds, algorithm_params=None):
    """
    Generic setup for genetic algorithm optimization
    
    Parameters:
    -----------
    fitness_function : callable
        Function to minimize (lower values = better fitness)
    parameter_bounds : numpy.ndarray
        Array of [min, max] bounds for each parameter
    algorithm_params : dict, optional
        GA algorithm parameters
    
    Returns:
    --------
    ga.geneticalgorithm
        Configured genetic algorithm instance
    """
    
    # Default algorithm parameters
    default_params = {
        'max_num_iteration': 200,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 50
    }
    
    if algorithm_params:
        default_params.update(algorithm_params)
    
    # Create GA model
    model = ga(
        function=fitness_function,
        dimension=len(parameter_bounds),
        variable_type='real',
        variable_boundaries=parameter_bounds,
        algorithm_parameters=default_params
    )
    
    return model

# Example: Optimize Pareto distribution parameters
def optimize_pareto_for_benford():
    """Optimize Pareto distribution parameters to fit Benford's Law"""
    
    def pareto_benford_fitness(params):
        a = params[0]  # Pareto shape parameter
        
        # Generate Pareto samples
        samples = np.random.pareto(a, 10000) + 1
        
        # Extract first digits
        first_digits = [int(str(abs(val))[0]) for val in samples if val > 0]
        
        # Calculate observed probabilities
        digit_counts = np.histogram(first_digits, bins=np.arange(0.5, 10.5, 1))[0]
        observed_probs = digit_counts / digit_counts.sum()
        
        # Benford's Law probabilities
        benford_probs = np.log10(1 + 1 / np.arange(1, 10))
        benford_probs /= benford_probs.sum()
        
        # Return chi-squared statistic as fitness (to minimize)
        return np.sum((observed_probs - benford_probs) ** 2 / benford_probs)
    
    # Parameter bounds: Pareto shape parameter between 0.5 and 5.0
    bounds = np.array([[0.5, 5.0]])
    
    # Setup and run GA
    model = setup_genetic_algorithm(pareto_benford_fitness, bounds)
    model.run()
    
    # Get results
    optimal_params = model.output_dict['variable']
    best_fitness = model.output_dict['function']
    
    print(f"Optimal Pareto shape parameter: {optimal_params[0]:.4f}")
    print(f"Best fitness (chi-squared): {best_fitness:.4f}")
    
    return optimal_params, best_fitness

# Run optimization
optimal_pareto, fitness = optimize_pareto_for_benford()
```

### 2. Multi-Parameter Optimization Example

```python
def optimize_weibull_for_benford():
    """Optimize Weibull distribution parameters to fit Benford's Law"""
    
    def weibull_benford_fitness(params):
        a, b = params  # Weibull shape and scale parameters
        
        try:
            # Generate Weibull samples
            samples = np.random.weibull(a, 10000) * b
            
            # Filter positive values
            samples = samples[samples > 0]
            
            if len(samples) < 100:  # Not enough valid samples
                return 1000  # Large penalty
            
            # Extract first digits
            first_digits = [int(str(abs(val))[0]) for val in samples]
            
            # Calculate observed probabilities
            digit_counts = np.histogram(first_digits, bins=np.arange(0.5, 10.5, 1))[0]
            observed_probs = digit_counts / digit_counts.sum()
            
            # Benford's Law probabilities
            benford_probs = np.log10(1 + 1 / np.arange(1, 10))
            benford_probs /= benford_probs.sum()
            
            # Return chi-squared statistic
            chi_squared = np.sum((observed_probs - benford_probs) ** 2 / benford_probs)
            
            # Add penalty for extreme parameter values
            penalty = 0
            if a < 0.1 or a > 10: penalty += 100
            if b < 0.1 or b > 100: penalty += 100
            
            return chi_squared + penalty
            
        except:
            return 1000  # Return large fitness for invalid parameters
    
    # Parameter bounds: [shape, scale]
    bounds = np.array([[0.1, 10.0], [0.1, 100.0]])
    
    # Custom algorithm parameters for more thorough search
    custom_params = {
        'max_num_iteration': 300,
        'population_size': 150,
        'mutation_probability': 0.15,
        'crossover_probability': 0.7
    }
    
    # Setup and run GA
    model = setup_genetic_algorithm(weibull_benford_fitness, bounds, custom_params)
    model.run()
    
    # Get results
    optimal_params = model.output_dict['variable']
    best_fitness = model.output_dict['function']
    
    print(f"Optimal Weibull parameters:")
    print(f"  Shape (a): {optimal_params[0]:.4f}")
    print(f"  Scale (b): {optimal_params[1]:.4f}")
    print(f"Best fitness (chi-squared): {best_fitness:.4f}")
    
    # Validate results
    samples = np.random.weibull(optimal_params[0], 10000) * optimal_params[1]
    first_digits = [int(str(abs(val))[0]) for val in samples if val > 0]
    
    # Plot results
    digit_counts = np.histogram(first_digits, bins=np.arange(0.5, 10.5, 1))[0]
    observed_probs = digit_counts / digit_counts.sum()
    benford_probs = np.log10(1 + 1 / np.arange(1, 10))
    benford_probs /= benford_probs.sum()
    
    plt.figure(figsize=(10, 6))
    x = np.arange(1, 10)
    plt.bar(x - 0.2, observed_probs, 0.4, label='Optimized Weibull', alpha=0.7)
    plt.bar(x + 0.2, benford_probs, 0.4, label="Benford's Law", alpha=0.7)
    plt.xlabel('First Digit')
    plt.ylabel('Probability')
    plt.title(f'Optimized Weibull Distribution vs Benford\'s Law\nShape={optimal_params[0]:.3f}, Scale={optimal_params[1]:.3f}')
    plt.legend()
    plt.xticks(x)
    plt.show()
    
    return optimal_params, best_fitness

# Run Weibull optimization
optimal_weibull, fitness = optimize_weibull_for_benford()
```

## Network Analysis

### 1. Comprehensive Network Analysis

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

def analyze_network_benford_conformity(graph, graph_name, detailed_analysis=True):
    """
    Comprehensive analysis of network degree distribution vs Benford's Law
    
    Parameters:
    -----------
    graph : networkx.Graph
        Network graph to analyze
    graph_name : str
        Name for the graph (used in plots)
    detailed_analysis : bool
        Whether to perform detailed statistical analysis
    
    Returns:
    --------
    dict
        Analysis results
    """
    
    # Get degree sequence
    degrees = list(dict(graph.degree()).values())
    
    # Filter out zero degrees and get first digits
    non_zero_degrees = [d for d in degrees if d > 0]
    first_digits = [int(str(d)[0]) for d in non_zero_degrees]
    
    # Calculate observed distribution
    digit_counts = np.histogram(first_digits, bins=np.arange(0.5, 10.5, 1))[0]
    observed_probs = digit_counts / digit_counts.sum()
    
    # Benford's Law distribution
    benford_probs = np.log10(1 + 1 / np.arange(1, 10))
    benford_probs /= benford_probs.sum()
    
    # Statistical tests
    chi_squared = np.sum((observed_probs - benford_probs) ** 2 / benford_probs)
    
    # Visualization
    if detailed_analysis:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Degree distribution
        axes[0, 0].hist(degrees, bins=50, alpha=0.7, density=True)
        axes[0, 0].set_xlabel('Degree')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title(f'{graph_name}: Degree Distribution')
        axes[0, 0].set_yscale('log')
        
        # First digit comparison
        x = np.arange(1, 10)
        axes[0, 1].bar(x - 0.2, observed_probs, 0.4, label='Observed', alpha=0.7)
        axes[0, 1].bar(x + 0.2, benford_probs, 0.4, label="Benford's Law", alpha=0.7)
        axes[0, 1].set_xlabel('First Digit')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].set_title(f'{graph_name}: First Digit Distribution')
        axes[0, 1].legend()
        axes[0, 1].set_xticks(x)
        
        # Log-log degree distribution
        degree_counts = np.bincount(degrees)
        non_zero_counts = degree_counts[degree_counts > 0]
        degrees_range = np.arange(len(degree_counts))[degree_counts > 0]
        
        axes[1, 0].loglog(degrees_range, non_zero_counts, 'bo-', alpha=0.7)
        axes[1, 0].set_xlabel('Degree (log scale)')
        axes[1, 0].set_ylabel('Count (log scale)')
        axes[1, 0].set_title(f'{graph_name}: Log-Log Degree Distribution')
        
        # Residuals
        residuals = observed_probs - benford_probs
        axes[1, 1].bar(x, residuals, alpha=0.7, 
                      color=['red' if r < 0 else 'blue' for r in residuals])
        axes[1, 1].axhline(y=0, color='black', linestyle='--')
        axes[1, 1].set_xlabel('First Digit')
        axes[1, 1].set_ylabel('Residual (Observed - Benford)')
        axes[1, 1].set_title(f'{graph_name}: Residuals Analysis')
        axes[1, 1].set_xticks(x)
        
        plt.tight_layout()
        plt.show()
    
    # Network statistics
    stats = {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'avg_degree': np.mean(degrees),
        'max_degree': max(degrees),
        'degree_std': np.std(degrees),
        'chi_squared': chi_squared,
        'conforms_to_benford': chi_squared < 15.507,
        'observed_probs': observed_probs,
        'benford_probs': benford_probs
    }
    
    print(f"\n{graph_name} Network Analysis:")
    print(f"Nodes: {stats['nodes']}")
    print(f"Edges: {stats['edges']}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Maximum degree: {stats['max_degree']}")
    print(f"Degree standard deviation: {stats['degree_std']:.2f}")
    print(f"Chi-squared statistic: {stats['chi_squared']:.4f}")
    print(f"Conforms to Benford's Law: {'Yes' if stats['conforms_to_benford'] else 'No'}")
    
    return stats

# Generate and analyze different network types
def comprehensive_network_study():
    """Perform comprehensive network analysis study"""
    
    networks = {}
    
    # Scale-free network (Barabási-Albert)
    networks['Scale-Free'] = nx.barabasi_albert_graph(1000, 3, seed=42)
    
    # Random network (Erdős-Rényi)
    networks['Random'] = nx.erdos_renyi_graph(1000, 0.006, seed=42)
    
    # Small-world network (Watts-Strogatz)
    networks['Small-World'] = nx.watts_strogatz_graph(1000, 6, 0.3, seed=42)
    
    # Regular network
    networks['Regular'] = nx.random_regular_graph(4, 1000, seed=42)
    
    # Power-law cluster graph
    networks['Power-Law Cluster'] = nx.powerlaw_cluster_graph(1000, 3, 0.3, seed=42)
    
    # Analyze all networks
    results = {}
    for name, graph in networks.items():
        results[name] = analyze_network_benford_conformity(graph, name, detailed_analysis=False)
    
    # Summary comparison
    plt.figure(figsize=(12, 8))
    
    # Chi-squared comparison
    plt.subplot(2, 2, 1)
    names = list(results.keys())
    chi_values = [results[name]['chi_squared'] for name in names]
    colors = ['green' if chi < 15.507 else 'red' for chi in chi_values]
    plt.bar(names, chi_values, color=colors, alpha=0.7)
    plt.axhline(y=15.507, color='red', linestyle='--', label='Critical value (α=0.05)')
    plt.ylabel('Chi-squared statistic')
    plt.title('Benford\'s Law Conformity by Network Type')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Average degree comparison
    plt.subplot(2, 2, 2)
    avg_degrees = [results[name]['avg_degree'] for name in names]
    plt.bar(names, avg_degrees, alpha=0.7)
    plt.ylabel('Average Degree')
    plt.title('Average Degree by Network Type')
    plt.xticks(rotation=45)
    
    # Degree standard deviation
    plt.subplot(2, 2, 3)
    std_degrees = [results[name]['degree_std'] for name in names]
    plt.bar(names, std_degrees, alpha=0.7)
    plt.ylabel('Degree Standard Deviation')
    plt.title('Degree Variability by Network Type')
    plt.xticks(rotation=45)
    
    # First digit distribution heatmap
    plt.subplot(2, 2, 4)
    digit_matrix = np.array([results[name]['observed_probs'] for name in names])
    im = plt.imshow(digit_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Probability')
    plt.yticks(range(len(names)), names)
    plt.xticks(range(9), range(1, 10))
    plt.xlabel('First Digit')
    plt.ylabel('Network Type')
    plt.title('First Digit Distribution Heatmap')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run comprehensive network study
network_results = comprehensive_network_study()
```

## Integration Examples

### 1. Complete Analysis Pipeline

```python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('src')

class BenfordAnalysisPipeline:
    """Complete Benford's Law analysis pipeline"""
    
    def __init__(self, output_dir='analysis_results'):
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def analyze_dataset(self, data, dataset_name, description=""):
        """Analyze a dataset for Benford's Law conformity"""
        
        print(f"\nAnalyzing dataset: {dataset_name}")
        
        # Extract first digits
        if isinstance(data[0], str):
            # Handle string data (e.g., from CSV)
            numeric_data = [float(x) for x in data if self._is_numeric(x)]
        else:
            numeric_data = [x for x in data if x > 0 and not np.isnan(x)]
        
        first_digits = [int(str(abs(val))[0]) for val in numeric_data if val > 0]
        
        # Calculate distributions
        digit_counts = np.histogram(first_digits, bins=np.arange(0.5, 10.5, 1))[0]
        observed_probs = digit_counts / digit_counts.sum()
        
        benford_probs = np.log10(1 + 1 / np.arange(1, 10))
        benford_probs /= benford_probs.sum()
        
        # Statistical analysis
        chi_squared = np.sum((observed_probs - benford_probs) ** 2 / benford_probs)
        conforms = chi_squared < 15.507
        
        # Store results
        self.results[dataset_name] = {
            'data_size': len(numeric_data),
            'digit_counts': digit_counts,
            'observed_probs': observed_probs,
            'benford_probs': benford_probs,
            'chi_squared': chi_squared,
            'conforms': conforms,
            'description': description
        }
        
        # Create visualization
        self._create_analysis_plot(dataset_name)
        
        # Print summary
        print(f"  Sample size: {len(numeric_data)}")
        print(f"  Chi-squared: {chi_squared:.4f}")
        print(f"  Conforms to Benford's Law: {'Yes' if conforms else 'No'}")
        
        return self.results[dataset_name]
    
    def _is_numeric(self, value):
        """Check if a value can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _create_analysis_plot(self, dataset_name):
        """Create analysis plot for a dataset"""
        
        result = self.results[dataset_name]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Distribution comparison
        x = np.arange(1, 10)
        axes[0].bar(x - 0.2, result['observed_probs'], 0.4, 
                   label=f'{dataset_name}', alpha=0.7)
        axes[0].bar(x + 0.2, result['benford_probs'], 0.4, 
                   label="Benford's Law", alpha=0.7)
        axes[0].set_xlabel('First Digit')
        axes[0].set_ylabel('Probability')
        axes[0].set_title(f'{dataset_name}: Distribution Comparison')
        axes[0].legend()
        axes[0].set_xticks(x)
        
        # Residuals
        residuals = result['observed_probs'] - result['benford_probs']
        axes[1].bar(x, residuals, alpha=0.7, 
                   color=['red' if r < 0 else 'blue' for r in residuals])
        axes[1].axhline(y=0, color='black', linestyle='--')
        axes[1].set_xlabel('First Digit')
        axes[1].set_ylabel('Residual')
        axes[1].set_title('Residuals (Observed - Benford)')
        axes[1].set_xticks(x)
        
        # Chi-squared visualization
        chi_components = (result['observed_probs'] - result['benford_probs']) ** 2 / result['benford_probs']
        axes[2].bar(x, chi_components, alpha=0.7)
        axes[2].set_xlabel('First Digit')
        axes[2].set_ylabel('Chi-squared Component')
        axes[2].set_title(f'Chi-squared Components (Total: {result["chi_squared"]:.3f})')
        axes[2].set_xticks(x)
        
        plt.suptitle(f'{dataset_name} - Benford\'s Law Analysis\n'
                    f'χ² = {result["chi_squared"]:.4f}, '
                    f'Conforms: {"Yes" if result["conforms"] else "No"}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'{dataset_name}_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, filename='benford_analysis_report.html'):
        """Generate comprehensive HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benford's Law Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .dataset {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .conforms {{ color: green; font-weight: bold; }}
                .does-not-conform {{ color: red; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Benford's Law Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total datasets analyzed: {len(self.results)}</p>
            </div>
        """
        
        # Summary table
        html_content += """
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Sample Size</th>
                    <th>Chi-squared</th>
                    <th>Conforms to Benford's Law</th>
                    <th>Description</th>
                </tr>
        """
        
        for name, result in self.results.items():
            conform_class = "conforms" if result['conforms'] else "does-not-conform"
            html_content += f"""
                <tr>
                    <td>{name}</td>
                    <td>{result['data_size']}</td>
                    <td>{result['chi_squared']:.4f}</td>
                    <td class="{conform_class}">{'Yes' if result['conforms'] else 'No'}</td>
                    <td>{result['description']}</td>
                </tr>
            """
        
        html_content += "</table>"
        
        # Detailed results for each dataset
        for name, result in self.results.items():
            html_content += f"""
                <div class="dataset">
                    <h3>{name}</h3>
                    <p><strong>Description:</strong> {result['description']}</p>
                    <p><strong>Sample size:</strong> {result['data_size']}</p>
                    <p><strong>Chi-squared statistic:</strong> {result['chi_squared']:.6f}</p>
                    <p><strong>Critical value (α=0.05):</strong> 15.507</p>
                    <p><strong>Conforms to Benford's Law:</strong> 
                       <span class="{'conforms' if result['conforms'] else 'does-not-conform'}">
                       {'Yes' if result['conforms'] else 'No'}
                       </span>
                    </p>
                    
                    <h4>Digit Distribution</h4>
                    <table>
                        <tr>
                            <th>Digit</th>
                            <th>Observed Count</th>
                            <th>Observed Probability</th>
                            <th>Benford Probability</th>
                            <th>Difference</th>
                        </tr>
            """
            
            for i in range(9):
                digit = i + 1
                obs_prob = result['observed_probs'][i]
                ben_prob = result['benford_probs'][i]
                diff = obs_prob - ben_prob
                html_content += f"""
                    <tr>
                        <td>{digit}</td>
                        <td>{result['digit_counts'][i]}</td>
                        <td>{obs_prob:.4f}</td>
                        <td>{ben_prob:.4f}</td>
                        <td>{diff:+.4f}</td>
                    </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to: {report_path}")

# Example usage of the pipeline
def run_complete_analysis():
    """Run complete analysis pipeline with example datasets"""
    
    # Initialize pipeline
    pipeline = BenfordAnalysisPipeline()
    
    # Example datasets
    datasets = {
        'Fibonacci Numbers': {
            'data': [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765] * 50,
            'description': 'First 20 Fibonacci numbers replicated 50 times'
        },
        'Powers of 2': {
            'data': [2**i for i in range(1, 101)],
            'description': 'Powers of 2 from 2^1 to 2^100'
        },
        'Random Exponential': {
            'data': np.random.exponential(1, 10000),
            'description': 'Random samples from exponential distribution (λ=1)'
        },
        'Physical Constants': {
            'data': [299792458, 6.62607015e-34, 1.380649e-23, 1.602176634e-19, 9.1093837015e-31] * 2000,
            'description': 'Physical constants replicated for analysis'
        }
    }
    
    # Analyze each dataset
    for name, dataset in datasets.items():
        pipeline.analyze_dataset(dataset['data'], name, dataset['description'])
    
    # Generate comprehensive report
    pipeline.generate_report()
    
    return pipeline

# Run the complete analysis
# complete_pipeline = run_complete_analysis()
```

## Troubleshooting

### Common Issues and Solutions

1. **Import Errors**
   ```bash
   # Install missing packages
   pip install numpy matplotlib scipy pandas seaborn networkx geneticalgorithm
   
   # Verify installation
   python -c "import numpy, matplotlib, scipy, pandas, seaborn, networkx; print('All packages installed')"
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Process data in chunks
   def process_large_dataset(data, chunk_size=10000):
       results = []
       for i in range(0, len(data), chunk_size):
           chunk = data[i:i+chunk_size]
           # Process chunk
           results.append(analyze_chunk(chunk))
       return combine_results(results)
   ```

3. **Genetic Algorithm Convergence Problems**
   ```python
   # Adjust GA parameters for better convergence
   algorithm_params = {
       'max_num_iteration': 500,  # Increase iterations
       'population_size': 200,    # Larger population
       'mutation_probability': 0.2,  # Higher mutation rate
       'max_iteration_without_improv': 100  # More patience
   }
   ```

4. **Network Analysis Performance**
   ```python
   # For large networks, use sampling
   def analyze_large_network(graph, sample_size=1000):
       if graph.number_of_nodes() > sample_size:
           nodes = np.random.choice(list(graph.nodes()), sample_size, replace=False)
           subgraph = graph.subgraph(nodes)
           return analyze_network_benford_conformity(subgraph, f"Sample of {graph.number_of_nodes()} nodes")
       else:
           return analyze_network_benford_conformity(graph, "Full Network")
   ```

### Performance Tips

1. **Use NumPy vectorization** for large-scale computations
2. **Set random seeds** for reproducible results
3. **Monitor memory usage** with large datasets
4. **Use multiprocessing** for independent analyses
5. **Save intermediate results** for long computations

### Validation Checklist

- [ ] All required packages installed
- [ ] Input data properly formatted (numeric, positive values)
- [ ] Sufficient sample size (>100 observations recommended)
- [ ] Results make intuitive sense
- [ ] Statistical significance properly interpreted
- [ ] Visualizations clearly labeled and interpretable