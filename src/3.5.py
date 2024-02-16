import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def benford_pmf(x):
    return np.log10(1 + 1 / x)


def plot_degree_and_benford_histogram(graph, graph_type):
    degrees = dict(graph.degree())
    histogram, bins = np.histogram(list(degrees.values()), bins=range(max(degrees.values()) + 2), density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers, histogram, width=0.4, alpha=0.7, label=f'{graph_type} Actual')

    degree_counts = list(degrees.values())
    benford_probs = [benford_pmf(x) for x in range(1, len(degree_counts) + 1)]
    benford_probs /= sum(benford_probs)

    # Rescale benford_probs to match the length of histogram
    benford_probs_rescaled = np.interp(bin_centers, np.arange(1, len(degree_counts) + 1) + 0.5, benford_probs)

    plt.bar(bin_centers, benford_probs_rescaled, width=0.4, alpha=0.7, label="Benford's Law")

    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title(f'Degree Histogram of {graph_type} Graph and Benford\'s Law')
    plt.legend()

    # Calculate the difference between the actual degree and Benford's Law histograms
    differences = histogram - benford_probs_rescaled
    return differences


def calculate_and_plot(graph_type, num_nodes, num_edges, seed=None):
    if graph_type == 'scale-free':
        graph = nx.barabasi_albert_graph(num_nodes, num_edges, seed=seed)
    elif graph_type == 'random':
        graph = nx.erdos_renyi_graph(num_nodes, 0.03, seed=seed)
    elif graph_type == 'small-world':
        graph = nx.watts_strogatz_graph(num_nodes, num_edges, p=0.3, seed=seed)

    plt.figure(figsize=(10, 6))

    differences = plot_degree_and_benford_histogram(graph, graph_type)

    print(f"{graph_type.capitalize()} Graph:")
    print(f"Number of Nodes: {len(graph.nodes)}")
    print(f"Number of Edges: {len(graph.edges)}")

    # Print the sum of differences
    print(f"Sum of Differences: {sum(differences)}")

    plt.show()


# Set seed for reproducibility
np.random.seed(42)

# Parameters for the graphs
num_nodes = 100
num_edges = 2

# Generate and analyze each type of graph
calculate_and_plot('scale-free', num_nodes, num_edges)
calculate_and_plot('random', num_nodes, num_edges)
calculate_and_plot('small-world', num_nodes, num_edges)
