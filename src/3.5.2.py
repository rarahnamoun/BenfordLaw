import networkx as nx
import numpy as np
from scipy.stats import entropy


def benford_law_digit_probability(n, d, sigma=1.0):
    probability = np.sum(np.log10(1 + 1 / (10 * k + d)) if (10 * k + d) != 0 else 0 for k in
                         range(int(10 ** (n - 2)), int(10 ** (n - 1))))
    return probability * 100


def observed_degree_distribution(graph):
    degrees = dict(graph.degree())
    histogram, bins = np.histogram(list(degrees.values()), bins=range(max(degrees.values()) + 2), density=True)

    # Ensure observed_probs has the same length as benford_probs
    observed_probs = np.interp(range(1, 11), bins[:-1], histogram)

    # Normalize to make sure the probability distribution sums to 1
    observed_probs /= sum(observed_probs)

    return observed_probs


def j_divergence(p, q):
    epsilon = 1e-10  # Small epsilon to avoid division by zero
    return np.sum(p * np.log((p + epsilon) / (q + epsilon)))


def jensen_shannon_divergence(p, q):
    m = (p + q) / 2
    kl_p = entropy(p, m)
    kl_q = entropy(q, m)
    return (kl_p + kl_q) / 2


def calculate_and_plot(graph_type, num_nodes, num_edges, seed=None):
    if graph_type == 'scale-free':
        graph = nx.barabasi_albert_graph(num_nodes, num_edges, seed=seed)
    elif graph_type == 'random':
        graph = nx.erdos_renyi_graph(num_nodes, 0.03, seed=seed)
    elif graph_type == 'small-world':
        graph = nx.watts_strogatz_graph(num_nodes, num_edges, p=0.3, seed=seed)

    benford_probs = [benford_law_digit_probability(2, d) for d in range(10)]
    benford_probs /= sum(benford_probs)

    observed_probs = observed_degree_distribution(graph)

    # Calculate Jensen-Shannon divergence
    js_divergence = jensen_shannon_divergence(benford_probs, observed_probs)

    # Calculate J-divergence (normalized)
    j_divergence_normalized = j_divergence(benford_probs, observed_probs) / np.log(2)

    print(f"{graph_type.capitalize()} Graph:")
    print(f"Number of Nodes: {len(graph.nodes)}")
    print(f"Number of Edges: {len(graph.edges)}")
    print(f"Jensen-Shannon Divergence: {js_divergence}")
    print(f"Normalized J-Divergence: {j_divergence_normalized}")


# Set seed for reproducibility
np.random.seed(42)

# Parameters for the graphs
num_nodes = 100
num_edges = 2

# Generate and analyze each type of graph
calculate_and_plot('scale-free', num_nodes, num_edges)
calculate_and_plot('random', num_nodes, num_edges)
calculate_and_plot('small-world', num_nodes, num_edges)
