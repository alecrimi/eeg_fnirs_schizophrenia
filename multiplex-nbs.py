#### Alessandro Crimi
### Multiplex NBS ###

import numpy as np
import networkx as nx
from scipy import stats
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def multiplex_nbs(case_networks, control_networks, threshold=3.0, n_permutations=1000, alpha=0.05):
    """
    Performs Network-Based Statistics on multiplex networks (3D approach).
    
    Parameters:
    -----------
    case_networks : dict
        Dictionary with structure {subject_id: [network_layer1, network_layer2]}
        Each network layer is a 2D numpy array representing connectivity matrix
    control_networks : dict
        Same structure as case_networks but for control group
    threshold : float
        t-statistic threshold for edge inclusion
    n_permutations : int
        Number of permutations for null distribution estimation
    alpha : float
        Significance level
        
    Returns:
    --------
    significant_components : list
        List of significant connected components
    component_pvalues : list
        p-values for each component
    max_component_sizes : list
        Maximum component sizes from permutation testing
    """
    # Check input data
    if len(case_networks) == 0 or len(control_networks) == 0:
        raise ValueError("Empty network dictionaries provided")
    
    # Check if all subjects have the same number of layers (should be 2 for this implementation)
    n_layers = len(next(iter(case_networks.values())))
    if not all(len(subj_nets) == n_layers for subj_nets in case_networks.values()) or \
       not all(len(subj_nets) == n_layers for subj_nets in control_networks.values()):
        raise ValueError("All subjects must have the same number of network layers")
    
    # Get dimensions from first subject
    first_case_subj = next(iter(case_networks.values()))
    n_nodes = first_case_subj[0].shape[0]
    
    # Check if all networks have the same dimensions
    for subj_nets in case_networks.values():
        for net in subj_nets:
            if net.shape != (n_nodes, n_nodes):
                raise ValueError(f"All networks must have the same dimensions ({n_nodes}x{n_nodes})")
    
    for subj_nets in control_networks.values():
        for net in subj_nets:
            if net.shape != (n_nodes, n_nodes):
                raise ValueError(f"All networks must have the same dimensions ({n_nodes}x{n_nodes})")
    
    # Calculate t-statistics for each edge in each layer
    case_matrices_layer1 = np.array([subj_nets[0] for subj_nets in case_networks.values()])
    case_matrices_layer2 = np.array([subj_nets[1] for subj_nets in case_networks.values()])
    
    control_matrices_layer1 = np.array([subj_nets[0] for subj_nets in control_networks.values()])
    control_matrices_layer2 = np.array([subj_nets[1] for subj_nets in control_networks.values()])
    
    # Calculate t-statistics matrix for each layer
    t_matrix_layer1 = np.zeros((n_nodes, n_nodes))
    t_matrix_layer2 = np.zeros((n_nodes, n_nodes))
    p_matrix_layer1 = np.ones((n_nodes, n_nodes))
    p_matrix_layer2 = np.ones((n_nodes, n_nodes))
    
    # Calculate t-statistics for each edge
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):  # Upper triangular only (undirected networks)
            # Layer 1
            t_stat, p_val = stats.ttest_ind(
                case_matrices_layer1[:, i, j],
                control_matrices_layer1[:, i, j],
                equal_var=False  # Welch's t-test
            )
            t_matrix_layer1[i, j] = t_stat
            t_matrix_layer1[j, i] = t_stat  # Mirror for undirected networks
            p_matrix_layer1[i, j] = p_val
            p_matrix_layer1[j, i] = p_val
            
            # Layer 2
            t_stat, p_val = stats.ttest_ind(
                case_matrices_layer2[:, i, j],
                control_matrices_layer2[:, i, j],
                equal_var=False  # Welch's t-test
            )
            t_matrix_layer2[i, j] = t_stat
            t_matrix_layer2[j, i] = t_stat  # Mirror for undirected networks
            p_matrix_layer2[i, j] = p_val
            p_matrix_layer2[j, i] = p_val
    
    # Create threshold adjacency matrices
    adj_matrix_layer1 = np.abs(t_matrix_layer1) > threshold
    adj_matrix_layer2 = np.abs(t_matrix_layer2) > threshold
    
    # Combine layers into a multiplex network representation
    # Create a multiplex graph
    G_multiplex = nx.Graph()
    
    # Add nodes for both layers (with layer identifiers)
    for i in range(n_nodes):
        G_multiplex.add_node((i, 0))  # Layer 1 nodes
        G_multiplex.add_node((i, 1))  # Layer 2 nodes
    
    # Add intralayer edges based on thresholded t-statistics
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adj_matrix_layer1[i, j]:
                G_multiplex.add_edge((i, 0), (j, 0), weight=abs(t_matrix_layer1[i, j]))
            if adj_matrix_layer2[i, j]:
                G_multiplex.add_edge((i, 1), (j, 1), weight=abs(t_matrix_layer2[i, j]))
    
    # Add interlayer edges (connecting the same node across layers)
    for i in range(n_nodes):
        G_multiplex.add_edge((i, 0), (i, 1), weight=1.0)  # Fixed weight for interlayer connections
    
    # Find connected components in the multiplex network
    components = list(nx.connected_components(G_multiplex))
    component_sizes = [len(comp) for comp in components]
    
    # If no components found above threshold, return empty results
    if not components:
        return [], [], []
    
    # Compute the observed component sizes
    observed_sizes = component_sizes
    
    # Permutation testing to establish null distribution
    max_component_sizes = []
    
    def single_permutation(perm_id):
        # Combine all subjects
        all_subjects_layer1 = np.vstack([case_matrices_layer1, control_matrices_layer1])
        all_subjects_layer2 = np.vstack([case_matrices_layer2, control_matrices_layer2])
        
        # Number of subjects in each group
        n_case = len(case_networks)
        n_control = len(control_networks)
        n_total = n_case + n_control
        
        # Random permutation of subjects
        perm_indices = np.random.permutation(n_total)
        perm_case_indices = perm_indices[:n_case]
        perm_control_indices = perm_indices[n_case:]
        
        # Create permuted groups
        perm_case_layer1 = all_subjects_layer1[perm_case_indices]
        perm_control_layer1 = all_subjects_layer1[perm_control_indices]
        perm_case_layer2 = all_subjects_layer2[perm_case_indices]
        perm_control_layer2 = all_subjects_layer2[perm_control_indices]
        
        # Calculate t-statistics for permuted data
        perm_t_matrix_layer1 = np.zeros((n_nodes, n_nodes))
        perm_t_matrix_layer2 = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                # Layer 1
                t_stat, _ = stats.ttest_ind(
                    perm_case_layer1[:, i, j],
                    perm_control_layer1[:, i, j],
                    equal_var=False
                )
                perm_t_matrix_layer1[i, j] = t_stat
                perm_t_matrix_layer1[j, i] = t_stat
                
                # Layer 2
                t_stat, _ = stats.ttest_ind(
                    perm_case_layer2[:, i, j],
                    perm_control_layer2[:, i, j],
                    equal_var=False
                )
                perm_t_matrix_layer2[i, j] = t_stat
                perm_t_matrix_layer2[j, i] = t_stat
        
        # Threshold permuted t-statistic matrices
        perm_adj_matrix_layer1 = np.abs(perm_t_matrix_layer1) > threshold
        perm_adj_matrix_layer2 = np.abs(perm_t_matrix_layer2) > threshold
        
        # Create multiplex graph for permuted data
        G_perm_multiplex = nx.Graph()
        
        # Add nodes for both layers
        for i in range(n_nodes):
            G_perm_multiplex.add_node((i, 0))
            G_perm_multiplex.add_node((i, 1))
        
        # Add intralayer edges
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if perm_adj_matrix_layer1[i, j]:
                    G_perm_multiplex.add_edge((i, 0), (j, 0), weight=abs(perm_t_matrix_layer1[i, j]))
                if perm_adj_matrix_layer2[i, j]:
                    G_perm_multiplex.add_edge((i, 1), (j, 1), weight=abs(perm_t_matrix_layer2[i, j]))
        
        # Add interlayer edges
        for i in range(n_nodes):
            G_perm_multiplex.add_edge((i, 0), (i, 1), weight=1.0)
        
        # Find connected components in permuted multiplex network
        perm_components = list(nx.connected_components(G_perm_multiplex))
        
        if not perm_components:
            return 0
        
        # Return the maximum component size
        return max(len(comp) for comp in perm_components)
    
    # Perform permutations in parallel
    max_component_sizes = Parallel(n_jobs=-1)(
        delayed(single_permutation)(i) for i in range(n_permutations)
    )
    
    # Calculate p-values for each observed component
    component_pvalues = []
    for size in observed_sizes:
        # Count how many permutations had a component at least as large as the observed
        p_value = sum(1 for max_size in max_component_sizes if max_size >= size) / n_permutations
        component_pvalues.append(p_value)
    
    # Identify significant components
    significant_components = [
        components[i] for i in range(len(components)) 
        if component_pvalues[i] < alpha
    ]
    significant_pvalues = [
        p for p in component_pvalues 
        if p < alpha
    ]
    
    return significant_components, significant_pvalues, max_component_sizes

def visualize_multiplex_components(G_multiplex, significant_components, filename=None):
    """
    Visualize the significant components in the multiplex network.
    
    Parameters:
    -----------
    G_multiplex : networkx.Graph
        The full multiplex network
    significant_components : list
        List of significant components to highlight
    filename : str, optional
        If provided, saves the figure to this file
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract node positions
    pos = {}
    n_nodes = len(set(n[0] for n in G_multiplex.nodes()))
    
    # Generate positions for the nodes
    for i in range(n_nodes):
        # Layer 1 nodes are positioned on the z=0 plane
        pos[(i, 0)] = (np.cos(2*np.pi*i/n_nodes), np.sin(2*np.pi*i/n_nodes), 0)
        # Layer 2 nodes are positioned on the z=1 plane
        pos[(i, 1)] = (np.cos(2*np.pi*i/n_nodes), np.sin(2*np.pi*i/n_nodes), 1)
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    
    # Determine if each node is in a significant component
    for node in G_multiplex.nodes():
        in_significant = False
        for comp in significant_components:
            if node in comp:
                in_significant = True
                break
        
        if in_significant:
            node_colors.append('red')
            node_sizes.append(100)
        else:
            node_colors.append('blue')
            node_sizes.append(50)
    
    # Draw nodes
    x = [pos[node][0] for node in G_multiplex.nodes()]
    y = [pos[node][1] for node in G_multiplex.nodes()]
    z = [pos[node][2] for node in G_multiplex.nodes()]
    
    ax.scatter(x, y, z, c=node_colors, s=node_sizes, alpha=0.7)
    
    # Draw edges
    for u, v in G_multiplex.edges():
        # Check if this edge is part of a significant component
        is_significant = False
        for comp in significant_components:
            if u in comp and v in comp:
                is_significant = True
                break
        
        # Draw the edge with appropriate color and width
        if is_significant:
            color = 'red'
            linewidth = 2.0
            alpha = 0.8
        else:
            color = 'gray'
            linewidth = 0.5
            alpha = 0.2
        
        # Extract coordinates
        x_coords = [pos[u][0], pos[v][0]]
        y_coords = [pos[u][1], pos[v][1]]
        z_coords = [pos[u][2], pos[v][2]]
        
        ax.plot(x_coords, y_coords, z_coords, color=color, linewidth=linewidth, alpha=alpha)
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multiplex Network Significant Components')
    
    # Adjust the viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_example_data(n_case=20, n_control=20, n_nodes=64, sparsity=0.1, effect_size=0.3):
    """
    Create example multiplex network data for testing the algorithm.
    
    Parameters:
    -----------
    n_case : int
        Number of case subjects
    n_control : int
        Number of control subjects
    n_nodes : int
        Number of nodes in each network
    sparsity : float
        Edge density in networks (0-1)
    effect_size : float
        Size of difference between groups
        
    Returns:
    --------
    case_networks : dict
        Dictionary of case networks
    control_networks : dict
        Dictionary of control networks
    """
    np.random.seed(42)  # For reproducibility
    
    # Create baseline connectivity pattern (common to both groups)
    base_conn = np.random.rand(n_nodes, n_nodes)
    base_conn = (base_conn + base_conn.T) / 2  # Make it symmetric
    np.fill_diagonal(base_conn, 0)  # No self-connections
    
    # Threshold to achieve desired sparsity
    threshold = np.percentile(base_conn.flatten(), 100 * (1 - sparsity))
    base_conn[base_conn < threshold] = 0
    
    # Create networks for control group
    control_networks = {}
    for i in range(n_control):
        # Add subject-specific noise
        noise_layer1 = np.random.normal(0, 0.1, (n_nodes, n_nodes))
        noise_layer1 = (noise_layer1 + noise_layer1.T) / 2  # Make symmetric
        np.fill_diagonal(noise_layer1, 0)
        
        noise_layer2 = np.random.normal(0, 0.1, (n_nodes, n_nodes))
        noise_layer2 = (noise_layer2 + noise_layer2.T) / 2  # Make symmetric
        np.fill_diagonal(noise_layer2, 0)
        
        # Create network layers
        network_layer1 = base_conn + noise_layer1
        network_layer2 = base_conn * 0.8 + noise_layer2  # Second layer is slightly different
        
        # Ensure networks are non-negative
        network_layer1[network_layer1 < 0] = 0
        network_layer2[network_layer2 < 0] = 0
        
        control_networks[f"control_{i}"] = [network_layer1, network_layer2]
    
    # Create effect pattern (for the case group)
    # This creates a connected component that spans both layers
    effect_pattern = np.zeros((n_nodes, n_nodes))
    
    # Create a connected community
    community_size = 8
    community_nodes = np.random.choice(n_nodes, community_size, replace=False)
    
    for i in range(community_size):
        for j in range(i+1, community_size):
            effect_pattern[community_nodes[i], community_nodes[j]] = effect_size
            effect_pattern[community_nodes[j], community_nodes[i]] = effect_size
    
    # Create networks for case group
    case_networks = {}
    for i in range(n_case):
        # Add subject-specific noise
        noise_layer1 = np.random.normal(0, 0.1, (n_nodes, n_nodes))
        noise_layer1 = (noise_layer1 + noise_layer1.T) / 2  # Make symmetric
        np.fill_diagonal(noise_layer1, 0)
        
        noise_layer2 = np.random.normal(0, 0.1, (n_nodes, n_nodes))
        noise_layer2 = (noise_layer2 + noise_layer2.T) / 2  # Make symmetric
        np.fill_diagonal(noise_layer2, 0)
        
        # Create network layers with effect
        # Effect is stronger in layer 1 than layer 2
        network_layer1 = base_conn + effect_pattern + noise_layer1
        network_layer2 = base_conn * 0.8 + effect_pattern * 0.7 + noise_layer2
        
        # Ensure networks are non-negative
        network_layer1[network_layer1 < 0] = 0
        network_layer2[network_layer2 < 0] = 0
        
        case_networks[f"case_{i}"] = [network_layer1, network_layer2]
    
    return case_networks, control_networks, community_nodes

def main():
    """
    Example usage of the multiplex NBS algorithm
    This can be substituted with real fMRI, EEG, fNIRS data
    """
    print("Generating example multiplex network data...")
    case_networks, control_networks, true_community = create_example_data(
        n_case=20, 
        n_control=20, 
        n_nodes=64, 
        sparsity=0.1, 
        effect_size=0.3
    )
    
    print(f"Generated data: {len(case_networks)} case networks and {len(control_networks)} control networks")
    print(f"Each network has 2 layers of size {case_networks['case_0'][0].shape}")
    print(f"True community nodes: {true_community}")
    
    print("\nPerforming multiplex NBS analysis...")
    significant_components, pvalues, max_component_sizes = multiplex_nbs(
        case_networks, 
        control_networks, 
        threshold=2.5,  # t-statistic threshold
        n_permutations=1000,
        alpha=0.05
    )
    
    print(f"\nFound {len(significant_components)} significant components:")
    for i, (comp, pval) in enumerate(zip(significant_components, pvalues)):
        layer1_nodes = sorted([n[0] for n in comp if n[1] == 0])
        layer2_nodes = sorted([n[0] for n in comp if n[1] == 1])
        overlap_nodes = sorted(set(layer1_nodes).intersection(set(layer2_nodes)))
        
        print(f"Component {i+1} (p={pval:.4f}):")
        print(f"  Size: {len(comp)} nodes")
        print(f"  Layer 1 nodes: {len(layer1_nodes)}")
        print(f"  Layer 2 nodes: {len(layer2_nodes)}")
        print(f"  Nodes present in both layers: {len(overlap_nodes)}")
        
        # Check overlap with true community
        layer1_overlap = set(layer1_nodes).intersection(set(true_community))
        layer2_overlap = set(layer2_nodes).intersection(set(true_community))
        print(f"  True positives in layer 1: {len(layer1_overlap)}/{len(true_community)}")
        print(f"  True positives in layer 2: {len(layer2_overlap)}/{len(true_community)}")
    
    # If no significant components found
    if not significant_components:
        print("No significant components found.")
    
    # Plot null distribution
    plt.figure(figsize=(10, 6))
    plt.hist(max_component_sizes, bins=30, alpha=0.7)
    
    # Plot observed component sizes
    observed_sizes = [len(comp) for comp in significant_components]
    for size in observed_sizes:
        plt.axvline(x=size, color='red', linestyle='--')
    
    plt.xlabel('Maximum Component Size')
    plt.ylabel('Frequency')
    plt.title('Null Distribution of Maximum Component Size')
    plt.tight_layout()
    plt.show()
    
    # Create and visualize multiplex network with significant components
    if significant_components:
        # Create the full multiplex network
        n_nodes = case_networks['case_0'][0].shape[0]
        G_multiplex = nx.Graph()
        
        # Add nodes for both layers
        for i in range(n_nodes):
            G_multiplex.add_node((i, 0))  # Layer 1 nodes
            G_multiplex.add_node((i, 1))  # Layer 2 nodes
        
        # Add interlayer edges (connecting same node across layers)
        for i in range(n_nodes):
            G_multiplex.add_edge((i, 0), (i, 1))
        
        # Add all possible edges (will be drawn with different styles based on significance)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                G_multiplex.add_edge((i, 0), (j, 0))
                G_multiplex.add_edge((i, 1), (j, 1))
        
        visualize_multiplex_components(G_multiplex, significant_components)

if __name__ == "__main__":
    main()
