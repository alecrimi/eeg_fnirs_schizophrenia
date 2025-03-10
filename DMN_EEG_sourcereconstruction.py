import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity
from nilearn import plotting
import os.path as op
import pandas as pd

# Step 1: Load your preprocessed EEG data
# Replace with your actual file path
raw_fname = 'your_preprocessed_eeg.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# Step 2: Define epochs around your events of interest (for resting state, 
# you might use continuous segments or create artificial events)
# For resting state without specific events:
events = mne.make_fixed_length_events(raw, duration=2.0)
epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)

# Step 3: Set up the source space with a template MRI
# For individual MRI use: subject = 'subject_id'
subject = 'fsaverage'
subjects_dir = mne.datasets.sample.data_path() / 'subjects'

# Create source space with 'oct6' spacing (or use 'oct5' for faster computation)
src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir)

# Step 4: Create the BEM model and solution
model = mne.make_bem_model(subject=subject, ico=4, subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# Step 5: Create the forward solution
trans = 'fsaverage'  # For template MRI. For individual MRI use path to '-trans.fif' file
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                meg=False, eeg=True, mindist=5.0)

# Step 6: Compute the noise covariance matrix from baseline or empty room data
# For resting state, you might use a small portion of your data
noise_cov = mne.compute_raw_covariance(raw, method='shrunk')

# Step 7: Make the inverse operator
inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)

# Step 8: Apply the inverse operator to get source time courses
# lambda2 is the regularization parameter (1/SNR²)
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # Options: MNE, dSPM, sLORETA, eLORETA
stc = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                          pick_ori="normal", return_generator=False)

# Step 9: Define DMN regions
# These are MNI coordinates for key DMN regions 
dmn_regions = {
    'mPFC': [0, 52, -6],       # medial prefrontal cortex
    'PCC': [0, -52, 26],       # posterior cingulate cortex
    'left_IPL': [-48, -68, 28], # left inferior parietal lobule
    'right_IPL': [48, -68, 28], # right inferior parietal lobule
    'left_MTL': [-28, -36, -12], # left medial temporal lobe
    'right_MTL': [28, -36, -12]  # right medial temporal lobe
}

# Step 10: Extract time series from DMN regions
# Convert MNI coordinates to source space vertices
# First, we need to get the source space we used
src = inverse_operator['src']

# Function to find nearest vertex to a given MNI coordinate
def find_nearest_vertex(mni_coord, subject='fsaverage', subjects_dir=None):
    """Find nearest vertex to an MNI coordinate"""
    # Transform MNI coordinate to the subject's RAS space
    point = mne.transforms.apply_trans(
        mne.transforms.invert_transform(
            mne.vertex_to_mni(np.array([0]), np.array([0]), 
                             subject=subject, subjects_dir=subjects_dir)[1]),
        np.array([mni_coord]))
    
    # Find the closest source space vertex
    distances = np.sum((src[0]['rr'] - point) ** 2, axis=1)
    nearest_vertex = np.argmin(distances)
    return nearest_vertex, src[0]['vertno'][nearest_vertex]

# Extract vertex indices for DMN regions
dmn_vertices = {}
for region_name, mni_coord in dmn_regions.items():
    src_idx, vertex_idx = find_nearest_vertex(mni_coord, subject=subject, subjects_dir=subjects_dir)
    dmn_vertices[region_name] = vertex_idx

# Create labels (regions of interest) for the DMN regions
dmn_labels = {}
for region_name, vertex_idx in dmn_vertices.items():
    # Create a 10mm radius label around each vertex
    dmn_labels[region_name] = mne.Label(
        vertices=[vertex_idx],
        pos=None,
        hemi='lh' if 'left' in region_name or region_name in ['mPFC', 'PCC'] else 'rh',
        name=region_name,
        subject=subject,
        verbose=None
    )

# Step 11: Extract time series from each DMN region
dmn_ts = {}
for region_name, label in dmn_labels.items():
    # Extract time series from each epoch
    ts_list = []
    for epoch_idx, stc_epoch in enumerate(stc):
        # Extract time courses for this label
        label_tc = stc_epoch.extract_label_time_course(label, src, mode='mean')
        ts_list.append(label_tc[0])  # Get the first (and only) time course
    
    # Store the time series for all epochs
    dmn_ts[region_name] = np.array(ts_list)

# Step 12: Compute connectivity between DMN regions using PLV
# For PLV, we need to compute phase time series
from scipy.signal import hilbert
from scipy.stats import circmean
import numpy as np

# Function to compute PLV between two time series
def compute_plv(ts1, ts2):
    # Apply Hilbert transform to get analytic signal
    analytic_signal1 = hilbert(ts1)
    analytic_signal2 = hilbert(ts2)
    
    # Extract phase information
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)
    
    # Compute phase difference
    phase_diff = phase1 - phase2
    
    # Compute PLV as the absolute of the mean of the complex exponential of the phase difference
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv

# Compute PLV between all pairs of DMN regions
region_names = list(dmn_ts.keys())
n_regions = len(region_names)
plv_matrix = np.zeros((n_regions, n_regions))

for i in range(n_regions):
    for j in range(n_regions):
        # Compute average PLV across epochs
        plvs = []
        for epoch in range(dmn_ts[region_names[i]].shape[0]):
            plv = compute_plv(dmn_ts[region_names[i]][epoch], dmn_ts[region_names[j]][epoch])
            plvs.append(plv)
        plv_matrix[i, j] = np.mean(plvs)

# Step 13: Visualize the DMN connectivity matrix
plt.figure(figsize=(10, 8))
plt.imshow(plv_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='PLV')
plt.xticks(np.arange(n_regions), region_names, rotation=45)
plt.yticks(np.arange(n_regions), region_names)
plt.title('DMN Connectivity Matrix (PLV)')
plt.tight_layout()
plt.savefig('dmn_connectivity_matrix.png')

# Step 14: Compute network metrics for the DMN network
# (This requires the networkx library)
import networkx as nx

# Create a graph from the PLV matrix
G = nx.from_numpy_array(plv_matrix)
# Relabel nodes with region names
mapping = {i: region_name for i, region_name in enumerate(region_names)}
G = nx.relabel_nodes(G, mapping)

# Compute clustering coefficient
clustering = nx.clustering(G)
print("Clustering coefficients:")
for node, coeff in clustering.items():
    print(f"{node}: {coeff:.3f}")

# Compute other network metrics
print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")
print(f"Transitivity: {nx.transitivity(G):.3f}")
print(f"Density: {nx.density(G):.3f}")

# Step 15: Visualize the DMN network
plt.figure(figsize=(10, 8))
nx.draw_networkx(G, pos=nx.spring_layout(G), 
                node_color='lightblue', 
                node_size=500, 
                font_size=10,
                width=[G[u][v]['weight'] * 5 for u, v in G.edges()],
                with_labels=True)
plt.title('DMN Network Graph')
plt.axis('off')
plt.tight_layout()
plt.savefig('dmn_network_graph.png')

# Step 16: Save results
np.save('dmn_plv_matrix.npy', plv_matrix)
# Save the region names for reference
with open('dmn_regions.txt', 'w') as f:
    for region in region_names:
        f.write(f"{region}\n")

print("DMN analysis complete. Results saved.")
