from nilearn import datasets 
import pdb
import numpy as np
import os
import matplotlib.pyplot as plt
from nilearn import plotting
import networkx as nx


atlas = datasets.fetch_atlas_aal()
labels = atlas.labels  # List of AAL region labels

top_rois = np.loadtxt('sorted_top_rois_116_step20.csv', delimiter=',')[0:100]
# Generate weights
num_connections = len(top_rois)
weights = np.linspace(1, 0.1, num_connections)  # Example: Linear decrease

G = nx.Graph()

# Add nodes (brain regions)
for label in labels:
    G.add_node(label) 

# Add edges with weights
for i, roi_pair in enumerate(top_rois):
    roi1_index = int(roi_pair[0])
    roi2_index = int(roi_pair[1])
    roi1_name = labels[roi1_index]
    roi2_name = labels[roi2_index]
    weight = weights[i]
    G.add_edge(roi1_name, roi2_name, weight=weight) 

# Customize colors etc. if desired
node_color = 'grey' 

# Get node positions from the AAL atlas 
coordinates = plotting.find_parcellation_cut_coords(atlas.maps) 

adjacency_matrix = nx.adjacency_matrix(G).todense() 

# Draw graph with AAL background
plotting.plot_connectome(adjacency_matrix, coordinates, node_color=node_color,
                         edge_threshold='1%', title='Top 10 Connections') 

plt.show()
