import networkx as nx
import numpy as np

G = nx.random_tree(200)

K = 5
G.graph['K'] = K

for node in G.nodes:
	G.nodes[node]['unary_potential'] = np.full(K, 5)
	G.nodes[node]['assignment'] = 1

for edge in G.edges:
	G.edges[edge]['binary_potential'] = np.full((K, K), 5)

nx.write_gpickle(G, "./graph1000.pickle")
