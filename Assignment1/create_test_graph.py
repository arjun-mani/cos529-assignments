import networkx as nx
import numpy as np

G = nx.random_powerlaw_tree(2)

G.graph['K'] = 2

G.nodes[0]["unary_potential"] = np.array([2, 2])
G.nodes[1]["unary_potential"] = np.array([1, 5])
G.edges[(0, 1)]["binary_potential"] = np.array([[4, 1], [1, 3]])

G.nodes[0]['assignment'] = 0
G.nodes[1]['assignment'] = 1

nx.write_gpickle(G, "./test_graph.pickle")