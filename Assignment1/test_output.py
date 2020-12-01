import pickle
import numpy as np
import argparse
import networkx as nx

def load_graph(filename):
    'load the graphical model (DO NOT MODIFY)'
    return pickle.load(open(filename, 'rb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='The input graph')
    args = parser.parse_args()
    G = load_graph(args.input)
    for v in G.nodes:
    	print("Vertex {}".format(v))
    
    print(G.edges[(0, 1)]['binary_potential'])
    print(G.edges[(1, 0)]['binary_potential'])