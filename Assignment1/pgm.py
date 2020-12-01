import pickle
import numpy as np
import argparse
import networkx as nx
import itertools # part of Python standard library

def load_graph(filename):
    'load the graphical model (DO NOT MODIFY)'
    return pickle.load(open(filename, 'rb'))

# Helper functions for inference_brute_force--------------------------------------------

# # For a particular assignment of the vertex values, compute the unnormalized probability
def compute_unn_probability(G, assignment):

    # product of unary potentials for this assignment
    unary_pot = 1
    for v in G.nodes:
        v_value = assignment[v]
        pot = G.nodes[v]['unary_potential'][v_value]
        unary_pot = unary_pot * pot

    # product of binary potentials for this assignment
    binary_pot = 1
    for e in G.edges:
        u, v = e[0], e[1]
        u_value, v_value = assignment[u], assignment[v]

        pot = G.edges[e]['binary_potential'][u_value][v_value]
        binary_pot = binary_pot * pot

    return unary_pot * binary_pot

# Iterate over all possible assignments and compute marginal probabilities as well
# as MAP inference (brute force solution, exponential running time)
def compute_bf(G):
    all_values = list(range(G.graph['K']))
    all_assignments = list(itertools.product(all_values, repeat=len(G.nodes)))

    max_prob = 0
    max_assignment = np.zeros(len(G.nodes))
    for assignment in all_assignments:
        p = compute_unn_probability(G, assignment)

        for u in G.nodes:
            u_value = assignment[u]
            # update marginal probabilities
            G.nodes[u]['marginal_prob'][u_value] += p

            # update value of gradient binary potential with probabilities
            for v in G.neighbors(u):
                if(v < u): continue
                v_value = assignment[v]
                G.edges[(u, v)]['gradient_binary_potential'][u_value][v_value] += p

        # Update MAP inference
        if(p > max_prob):
            max_prob = p
            max_assignment = np.array(assignment)

    G.graph['v_map'] = max_assignment

    # Normalize marginal probabilities
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = G.nodes[v]['marginal_prob'] / sum(G.nodes[v]['marginal_prob'])

    # Normalize gradient binary potential (right now, only probabilities)
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = G.edges[e]['gradient_binary_potential'] / np.sum(G.edges[e]['gradient_binary_potential'])


# Used for both inference and inference_brute_force
def compute_unary_gradient(G):

    for v in G.nodes:
        assignment = G.nodes[v]['assignment']
        one_hot = np.zeros(G.graph['K'])
        one_hot[assignment] = 1

        G.nodes[v]['gradient_unary_potential'] = (one_hot - G.nodes[v]['marginal_prob']) / G.nodes[v]['unary_potential']

# Compute binary gradient potential using stored joint probabilities
def compute_binary_gradient_bf(G):
    for u in G.nodes:
        for v in G.neighbors(u):
            if(v < u): continue
            K = G.graph['K']
            one_hot = np.zeros((K, K))
            u_value, v_value = G.nodes[u]['assignment'], G.nodes[v]['assignment']
            one_hot[u_value][v_value] = 1

            e = (u, v)
            G.edges[e]['gradient_binary_potential'] = (one_hot - G.edges[e]['gradient_binary_potential'])  / G.edges[e]['binary_potential']

#-------------------------------------------------------------------------------------------

def inference_brute_force(G):
    '''
    Perform probabilistic inference on graph G, and compute the gradients of the log-likelihood
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        K: 
            G.graph['K']
        unary potentials: 
            G.nodes[v]['unary_potential'], a 1-d numpy array of length K
        binary potentials: 
            G.edges[u, v]['binary_potential'], a 2-d numpy array of shape K x K
        assignment for computing the gradients: 
            G.nodes[v]['assignment'], an integer within [0, K - 1]
    Output:
        G.nodes[v]['marginal_prob']: 
            the marginal probability distribution for v, a 1-d numpy array of length K
        G.graph['v_map']: 
            the MAP assignment, a 1-d numpy arrary of length n, where n is the number of vertices
        G.nodes[v]['gradient_unary_potential']: 
            the gradient of the log-likelihood w.r.t. the unary potential of vetext v, a 1-d numpy array of length K
        G.edges[u, v]['gradient_binary_potential']: 
            the gradient of the log-likelihood w.r.t. the binary potential of edge (u, v), a 2-d numpy array of shape K x K
    '''
    # initialize the output buffers
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = np.zeros(G.graph['K'])
        G.nodes[v]['gradient_unary_potential'] = np.zeros(G.graph['K'])
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = np.zeros((G.graph['K'], G.graph['K']))
    G.graph['v_map'] = np.zeros(len(G.nodes))
    
    compute_bf(G)
    compute_unary_gradient(G)
    compute_binary_gradient_bf(G)

# Helper functions for inference ----------------------------------------------------------

# Check whether the current node is a leaf node
def check_leaf(G, curr_node, curr_parent):
    for v in G.neighbors(curr_node):
        if(v != curr_parent): return False

    return True

# computes and stores all messages up the tree
def belief_prop_up(G, curr_node, curr_parent):

    if(check_leaf(G, curr_node, curr_parent)):
        return G.nodes[curr_node]['unary_potential']

    # iterate over all neighbors of node
    ans = G.nodes[curr_node]['unary_potential']
    for nbor in G.neighbors(curr_node):
        # skip parent
        if(nbor == curr_parent): continue

        # get matrix of binary potentials for edge (nbor, curr_node)
        e = (nbor, curr_node)
        binary_pot = G.edges[e]['binary_potential']

        # networkx quirk, matrix is ordered such that the smaller vertex values are on axis 0
        # we need the vertex values of the node higher in the tree to be on axis 0
        if(nbor < curr_node): binary_pot = binary_pot.T

        # compute message from nbor to curr_node and normalize
        G.edges[e]['message'] = (binary_pot @ belief_prop_up(G, nbor, curr_node))
        G.edges[e]['message'] = G.edges[e]['message'] / sum(G.edges[e]['message'])
        ans = ans * G.edges[e]['message']

    return ans


# downward pass, all messages up the tree have been calculated
def belief_prop_down(G, curr_node, curr_parent):

    for nbor in G.neighbors(curr_node):
        if(nbor == curr_parent): continue

        e = (nbor, curr_node)
        binary_pot = G.edges[e]['binary_potential']
        if(nbor > curr_node): binary_pot = binary_pot.T

        # initialize backward message
        G.edges[e]["bwd_message"] = G.nodes[curr_node]["unary_potential"]

        # compute product of messages to curr_node
        for nbor2 in G.neighbors(curr_node):
            if(nbor == nbor2): continue
            e2 = (curr_node, nbor2)
            if(nbor2 == curr_parent):
                G.edges[e]["bwd_message"] = G.edges[e]["bwd_message"] * G.edges[e2]["bwd_message"]

            else:
                G.edges[e]["bwd_message"] = G.edges[e]["bwd_message"] * G.edges[e2]["message"]

        G.edges[e]["bwd_message"] = (binary_pot @ G.edges[e]["bwd_message"])

        # normalize message
        G.edges[e]["bwd_message"] = G.edges[e]["bwd_message"] / sum(G.edges[e]["bwd_message"])

        belief_prop_down(G, nbor, curr_node)

# Compute marginal probabilities as product of messages from neighboring factors
def compute_mps(G, curr_node, curr_parent):
    G.nodes[curr_node]['marginal_prob'] = G.nodes[curr_node]['unary_potential']

    for nbor in G.neighbors(curr_node):
        e = (curr_node, nbor)
        if(nbor == curr_parent):
            msg = G.edges[e]['bwd_message']
        else:
            msg = G.edges[e]['message']

        G.nodes[curr_node]['marginal_prob'] = G.nodes[curr_node]['marginal_prob'] * msg

    for nbor in G.neighbors(curr_node):
        if(nbor != curr_parent):
            compute_mps(G, nbor, curr_node)

# compute messages up the tree for max-product algorithm
def max_prop_up(G, curr_node, curr_parent):

    # base case when we have reached leaf
    if(check_leaf(G, curr_node, curr_parent)):
        return G.nodes[curr_node]['unary_potential']

    ans = G.nodes[curr_node]['unary_potential']

    # compute message from each neighbor to the current node
    for nbor in G.neighbors(curr_node):
        if(nbor == curr_parent): continue

        e = (nbor, curr_node)
        binary_pot = G.edges[e]['binary_potential']

        if(nbor < curr_node): binary_pot = binary_pot.T

        prev_msgs = max_prop_up(G, nbor, curr_node)

        curr_msg = np.zeros(G.graph['K'])

        # assume nbor = u, curr_node = v, we are trying to compute m_(u->v)
        # iterate over all possible values of v and u
        for v_value in range(0, G.graph['K']):
            # fill in m_(u->v)[v]
            for u_value in range(0, G.graph['K']):
                new_msg = binary_pot[v_value][u_value]*prev_msgs[u_value]
                if(new_msg > curr_msg[v_value]):
                    curr_msg[v_value] = new_msg
                    G.edges[e]["curr_map"][v_value] = u_value

        # normalize message
        curr_msg = curr_msg / sum(curr_msg)
        ans = ans * curr_msg

    return ans

# Computes map inference by propagating down tree given map for root
def compute_map(G, curr_node, curr_parent):

    for nbor in G.neighbors(curr_node):
        if(nbor == curr_parent): continue
        curr_vmap = int(G.graph["v_map"][curr_node])
        e = (curr_node, nbor)
        G.graph['v_map'][nbor] = G.edges[e]["curr_map"][curr_vmap]
        compute_map(G, nbor, curr_node)

# Computes binary potential gradient for each edge using messages
def compute_binary_gradient(G, curr_node, curr_parent):

    K = G.graph['K']

    # compute joint probability between each node and neighbor
    for nbor in G.neighbors(curr_node):
        if(nbor == curr_parent): continue

        # initialize joint_prob to product of unary potentials and binary potential
        joint_prob = np.outer(G.nodes[curr_node]['unary_potential'], G.nodes[nbor]['unary_potential'])
        if(curr_node > nbor): joint_prob = joint_prob.T
        joint_prob = joint_prob * G.edges[(curr_node, nbor)]['binary_potential']

        # compute product of messages to curr_node excluding one from nbor
        msg_parent = np.ones(K)
        for nbor2 in G.neighbors(curr_node):
            if(nbor == nbor2): continue
            e = (curr_node, nbor2)
            if(nbor2 == curr_parent):
                msg_parent = msg_parent * G.edges[e]['bwd_message']
            else:
                msg_parent = msg_parent * G.edges[e]['message']

        # compute product of messages to nbor excluding one from curr_node
        msg_child = np.ones(K)
        for nbor2 in G.neighbors(nbor):
            if(nbor2 == curr_node): continue
            e = (nbor, nbor2)
            msg_child = msg_child * G.edges[e]['message']


        # compute outer product of messages
        msg_product = np.outer(msg_parent, msg_child)
        if(curr_node > nbor): msg_product = msg_product.T

        # update joint_prob with outer product of messages
        joint_prob = joint_prob * msg_product
        joint_prob = joint_prob / np.sum(joint_prob)

        # compute one-hot matrix
        one_hot = np.zeros((K, K))
        u_assign = G.nodes[curr_node]['assignment']
        v_assign = G.nodes[nbor]['assignment']
        one_hot[u_assign][v_assign] = 1
        if(curr_node > nbor): one_hot = one_hot.T

        # calculate gradient binary potential
        e = (curr_node, nbor)
        G.edges[e]['gradient_binary_potential'] = (one_hot - joint_prob) / G.edges[e]['binary_potential']

        compute_binary_gradient(G, nbor, curr_node)

#------------------------------------------------------------------------------------------------------

def inference(G):
    '''
    Perform probabilistic inference on graph G, and compute the gradients of the log-likelihood
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        K: 
            G.graph['K']
        unary potentials: 
            G.nodes[v]['unary_potential'], a 1-d numpy array of length K
        binary potentials: 
            G.edges[u, v]['binary_potential'], a 2-d numpy array of shape K x K
        assignment for computing the gradients: 
            G.nodes[v]['assignment'], an integer within [0, K - 1]
    Output:
        G.nodes[v]['marginal_prob']: 
            the marginal probability distribution for v, a 1-d numpy array of length K
        G.graph['v_map']: 
            the MAP assignment, a 1-d numpy arrary of length n, where n is the number of vertices
        G.nodes[v]['gradient_unary_potential']: 
            the gradient of the log-likelihood w.r.t. the unary potential of vetext v, a 1-d numpy array of length K
        G.edges[u, v]['gradient_binary_potential']: 
            the gradient of the log-likelihood w.r.t. the binary potential of edge (u, v), a 2-d numpy array of shape K x K
    '''
    # initialize the output buffers
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = np.zeros(G.graph['K'])
        G.nodes[v]['gradient_unary_potential'] = np.zeros(G.graph['K'])
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = np.zeros((G.graph['K'], G.graph['K']))
    G.graph['v_map'] = np.zeros(len(G.nodes))

    # store messages during upward pass
    for e in G.edges:
        G.edges[e]['fwd_message'] = np.zeros(G.graph['K'])

    # store backward messages during downward pass
    for e in G.edges:
        G.edges[e]['bwd_message'] = np.zeros(G.graph['K'])

    # compute marginal probabilities using the sum-product algorithm
    belief_prop_up(G, 0, -1)
    belief_prop_down(G, 0, -1)
    compute_mps(G, 0, -1)

    # normalize marginal probabilities
    for v in G.nodes:
        G.nodes[v]["marginal_prob"] = G.nodes[v]["marginal_prob"] / sum(G.nodes[v]["marginal_prob"])

    for e in G.edges:
        G.edges[e]["curr_map"] = np.zeros(G.graph["K"])

    # compute MAP inference using the max-product algorithm
    ans = max_prop_up(G, 0, -1)
    G.graph['v_map'][0] = np.argmax(ans)
    compute_map(G, 0, -1)

    compute_unary_gradient(G)
    compute_binary_gradient(G, 0, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='The input graph')
    args = parser.parse_args()
    G = load_graph(args.input)
    inference(G)

    # for v in G.nodes:
    #     print(G.nodes[v]["marginal_prob"])

    # print(G.graph["v_map"])

    # for v in G.nodes:
    #     print(G.nodes[v]["gradient_unary_potential"])

    # for e in G.edges:
    #     print(G.edges[e]["gradient_binary_potential"])

    pickle.dump(G, open('results_' + args.input, 'wb'))