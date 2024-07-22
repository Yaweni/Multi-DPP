from itertools import combinations, groupby
import random
import networkx as nx
import numpy as np

def find_furthest_nodes(G, start_node):
    """
    Finds the nodes that are furthest away from the given start node in the graph G.

    Parameters:
    G (networkx.Graph): The input graph.
    start_node (any): The node to start from.

    Returns:
    list: A list of nodes that are furthest away from the start node.
    """
    # Calculate the shortest path lengths from the start node to all other nodes
    path_lengths = dict(nx.shortest_path_length(G, source=start_node))

    # Find the maximum path length
    max_path_length = max(path_lengths.values())

    # Find the nodes that have the maximum path length
    furthest_nodes = [node for node, length in path_lengths.items() if length == max_path_length]

    return furthest_nodes

def gnp_random_connected_graph(n, p,targets):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(10, 100)
    G.graph['Targets']=random.sample(list(G.nodes),targets)
    G.graph['Target Probabilities']={}
    return G



def dual_graphs(n1,n2,t1,t2,p1,p2):

    g1=gnp_random_connected_graph(n1,p1,t1)
    g2=gnp_random_connected_graph(n2,p2,t2)
    for target in g1.graph['Targets']:
        cors = len(g2.graph['Targets'])
        corresponding_targets=g2.graph['Targets']
        #corresponding_targets=random.sample(g2.graph['Targets'],cors)
        distr = [1]*cors
        distr = np.random.dirichlet((tuple(distr)))
        buff_dict={}
        for prob,corresponding_target in zip(distr,corresponding_targets):
            buff_dict.setdefault(corresponding_target,prob)
        g1.graph['Target Probabilities'][target] = buff_dict
    run = 0
    for targets2 in g2.graph['Targets']:
        buff_dict = {}
        for targets1 in g1.graph['Targets']:

            reverse_probs = g1.graph['Target Probabilities'][targets1]
            d=reverse_probs.get(targets2,0)

            if (d > 0):
                buff_dict.setdefault(targets1,d)
        total = sum(buff_dict.values())
        #print('Buff dict Target',targets2,' is ',buff_dict)
        buff_dict2={k: v / total for k, v in buff_dict.items()}
            #print(buff_dict2)
        g2.graph['Target Probabilities'][targets2] = buff_dict2
            #print(g1.graph['Target Probabilities'][targets2])
    return g1,g2





g1,g2=dual_graphs(40,30,5,7,0.3,0.26)
