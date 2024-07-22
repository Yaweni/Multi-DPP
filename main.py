import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
import random

def gnp_random_connected_target_graph(n, p,targets):
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
        G[u][v]['weight'] = random.randint(1, 3)
    G.graph['Targets']=random.sample(G.nodes,targets)
    return G

# Create a directed graph
G = gnp_random_connected_graph(80,0.2)


# Define target nodes and their colors
target_nodes = {1: 'red', 8: 'blue', 5: 'green'}

# Calculate shortest paths and costs to target nodes
shortest_paths = {target: nx.single_source_dijkstra_path_length(G, target) for target in target_nodes}

# Assign probabilities and colors to nodes
node_colors = {}
for node in G.nodes:
    if node in target_nodes:
        node_colors[node] = target_nodes[node]
    else:
        costs = {target: shortest_paths[target].get(node, float('inf')) for target in target_nodes}
        total_cost = sum(costs.values())
        probabilities = {target: (total_cost - cost) / total_cost for target, cost in costs.items()}
        max_prob_target = max(probabilities, key=probabilities.get)
        node_colors[node] = target_nodes[max_prob_target]

def draw_half_colored_nodes(G, pos, node_colors):
    for node, color in node_colors.items():
        x, y = pos[node]
        circle = plt.Circle((x, y), radius=0.05, color=color, zorder=2)
        plt.gca().add_patch(circle)
        if node not in target_nodes:
            half_circle = plt.Circle((x, y), radius=0.05, color='white', zorder=3)
            plt.gca().add_patch(half_circle)
            plt.plot([x, x], [y-0.05, y+0.05], color=color, linewidth=2, zorder=4)


# Draw the graph
pos = nx.spring_layout(G)
colors = [node_colors[node] for node in G.nodes]
#nx.draw(G, pos, with_labels=True, node_color=colors, edge_color='black', node_size=700, font_size=10, font_color='white')
nx.draw(G, pos, with_labels=True, edge_color='black', node_size=700, font_size=10, font_color='white')
draw_half_colored_nodes(G, pos, node_colors)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
#--------------------------------------------------------------------------------------------------


# Function 2: Deceptive path planning strategy 2
def deceptive_path_strategy_2(G, entry, target, target_nodes, probabilities):
# Eliminate nodes where target node is the maximum probability
subgraph = G.copy()
for node in list(subgraph.nodes):
if node != entry and node != target and max(probabilities[node], key=probabilities[node].get) == target:
subgraph.remove_node(node)

# Restore the LDP node
ldp = None
for node in G.nodes:
if node in subgraph.nodes and node != target and any(neighbor not in target_nodes[target] for neighbor in G.neighbors(node)):
ldp = node
break

# Select nodes by least probability of the real target in the next graph
ranked_nodes = sorted(subgraph.nodes, key=lambda x: sum(probabilities[x][t] * probabilities[t][target] for t in target_nodes))

# Include nodes until there is a path from entry to LDP
for node in ranked_nodes:
if node != entry and node != ldp and nx.has_path(subgraph, entry, ldp):
subgraph.remove_node(node)

# Calculate the shortest path from entry to LDP
path = nx.shortest_path(subgraph, source=entry, target=ldp, weight='weight')
return path

# Routine to compare the performance of the two functions
def compare_strategies(G1, G2, entry1, target1, entry2, target2, target_nodes1, target_nodes2, probabilities1, probabilities2):
path1_strategy_1 = deceptive_path_strategy_1(G1, entry1, target1, target_nodes2, probabilities1)
path2_strategy_1 = deceptive_path_strategy_1(G2, entry2, target2, target_nodes1, probabilities2)

path1_strategy_2 = deceptive_path_strategy_2(G1, entry1, target1, target_nodes1, probabilities1)
path2_strategy_2 = deceptive_path_strategy_2(G2, entry2, target2, target_nodes2, probabilities2)

# Compare paths and calculate cross probabilities
def calculate_cross_probabilities(path1, path2, probabilities1, probabilities2):
min_length = min(len(path1), len(path2))
path1 = path1[:min_length]
path2 = path2[:min_length]

truthful_steps = 0
cross_probabilities = []

for i in range(min_length):
node1 = path1[i]
node2 = path2[i]
cross_prob = {target: probabilities1[node1][target] * probabilities2[node2][target] for target in probabilities1[node1]}
cross_probabilities.append(cross_prob)
if max(cross_prob, key=cross_prob.get) == target1:
truthful_steps += 1

return truthful_steps, cross_probabilities

truthful_steps_1, cross_probabilities_1 = calculate_cross_probabilities(path1_strategy_1, path2_strategy_1, probabilities1, probabilities2)
truthful_steps_2, cross_probabilities_2 = calculate_cross_probabilities(path1_strategy_2, path2_strategy_2, probabilities1, probabilities2)

return {
'strategy_1': {
'truthful_steps': truthful_steps_1,
'cross_probabilities': cross_probabilities_1
},
'strategy_2': {
'truthful_steps': truthful_steps_2,
'cross_probabilities': cross_probabilities_2
}
}
