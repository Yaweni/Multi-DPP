import random

import networkx as nx
import numpy as np

from graph_creation import dual_graphs
import matplotlib.pyplot as plt
from collections import deque
from itertools import combinations, groupby

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
    for node in furthest_nodes:
        if node in G.graph['Targets']:
            furthest_nodes.remove(node)

    return furthest_nodes



def rank_nodes(subgraph, G2,t2,node_probabilities,entry,ldp):
    cross_prob_dict={}
    for node in subgraph.nodes:
        if node != entry:
            cross_prob_dict[node]={}
            for target in G2.graph['Targets']:
                cross_prob_dict[node][target]=sum([v * subgraph.graph['Target Probabilities'][k][target]\
                                                   for k,v in node_probabilities[node].items()])
    #print('Cros Prob',cross_prob_dict)
    return sorted(cross_prob_dict.items(), key=lambda x: x[1][t2], reverse=True)

def find_probs_in_tup_list(node,tup_list):
    for tup in tup_list:
        if tup[0] == node:
            return tup[1]


# Function 1: Deceptive path planning strategy 1
def deceptive_path_strategy_1(G1, G2, e1, e2, t1, t2):
    paths=[]
    path_probs=[]
    t11=t1
    t22=t2
    # Eliminate nodes where target node is the maximum probability
    if e1 in G1.graph['Targets'] or e2 in G2.graph['Targets']:
        raise
    for G,entry,target in zip([G1,G2],[e1,e2],[t1,t2]):
       #print('Entry Node:', entry, 'Target node: ', target[0])
        subgraph = G.copy()
        node_probabilities={}
        costs={}
        max_prob_target = {}
        min_cost = float('inf')
        temp_ldp= None
        ldp = None
        to_be_removed=[]

        for node in subgraph.nodes:
            if node in subgraph.graph['Targets']:
                node_probabilities[node] = {node:1}
                max_prob_target[node]=(node,1)
            else:
                costs[node] = {target: nx.dijkstra_path_length(subgraph,node,target,weight='weight') for target in subgraph.graph['Targets']}
                #print(costs[node])
                total_cost = sum([1/costs[node][k] for k,v in costs[node].items()])
                #print(total_cost)
                node_probabilities[node] = {target: (1/cost) / total_cost for target, cost in costs[node].items()}

                max_prob_target[node] = max(node_probabilities[node].items(), key=lambda x: x[1])

        #print(max_prob_target)
        #print(target[0])
        for node in list(subgraph.nodes):
            if (node != entry and max_prob_target[node][0] == target[0]) or (node in subgraph.graph['Targets']):
                to_be_removed.append(node)
        #print(to_be_removed)
        for node in list(subgraph.nodes):
            if node not in to_be_removed:
                #print(target[0])
                #print(costs[node].get(target[0]))
                if costs[node].get(target[0]) < min_cost:
                    min_cost = costs[node].get(target[0])
                    temp_ldp = node
        #print(temp_ldp)
        #print('Neighbors',list(subgraph.neighbors(temp_ldp)))
        if target[0] not in subgraph.neighbors(temp_ldp):
            for node in subgraph.neighbors(temp_ldp):
                if node not in subgraph.graph['Targets']:
                    if costs[node].get(target[0]) < min_cost:
                        min_cost = costs[node].get(target[0])
                        ldp = node
                        if ldp in to_be_removed:
                            to_be_removed.remove(ldp)
        else:
            ldp= temp_ldp
            if ldp in to_be_removed:
                to_be_removed.remove(ldp)
        subgraph.remove_nodes_from(to_be_removed)

#---------------------Rank the nodes, remove and complete the algo
        # Rank nodes by probability of the real target in the other graph
        if G == G2:
            t2,t1=t1,t2
            G1,G2,=G2,G1
            e1,e2=e2,e1
        #print(t2[0],'t2')
        ranked_nodes = rank_nodes(subgraph,G2,t2[0],node_probabilities,e2,ldp)
        # Remove nodes by highest value of real target probability
        #print(ranked_nodes,'ranked')

        for node in ranked_nodes:
            node = node[0]
            if node != entry and node != ldp and nx.has_path(subgraph, entry, ldp):
                dummy_sub=subgraph.copy()
                dummy_sub.remove_node(node)
                if nx.has_path(dummy_sub,entry,ldp):
                    subgraph.remove_node(node)
                else:
                    break
        #print(ranked_nodes)
        # Calculate the shortest path from entry to LDP
        path = nx.shortest_path(subgraph, source=entry, target=ldp, weight='weight')
        paths.append(path)
        store1=[]
        store2=[]

        for rank in ranked_nodes:
            for node in path:
                if node==rank[0] and node != entry:
                    store1.append(rank)
                    store2.append(node_probabilities[node])
 #       print('Cross prob path',store1)
  #      print('Target prob path',store2)
        path_probs.append((store1,store2))


    #print('Targets graph 1:', G2.graph['Targets'])
    #print('Targets graph 2:', G1.graph['Targets'])
    return paths,path_probs,t11[0],t22[0]


def deceptive_path_strategy_2(G1, G2, e1, e2, t1, t2):
    paths=[]
    path_probs=[]
    t11=t1
    t22=t2
    # Eliminate nodes where target node is the maximum probability
    if e1 in G1.graph['Targets'] or e2 in G2.graph['Targets']:
        raise
    for G,entry,target in zip([G1,G2],[e1,e2],[t1,t2]):
        #print('Entry Node:', entry, 'Target node: ', target[0])
        subgraph = G.copy()
        node_probabilities={}
        costs={}
        max_prob_target = {}
        min_cost = float('inf')
        temp_ldp= None
        ldp = None
        to_be_removed=[]

        for node in subgraph.nodes:
            if node in subgraph.graph['Targets']:
                node_probabilities[node] = {node:1}
                max_prob_target[node]=(node,1)
            else:
                costs[node] = {target: nx.dijkstra_path_length(subgraph,node,target,weight='weight') for target in subgraph.graph['Targets']}
                #print(costs[node])
                total_cost = sum([1/costs[node][k] for k,v in costs[node].items()])
                #print(total_cost)
                node_probabilities[node] = {target: (1/cost) / total_cost for target, cost in costs[node].items()}

                max_prob_target[node] = max(node_probabilities[node].items(), key=lambda x: x[1])

        #print(max_prob_target)
        #print(target[0])
        for node in list(subgraph.nodes):
            if (node != entry and max_prob_target[node][0] == target[0]) or (node in subgraph.graph['Targets']):
                to_be_removed.append(node)
        #print(to_be_removed)
        for node in list(subgraph.nodes):
            if node not in to_be_removed:
                #print(target[0])
                #print(costs[node].get(target[0]))
                if costs[node].get(target[0]) < min_cost:
                    min_cost = costs[node].get(target[0])
                    temp_ldp = node
        #print(temp_ldp)
        #print('Neighbors',list(subgraph.neighbors(temp_ldp)))
        if target[0] not in subgraph.neighbors(temp_ldp):
            for node in subgraph.neighbors(temp_ldp):
                if node not in subgraph.graph['Targets']:
                    if costs[node].get(target[0]) < min_cost:
                        min_cost = costs[node].get(target[0])
                        ldp = node
                        if ldp in to_be_removed:
                            to_be_removed.remove(ldp)
        else:
            ldp= temp_ldp
            if ldp in to_be_removed:
                to_be_removed.remove(ldp)
        subgraph.remove_nodes_from(to_be_removed)
        if G == G2:
            t2,t1=t1,t2
            G1,G2,=G2,G1
            e1,e2=e2,e1
        ranked_nodes = rank_nodes(subgraph,G2,t2[0],node_probabilities,e2,ldp)
        path = nx.shortest_path(subgraph, source=entry, target=ldp, weight='weight')
        paths.append(path)
        store1=[]
        store2=[]

        for rank in ranked_nodes:
            for node in path:
                if node==rank[0] and node != entry:
                    store1.append(rank)
                    store2.append(node_probabilities[node])
 #       print('Cross prob path',store1)
  #      print('Target prob path',store2)
        path_probs.append((store1,store2))


    #print('Targets graph 1:', G2.graph['Targets'])
    #print('Targets graph 2:', G1.graph['Targets'])
    return paths,path_probs,t11[0],t22[0]


def evaluate_strategy(paths,path_probs,t1,t2):
    #for path in paths:
        #print('Path graph',paths.index(path)+1,':',path)
    #for path_prob in path_probs:
     #   print(path_prob[0])
      #  print(path_prob[1])
    paths[0]=paths[0][1:]
    paths[1]=paths[1][1:]
    numbers = abs(len(paths[0]) - len(paths[1]))
    cut=min(len(paths[0]),len(paths[1]))
    #print(numbers)
    path1=paths[0][-cut:]
    path2=paths[1][-cut:]
    #print('Trimmed',path1,path2)
    single_truthful_steps_G1=[]
    single_deceitful_steps_G1=[]
    num_sing_truth_G1=0
    num_sing_deceitful_G1=0
    single_truthful_steps_G2=[]
    single_deceitful_steps_G2=[]
    num_sing_truth_G2=0
    num_sing_deceitful_G2=0
    if len(paths[0]) > cut:
        target = t2
        single_cut=len(paths[0])-cut
        single = paths[0][:single_cut]
       # print('Single steps Graph 1',single)
        probs = path_probs[0][0]
        for sing in single:
            for prob in probs:
                if prob[0] == sing:
                    if max(prob[1],key=prob[1].get) == t2:
                        single_truthful_steps_G1.append((sing,prob[1][t2]))
                        num_sing_truth_G1 += 1
                    else:
                        single_deceitful_steps_G1.append((prob[1][t2],(max(prob[1],key=prob[1].get),max(prob[1].values()))))
                        num_sing_deceitful_G1+=1
        #print('Single deceitful steps G1',single_deceitful_steps_G1)
        #print('Single truthful steps G1',single_truthful_steps_G1)
    elif len(paths[1]) > cut:
        target = t1
        single_cut=len(paths[1])-cut
        single = paths[1][:single_cut]
        probs = path_probs[1][0]
        #print('Single steps graph 2',single)
        #print(probs)
        for sing in single:
            for prob in probs:
                if prob[0] == sing:
                    if max(prob[1],key=prob[1].get) == t1:
                        single_truthful_steps_G2.append((sing,prob[1][t1]))
                        num_sing_truth_G2 += 1
                    else:
                        single_deceitful_steps_G2.append((prob[1][t1],(max(prob[1],key=prob[1].get),max(prob[1].values()))))
                        num_sing_deceitful_G2+=1
     #   print('Single deceitful steps G2',single_deceitful_steps_G2)
      #  print('Single truthful steps G2',single_truthful_steps_G2)

    joint_truthful_steps=[]
    joint_deceitful_steps=[]
    num_joint_truth=0
    num_joint_deceitful=0
    #print('Joint steps:',[(n1,n2) for n1,n2 in zip(path1,path2)])
    for n1,n2 in zip(path1,path2):
        joint_step_probs={}
        pos1 = paths[0].index(n1)
        pos2 = paths[1].index(n2)
        step_inprobs_G1=path_probs[0][1][pos1]
        step_inprobs_G2=path_probs[1][1][pos2]
        step_outprobs_G1=find_probs_in_tup_list(n1,path_probs[0][0])
        step_outprobs_G2=find_probs_in_tup_list(n2,path_probs[1][0])
        g1_res_dict={}
        g2_res_dict={}
        for key in step_inprobs_G1:
            if key in step_outprobs_G2:
                g1_res_dict[key] = step_inprobs_G1[key] * step_outprobs_G2[key]
        for key in step_inprobs_G2:
            if key in step_outprobs_G1:
                g2_res_dict[key] = step_inprobs_G2[key] * step_outprobs_G1[key]
        #print(g1_res_dict)
        #print(g2_res_dict)
        if (max(g1_res_dict,key=g1_res_dict.get)==t1) or (max(g2_res_dict,key=g2_res_dict.get)==t2) :
            joint_truthful_steps.append((n1,n2))
            num_joint_truth+=1
        else:
            joint_deceitful_steps.append((n1,n2))
            num_joint_deceitful+=1

    #print('Joint deceitful: ',joint_deceitful_steps)
    #print('Joint truthful: ',joint_truthful_steps)
    if num_sing_deceitful_G1 > 0:
        return single_truthful_steps_G1,single_deceitful_steps_G1,joint_truthful_steps,joint_deceitful_steps
    else:
        return single_truthful_steps_G2,single_deceitful_steps_G2,joint_truthful_steps,joint_deceitful_steps


def compare_strategies(truth1,dec1,joint_t1,joint_d1,truth2,dec2,joint_t2,joint_d2):
    path_length1 = sum([len(truth1),len(dec1),len(joint_d1),len(joint_t1)])
    path_length2 = sum([len(truth2),len(dec2),len(joint_d2),len(joint_t2)])

    if path_length1 > path_length2 :
        print('Path 1 longer:' ,path_length1, path_length2)
    elif path_length2 > path_length1:
        print('Path 2 longer:' ,path_length1, path_length2)
    percent_decep_s1=sum([len(dec1),len(joint_d1)])/path_length1
    percent_decep_s2=sum([len(dec2),len(joint_d2)])/path_length2
    print('Deception percentages Optimized vs plain',percent_decep_s1,percent_decep_s2)



'''for nodes_g1 in range(50,252,10):
    for nodes_g2 in range(50,252,10):
        for targets1 in range(5,15):
            for targets2 in range(5,15):
                for prob in np.arange(0.2,0.8,0.04):
                    g1,g2=dual_graphs(nodes_g1,nodes_g2,targets1,targets2,prob,prob)
                    '''





g1,g2=dual_graphs(125,180,5,7,0.03,0.01)
rand_t1=random.sample(g1.graph['Targets'],1)
rand_t2= random.sample(g2.graph['Targets'],1)
entry1=find_furthest_nodes(g1,rand_t1[0])[0]
entry2=find_furthest_nodes(g2,rand_t2[0])[0]
print(entry1,entry2,' Entries')
#print(rand_t2)
try:
    paths1,path_probs1,t1,t2=deceptive_path_strategy_1(g1,g2,11,9,rand_t1,rand_t2)

    paths2,path_probs2,t1,t2=deceptive_path_strategy_2(g1,g2,11,9,rand_t1,rand_t2)

    truth1,dec1,joint_t1,joint_d1=evaluate_strategy(paths1,path_probs1,t1,t2)

    truth2,dec2,joint_t2,joint_d2=evaluate_strategy(paths2,path_probs2,t1,t2)
except:
    print('Error')
else:
    compare_strategies(truth1,dec1,joint_t1,joint_d1,truth2,dec2,joint_t2,joint_d2)
