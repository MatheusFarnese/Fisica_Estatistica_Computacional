import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from statistics import mean

def FindPathLengthsFromNode(graph, node, node_list):
    start_node = node
    distances = nx.shortest_path_length(graph, source=start_node)
    distance_list = [distances.get(node, 0) for node in node_list]
    return distance_list

def FindAllPathLengths(graph, nodes):
    full = []
    for i in nodes:
        s = FindPathLengthsFromNode(graph, i, nodes)
        full += s
    return full

def FindAveragePathLength(graph, nodes):
    av = []
    for i in nodes:
        s = FindPathLengthsFromNode(graph, i, nodes)
        av.append(mean(s))
    return mean(av)

def create_graph(file_path):
    G = nx.Graph()

    with open(file_path, 'r') as file:
        nodes_line = file.readline().strip()
        nodes = list(map(int, nodes_line.split(',')))

        G.add_nodes_from(nodes)

        for line in file:
            u, v = map(int, line.split())

            if u in nodes and v in nodes:
                G.add_edge(u, v)

    return (G, nodes)

file_path = 'proteins.txt'
(graph, node_list) = create_graph(file_path)

print("Number of nodes:", graph.number_of_nodes())
print("Number of edges:", graph.number_of_edges())

paths = FindAllPathLengths(graph, node_list)
plt.hist(paths)
plt.show()

print("Distância média: ", FindAveragePathLength(graph, node_list))

nx.draw(graph, node_shape='')
plt.show()
