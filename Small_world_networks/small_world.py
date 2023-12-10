import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random as rd
from statistics import mean
from math import pi

def build(N, Z, p, show=False):
    G = nx.Graph()
    V = list(range(N))
    E = []
    for i in range (N):
        for j in range (1, int(Z/2 + 1)):
            E.append((i, (i + j)%N))
            E.append((i, (i - j)%N))
    unique_tuples = set(frozenset(pair) for pair in E)
    E = [tuple(pair) for pair in unique_tuples]
    G.add_nodes_from(V)
    G.add_edges_from(E)

    n_shortcuts = 0
    for i in range(int(p*N*Z/2)):
        a = rd.randint(0, N-1)
        b = rd.randint(0, N-1)
        G.add_edge(a, b)
        if (not((abs(a-b) <= int(Z/2)) or(abs(a-b) >= int(N - Z/2)))):
            n_shortcuts += 1

    if(show):
        nx.draw_circular(G, node_shape='.')
        plt.show()
    return (G, n_shortcuts)

def BFS(graph, node):
    start_node = node
    am = nx.adjacency_matrix(graph)
    am = am.todense()
    shell = [node]
    next_shell = []
    distances = np.zeros(graph.number_of_nodes())
    distances = distances.tolist()
    d = 1
    distances[node] = -1
    while (shell):
        next_shell = []
        for i in range (graph.number_of_nodes()):
            for j in range (len(shell)):
                if ((am[shell[j], i]) == 1):
                    if (distances[i] == 0):
                        distances[i] = d
                        next_shell.append(i)
        shell = next_shell
        d = d + 1
    distances[node] = 0
    return distances

def FindPathLengthsFromNode(graph, node):
    start_node = node
    distances = nx.shortest_path_length(graph, source=start_node)
    distance_list = [distances.get(node, 0) for node in range(graph.number_of_nodes())]
    return distance_list

def FindAllPathLengths(graph):
    full = []
    for i in range (graph.number_of_nodes()):
        s = FindPathLengthsFromNode(graph, i)
        full += s
    return full

def FindAveragePathLength(graph):
    av = []
    for i in range (graph.number_of_nodes()):
        s = FindPathLengthsFromNode(graph, i)
        av.append(mean(s))
    return mean(av)

def analysis_a(N, Z, p):
    build(N, Z, p, show=True)

def analysis_b(N, Z, p):
    (G_base, sc) = build(N, Z, 0)
    (G1, sc1) = build(N, Z, p)
    (G2, sc2) = build(N, Z, p * 0.1)

    paths_base = FindAllPathLengths(G_base)
    plt.hist(paths_base)
    plt.show()

    paths1 = FindAllPathLengths(G1)
    plt.hist(paths1)
    plt.show()

    paths2 = FindAllPathLengths(G2)
    plt.hist(paths2)
    plt.show()

    nx.draw_circular(G1, node_shape='.')
    plt.show()

    nx.draw_circular(G2, node_shape='.')
    plt.show()

    print("Número de atalhos: ", sc)
    print("Número de atalhos: ", sc1)
    print("Número de atalhos: ", sc2)

    for i in range (20):
        (G, sc) = build(100, 2, 0.1)
        print("Caminho médio:  ", FindAveragePathLength(G))
        print("Número de arestas longas:  ", sc)
        print()

def analysis_c(N, Z, p):
    (base_graph, trash) = build(N, Z, 0)
    dp0 = FindAveragePathLength(base_graph)
    result = []
    p_list = []
    while (p < 1100):
        p_list.append(p)
        (G, trash) = build(N, Z, p)
        dp = FindAveragePathLength(G)
        result.append(dp/dp0)
        p = p * 2
    plt.scatter(p_list, result, color='b', label='Points')
    plt.xscale('log')
    plt.show()

def analysis_d1(N, Z, p):
    (G, sc) = build(N, Z, p)
    dp = FindAveragePathLength(G)
    print("Delta_theta = ", pi*Z*dp/N)

    G1 = nx.watts_strogatz_graph(1000, 10, p)
    dp = FindAveragePathLength(G1)
    print("Delta_theta(p=0.1) = ", pi*10*dp/1000)

    G2 = nx.watts_strogatz_graph(1000, 10, p*0.01)
    dp = FindAveragePathLength(G2)
    print("Delta_theta(p=0.001) = ", pi*10*dp/1000)

    M1 = 0.1*1000*10/2
    M2 = 0.001*1000*10/2
    print()
    print("Shortcuts = ", sc)
    print("M(p=0.1) = ", M1)
    print("M(p=0.001) = ", M2)

    nx.draw_circular(G, node_shape='.')
    plt.show()
    nx.draw_circular(G1, node_shape='.')
    plt.show()
    nx.draw_circular(G2, node_shape='.')
    plt.show()

def analysis_d2(N, Z, p):
    delta_theta1 = []
    n_shortcuts1 = []
    delta_theta2 = []
    n_shortcuts2 = []
    delta_theta3 = []
    n_shortcuts3 = []
    delta_theta4 = []
    n_shortcuts4 = []

    while (p < 1100):

        (G1, sc1) = build(N  , Z  , p)
        (G2, sc2) = build(N*2, Z  , p)
        (G3, sc3) = build(N  , Z*2, p)
        (G4, sc4) = build(N*2, Z*2, p)

        dp1 = FindAveragePathLength(G1)
        dp2 = FindAveragePathLength(G2)
        dp3 = FindAveragePathLength(G3)
        dp4 = FindAveragePathLength(G4)

        delta_theta1.append(pi*Z*dp1/N)
        n_shortcuts1.append(sc1)
        delta_theta2.append(pi*Z*dp2/(N*2))
        n_shortcuts2.append(sc2)
        delta_theta3.append(pi*Z*2*dp3/N)
        n_shortcuts3.append(sc3)
        delta_theta4.append(pi*Z*2*dp4/(N*2))
        n_shortcuts4.append(sc4)

        p = p * 2

    plt.scatter(n_shortcuts1, delta_theta1, color='b')
    plt.xscale('log')
    plt.show()
    plt.scatter(n_shortcuts2, delta_theta2, color='b')
    plt.xscale('log')
    plt.show()
    plt.scatter(n_shortcuts3, delta_theta3, color='b')
    plt.xscale('log')
    plt.show()
    plt.scatter(n_shortcuts4, delta_theta4, color='b')
    plt.xscale('log')
    plt.show()


analysis_a  (N = 20,   Z = 4, p = 0.2)
analysis_b  (N = 1000, Z = 2, p = 0.2)
analysis_c  (N = 50,   Z = 2, p = 0.001)
analysis_d1 (N = 50,   Z = 2, p = 0.1)
analysis_d2 (N = 100,  Z = 2, p = 0.001)
