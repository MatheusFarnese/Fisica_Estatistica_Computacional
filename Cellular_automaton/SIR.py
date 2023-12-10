import numpy as np
import random as rd
import matplotlib.pyplot as plt
import math


def neighbours(N):
    L=int(np.sqrt(N))
    ngh = np.zeros((N,4),dtype=np.int16)
    for k in range(N):
        ngh[k,0] = k + 1
        if ((k + 1) % L == 0):
            ngh[k,0] = k + 1 - L

        ngh[k,1] = k + L
        if (k > (N - L - 1)):
            ngh[k,1] = k + L - N

        ngh[k,2] = k - 1
        if (k % L == 0):
            ngh[k,2] = k + L - 1

        ngh[k,3] = k - L
        if (k < L):
            ngh[k,3] = k + N - L
    return ngh


def print_network(network, n):
    dummy_line = np.zeros(n)
    dummy_line[1] = 1
    dummy_line[2] = 2
    net = network.reshape(n, n)
    net = np.vstack((net, dummy_line))
    label_map = {0: "SUS", 1: "SICK", 2: "REC"}
    mapped_array = np.vectorize(label_map.get)(net)

    cmap = plt.get_cmap('inferno', len(label_map))

    fig, ax = plt.subplots()

    im = ax.imshow(net, cmap=cmap, interpolation='nearest')

    cbar = plt.colorbar(im, ticks=list(label_map.keys()))
    cbar.set_ticklabels(list(label_map.values()))
    cbar.set_label('State')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(np.arange(1, n + 1))
    ax.set_yticklabels(np.arange(1, n + 1))

    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, mapped_array[i, j], ha='center', va='center', color='g')

    plt.show()


def replicate_last_element(lst, n):
    if len(lst) >= n:
        return lst[:n]

    last_element = lst[-1]
    replicated_list = lst + [last_element] * (n - len(lst))
    return replicated_list


def mean_of_lists(list_of_lists):
    if not list_of_lists or not all(len(lst) == len(list_of_lists[0]) for lst in list_of_lists):
        raise ValueError("Input lists must be non-empty and have the same size.")

    n = len(list_of_lists[0])
    transposed_lists = zip(*list_of_lists)
    mean_values = [sum(pos_elements) / len(pos_elements) for pos_elements in transposed_lists]
    return mean_values


def plot_values(sus, infec, rec):
    n = len(sus)

    if n != len(infec) or n != len(rec):
        raise ValueError("All lists must have the same length.")

    time_values = list(range(n))
    fig, ax = plt.subplots()

    #ax.semilogy(time_values, sus, label='Susceptible', color='blue')
    ax.plot(time_values, sus, label='Susceptible', color='blue')
    ax.plot(time_values, infec, label='Infected', color='red')
    ax.plot(time_values, rec, label='Recovered', color='green')

    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.set_title('Evolution over time')
    ax.legend()

    plt.show()


def plot_var_prob(sus, infec, rec, prob):
    n = len(sus)

    if n != len(infec) or n != len(rec) or n != len(prob):
        raise ValueError("All lists must have the same length.")

    fig, ax = plt.subplots()

    ax.plot(prob, sus, label='Susceptible', color='blue')
    ax.plot(prob, infec, label='Infected', color='red')
    ax.plot(prob, rec, label='Recovered', color='green')

    ax.set_xlabel('Recovery probability')
    ax.set_ylabel('Values')
    ax.set_title('End values for each recovery probability')
    ax.legend()

    plt.show()


def cell_automaton(size, contamination_probability, recovery_probability, show_network = False):
    neighborhood = neighbours(size)
    network = np.zeros(size)
    network = network.astype(int)

    seed = rd.randint(0, size - 1)
    network[seed] = 1
    infected = [seed]
    if (show_network):
        print_network(network, int(np.sqrt(size)))

    new_infected = []
    sus_count = size - 1
    infected_count = 1
    rec_count = 0

    sus_n_list = [sus_count]
    infected_n_list = [infected_count]
    rec_n_list = [rec_count]
    it = 0
    while (infected_count != 0 and it < 200):
        it += 1
        for propagator in infected:
            for victim in neighborhood[propagator]:
                if (network[victim] == 0):
                    p = rd.random()
                    if (p < contamination_probability):
                        new_infected.append(victim)
                        network[victim] = 1
                        infected_count += 1
                        sus_count -= 1
        for individual in infected:
            p = rd.random()
            if (p < recovery_probability):
                network[individual] = 2
                infected_count -= 1
                rec_count += 1
            else:
                new_infected.append(individual)
        infected = new_infected
        new_infected = []
        sus_n_list.append(sus_count)
        infected_n_list.append(infected_count)
        rec_n_list.append(rec_count)
        if (show_network):
            print_network(network, int(np.sqrt(size)))
    return (network, sus_n_list, infected_n_list, rec_n_list)

def run_sir(L, n_it, contamination_probability, recovery_probability, op, delta_p = 0.01):
    if (op == 1):
        size = L * L
        sus_results = []
        infec_results = []
        rec_results = []
        biggest_sim = 0

        for i in range (n_it):
            net, sus, infec, rec = cell_automaton(size, contamination_probability, recovery_probability)
            sus_results.append(sus)
            infec_results.append(infec)
            rec_results.append(rec)
            if (len(sus) > biggest_sim):
                biggest_sim = len(sus)

        for i in range (n_it):
            sus_results[i] = replicate_last_element(sus_results[i], biggest_sim)
            infec_results[i] = replicate_last_element(infec_results[i], biggest_sim)
            rec_results[i] = replicate_last_element(rec_results[i], biggest_sim)

        sus_mean = mean_of_lists(sus_results)
        infec_mean = mean_of_lists(infec_results)
        rec_mean = mean_of_lists(rec_results)
        plot_values(sus_mean, infec_mean, rec_mean)
    elif (op == 2):
        size = L * L
        for i in range (n_it):
            cell_automaton(size, contamination_probability, recovery_probability, True)
    elif (op == 3):
        size = L * L
        sus_list = []
        infec_list = []
        rec_list = []

        sus_results = []
        infec_results = []
        rec_results = []

        rec_p = recovery_probability
        prob_list = []
        for i in range (n_it):
            while(rec_p <= 1):
                net, sus, infec, rec = cell_automaton(size, contamination_probability, rec_p)
                sus_list.append(sus[-1])
                infec_list.append(infec[-1])
                rec_list.append(rec[-1])
                if (i == 0):
                    prob_list.append(rec_p)
                rec_p = rec_p + delta_p

            sus_results.append(sus_list)
            infec_results.append(infec_list)
            rec_results.append(rec_list)
            sus_list = []
            infec_list = []
            rec_list = []
            rec_p = recovery_probability

        sus_mean = mean_of_lists(sus_results)
        infec_mean = mean_of_lists(infec_results)
        rec_mean = mean_of_lists(rec_results)
        plot_var_prob(sus_mean, infec_mean, rec_mean, prob_list)


run_sir(10, 100, 0.05, 0.05, 1)
run_sir(10, 100, 0.2, 0.2, 1)
run_sir(10, 100, 0.4, 0.4, 1)
run_sir(10, 100, 0.8, 0.8, 1)

run_sir(10, 100, 0.4, 0.6, 1)
run_sir(10, 100, 0.4, 0.8, 1)
run_sir(10, 100, 0.6, 0.4, 1)
run_sir(10, 100, 0.8, 0.4, 1)

run_sir(10, 1, 0.4, 0.6, 2)

#run_sir(50, 100, 0.2, 0, 3)
#run_sir(50, 100, 0.4, 0, 3)
#run_sir(50, 100, 0.6, 0, 3)
#run_sir(50, 100, 0.8, 0, 3)
