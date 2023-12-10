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


def plot_values(sus, infec, rec, dead):
    n = len(sus)

    if n != len(infec) or n != len(rec) or n != len(dead):
        raise ValueError("All lists must have the same length.")

    time_values = list(range(n))
    fig, ax = plt.subplots()

    #ax.semilogy(time_values, sus, label='Susceptible', color='blue')
    ax.plot(time_values, sus, label='Susceptible', color='blue')
    ax.plot(time_values, infec, label='Infected', color='red')
    ax.plot(time_values, rec, label='Recovered', color='green')
    ax.plot(time_values, dead, label='Dead', color='black')

    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.set_title('Evolution over time')
    ax.legend()

    plt.show()


def cell_automaton(size, contamination_probability, recovery_probability, death_probability, show_network = False):
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
    dead_count = 0

    sus_n_list = [sus_count]
    infected_n_list = [infected_count]
    rec_n_list = [rec_count]
    dead_n_list = [dead_count]
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
            while (True):
                p1 = rd.random()
                p2 = rd.random()
                if (p1 < recovery_probability and p2 > death_probability):
                    network[individual] = 2
                    infected_count -= 1
                    rec_count += 1
                    break
                elif (p1 > recovery_probability and p2 < death_probability):
                    network[individual] = -1
                    infected_count -= 1
                    dead_count += 1
                    break
                elif (p1 > recovery_probability and p2 > death_probability):
                    new_infected.append(individual)
                    break

        infected = new_infected
        new_infected = []
        sus_n_list.append(sus_count)
        infected_n_list.append(infected_count)
        rec_n_list.append(rec_count)
        dead_n_list.append(dead_count)
        if (show_network):
            print_network(network, int(np.sqrt(size)))
    return (network, sus_n_list, infected_n_list, rec_n_list, dead_n_list)

def run_sir(L, n_it, contamination_probability, recovery_probability, death_probability, delta_p = 0.01):
    size = L * L
    sus_results = []
    infec_results = []
    rec_results = []
    dead_results = []
    biggest_sim = 0

    for i in range (n_it):
        net, sus, infec, rec, dead = cell_automaton(size, contamination_probability, recovery_probability, death_probability)
        sus_results.append(sus)
        infec_results.append(infec)
        rec_results.append(rec)
        dead_results.append(dead)
        if (len(sus) > biggest_sim):
            biggest_sim = len(sus)

    for i in range (n_it):
        sus_results[i] = replicate_last_element(sus_results[i], biggest_sim)
        infec_results[i] = replicate_last_element(infec_results[i], biggest_sim)
        rec_results[i] = replicate_last_element(rec_results[i], biggest_sim)
        dead_results[i] = replicate_last_element(dead_results[i], biggest_sim)

    sus_mean = mean_of_lists(sus_results)
    infec_mean = mean_of_lists(infec_results)
    rec_mean = mean_of_lists(rec_results)
    dead_mean = mean_of_lists(dead_results)
    plot_values(sus_mean, infec_mean, rec_mean, dead_mean)


run_sir(10, 100, 0.8, 0.5, 0.9)
run_sir(10, 100, 0.8, 0.5, 0.05)
run_sir(10, 100, 0.3, 0.5, 0.8)
run_sir(10, 100, 0.3, 0.5, 0.001)
