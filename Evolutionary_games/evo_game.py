import numpy as np
import random as rd
import matplotlib.pyplot as plt
from numba import jit


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


@jit(nopython=True)
def step_and_a_step(network, size, neighborhood, payoff):
    for i in range (size):
        if (network[i] > 0.1707):
            network[i] = 1
            for neighbour in neighborhood[i]:
                if (network[neighbour] > 0.1707):
                    network[i] += 1
        else:
            network[i] = 0
            for neighbour in neighborhood[i]:
                if (network[neighbour] > 0.1707):
                    network[i] -= payoff

    return network


@jit(nopython=True)
def change_strategies(network, size, neighborhood, K):
    n_cooperators = 0
    changed_network = np.zeros(size)

    for i in range (size):
        target_neighbour = rd.randint(0, 3)
        ngh = neighborhood[i][target_neighbour]
        change_probability = 1 / (1 + np.exp( - ((abs(network[ngh]) - abs(network[i])) / K)))
        if (rd.random() < change_probability):
            changed_network[i] = network[ngh]
            if (network[ngh] > 0):
                n_cooperators += 1
        else:
            changed_network[i] = network[i]
            if (network[i] > 0):
                n_cooperators += 1

    return (changed_network, n_cooperators / size)


def print_network(network, n):
    for i in range (len(network)):
        if (network[i] > 0):
            network[i] = 1
        else:
            network[i] = 0
    net = network.reshape(n, n)
    label_map = {0: "SUCKER", 1: "COOPERATOR"}
    mapped_array = np.vectorize(label_map.get)(net)

    cmap = plt.get_cmap('prism', len(label_map))

    fig, ax = plt.subplots()

    im = ax.imshow(net, cmap=cmap, interpolation='nearest')

    cbar = plt.colorbar(im, ticks=list(label_map.keys()))

    plt.show()


def plot_b(b_list, cooperators):
    fig, ax = plt.subplots()
    ax.plot(b_list, cooperators, color='deeppink')
    ax.set_xlabel('Payoff (b)')
    ax.set_ylabel('Cooperators')
    ax.set_title('Number of Cooperators for different payoff (b) values')
    plt.show()


def evo_archer(coop_list, b_list):
    colors = ['deeppink', 'magenta', 'darkviolet', 'mediumblue', 'cornflowerblue', 'lightseagreen', 'aqua', 'lime', 'orange', 'darkorange', 'crimson']
    fig, ax = plt.subplots()

    for i in range (len(coop_list)):
        n = len(coop_list[i])
        time_values = list(range(n))
        ax.plot(time_values, coop_list[i], label=f'b = {b_list[i]}', color=colors[i])
        if (i == 10):
            break

    ax.set_xlabel('Time')
    ax.set_ylabel('Cooperators')
    ax.set_title('Evolution over time')
    ax.legend()
    plt.show()


def evo_cracker(size, steps, neighborhood, payoff, K, show_network = False):
    network = np.zeros(size)
    n_cooperators = 0

    for i in range (size):
        random_value = rd.choice([0, 1])
        network[i] = random_value
        if (random_value == 1):
            n_cooperators += 1

    cooperators = [n_cooperators / size]


    for i in range (steps):
        network = step_and_a_step(network, size, neighborhood, payoff)
        network, n_cooperators = change_strategies(network, size, neighborhood, K)
        cooperators.append(n_cooperators)

    return network, cooperators


def run_game(square_side, steps, ini_payoff = 1, end_payoff = 2, n_it = 50, K = 0.26, op = 1):
    size = square_side * square_side
    neighborhood = neighbours(size)

    if (op == 1):
        coop_list = []
        payoff_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        for payoff in payoff_list:
            network, cooperators = evo_cracker(size, steps, neighborhood, payoff, K)
            coop_list.append(cooperators)
        evo_archer(coop_list, payoff_list)

    if (op == 2):
        payoff_list = [1.29, 1.3, 1.6, 1.63]
        for payoff in payoff_list:
            network, cooperators = evo_cracker(size, steps, neighborhood, payoff, K)
            print_network(network, int(np.sqrt(size)))

    if (op == 3):
        payoff = ini_payoff
        delta_payoff = (end_payoff - ini_payoff) / (n_it - 1)
        payoff_list = []
        final_coop_nums = []
        if (steps < 401):
            steps = 401

        for i in range (n_it):
            payoff_list.append(payoff)
            network, cooperators = evo_cracker(size, steps, neighborhood, payoff, K)
            payoff = payoff + delta_payoff
            final_coop_nums.append(np.array(cooperators[400:]).mean())

        plot_b(payoff_list, final_coop_nums)
        print(final_coop_nums)

run_game(200, 800, op = 1)
run_game(200, 800, op = 2)
run_game(200, 800, op = 3)
