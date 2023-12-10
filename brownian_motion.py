import numpy as np
import random as rd
import matplotlib.pyplot as plt
from scipy.stats import norm
from numba import jit


@jit(nopython=True)
def random_walk(n_steps, d):
    path = [(0, 0)]

    if (d == 1):
        position = 0
        for i in range (n_steps):
            step = rd.uniform(-0.5, 0.5)
            position += step
            path.append((position, i + 1))

    if (d == 2):
        position_x = 0
        position_y = 0
        for i in range (n_steps):
            step_x = rd.uniform(-0.5, 0.5)
            step_y = rd.uniform(-0.5, 0.5)
            position_x += step_x
            position_y += step_y
            path.append((position_x, position_y))

    return path


def plot_walk(path, d):
    x_values, y_values = zip(*path)
    plt.plot(x_values, y_values, marker='', linestyle='-', color='b')

    if (d == 1):
        plt.title('Random walk')
        plt.xlabel('Position x(t)')
        plt.ylabel('Time t')
        plt.xlim([-max(abs(min(x_values)), abs(max(x_values))) - 0.1, max(abs(min(x_values)), abs(max(x_values))) + 0.1])
        plt.axvline(0, color='gray', linestyle='--')

    if (d == 2):
        plt.title('Random walk')
        plt.xlabel('Position x(t)')
        plt.ylabel('Position y(t)')

        absolute_path = [abs(value) for tup in path for value in tup]
        limit = max(absolute_path)
        plt.xlim([-limit - (0.1 * limit), limit + (0.1 * limit)])
        plt.ylim([-limit - (0.1 * limit), limit + (0.1 * limit)])
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')

    plt.show()


def plot_final_positions(final_pos1, final_pos2):
    x1_values, y1_values = zip(*final_pos1)
    x2_values, y2_values = zip(*final_pos2)

    plt.scatter(x1_values, y1_values, color='green', label='N = 1', marker='.')
    plt.scatter(x2_values, y2_values, color='purple', label='N = 10', marker='.', alpha=0.25)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.title('Scatter plot for the final positions')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')

    plt.legend()
    plt.show()


def plot_histogram(final_positions, steps):
    plt.hist(final_positions, bins=50, color='blue', density=True)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    sigma = np.sqrt(steps / 12)
    px = norm.pdf(x, 0, sigma)
    plt.plot(x, px, linewidth=2, color='red', label='p(x)')

    plt.title(f'Histogram for final positions with {steps} steps')
    plt.xlabel('Position')
    plt.ylabel('Frequency')

    plt.legend()
    plt.show()
    return 0
    

def run_brownian_motion(n_steps, d, n_it = 1, op = 1):
    if (op == 1):
        path = random_walk(n_steps, d)
        plot_walk(path, d)

    if (op == 2):
        final_positions1 = []
        final_positions2 = []
        for i in range (n_it):
            path = random_walk(n_steps, d)
            final_positions1.append(path[-1])
            path = random_walk(n_steps * 10, d)
            final_positions2.append(path[-1])
        plot_final_positions(final_positions1, final_positions2)

    if (op == 3):
        final_positions = []
        for i in range (n_it):
            path = random_walk(n_steps, d)
            final_positions.append(path[-1][0])
        plot_histogram(final_positions, n_steps)


run_brownian_motion(n_steps = 10000, d = 1)
run_brownian_motion(n_steps = 10, d = 2)
run_brownian_motion(n_steps = 1000, d = 2)
run_brownian_motion(n_steps = 100000, d = 2)

run_brownian_motion(n_steps = 1, d = 2, n_it = 10000, op = 2)

run_brownian_motion(n_steps = 1, d = 1, n_it = 10000, op = 3)
run_brownian_motion(n_steps = 2, d = 1, n_it = 10000, op = 3)
run_brownian_motion(n_steps = 3, d = 1, n_it = 10000, op = 3)
run_brownian_motion(n_steps = 5, d = 1, n_it = 10000, op = 3)
run_brownian_motion(n_steps = 1000, d = 1, n_it = 10000, op = 3)
