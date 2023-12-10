import time
from numba import jit
from numba.typed import List
import numpy as np
import matplotlib.pyplot as plt


#define a distancia entre duas cidades quaisquer
@jit(nopython=True)
def distances(N, x, y):

    dist = np.zeros((N,N),dtype=np.float32)
    for i in range(N):
        for j in range(N):
            dist[i,j] = np.sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))

    return dist

@jit(nopython=True)
def cost(N, path, dist):
    # calcula a distancia total percorrida pela caminhada
    ener = 0
    for i in range(N-1):
        ener += dist[path[i],path[i+1]]
    ener += dist[path[0],path[N-1]]     # conecta a última e a primeira cidades do caminho

    return ener

@jit(nopython=True)
def newpath(N, path):

    # define uma nova caminhada

    newpath = np.zeros(N, dtype=np.int16)

    i=np.random.randint(N)   # escolhe uma posição aleatória da caminhada
    j=i
    while j==i:
        j=np.random.randint(N)  # escolhe outra posição
    if i>j:                    # ordena os índices
        ini = j
        fin = i
    else:
        ini = i
        fin = j

    for k in range(N):        # inverte o sentido em que percorre o caminho entre os indices escolhidos
        if k >= ini and k <= fin:
            newpath[k] = path[fin-k+ini]
        else:
            newpath[k] = path[k]

    return (newpath, ini, fin)

@jit(nopython=True)
def step_and_a_step(N, beta, en, path, best_e, best_p, dist, steps):
    # realiza um passo de Monte Carlo
    np1 = np.zeros(N,dtype=np.int16)

    for i in range (steps):
        np1,ini,fin = newpath(N, path) # propoe um novo caminho

        # determina a diferença de energia
        esq = ini-1         # cidade anterior a inicial
        if esq < 0: esq=N-1      # condicao de contorno
        dir = fin +1        # cidade apos a final
        if dir > N-1: dir=0      # condicao de contorno
        de = -dist[path[esq],path[ini]] - dist[path[dir],path[fin]]+ dist[np1[esq],np1[ini]] + dist[np1[dir],np1[fin]]

        if de < 0:         # aplica o criterio de Metropolis
            en += de
            path = np1
            if en < best_e:  # guarda o melhor caminho gerado até o momento
                best_e = en
                best_p = path
        else:              # aplica o criterio de Metropolis
            if np.random.random() < np.exp(-beta*de):
                en += de
                path = np1
    return (en, path, best_e, best_p)

def print_enrg(n_steps, energies, temperatures):
    steps = np.arange(0, n_steps)

    fig, ax1 = plt.subplots()
    ax1.plot(steps, temperatures, color='tab:red', label='Temperature')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Temperature', color='tab:red')

    ax2 = ax1.twinx()

    ax2.plot(steps, energies, color='tab:blue', label='Energy')
    ax2.set_ylabel('Energy', color='tab:blue')

    plt.title('Temperature and Energy vs. Steps')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(0.7, 1))

    plt.show()

def print_cities(x, y, path):
    points = [(x[i], y[i]) for i in range(len(x))]
    index_array = path
    x_coords, y_coords = zip(*points)
    fig, ax = plt.subplots()

    ax.scatter(x_coords, y_coords, color='b', label='Points')

    for i in range(len(index_array) - 1):
        start_idx = index_array[i]
        end_idx = index_array[i + 1]
        start_point = points[start_idx]
        end_point = points[end_idx]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='r')

    start_idx = index_array[len(x) - 1]
    end_idx = index_array[0]
    start_point = points[start_idx]
    end_point = points[end_idx]
    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='r')

    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')

    ax.set_title('TS path')

    plt.legend()
    plt.grid(True)
    plt.show()

def tsp(N_mcs, dist, x, y, ini_path, ini_temp, end_temp, n_cities, delta_t, steps):
    temp = ini_temp
    best_energy = cost(n_cities, ini_path, dist)
    best_path = ini_path
    energy = best_energy
    path = best_path
    energies = []
    energies.append(energy)
    temperatures = []
    temperatures.append(ini_temp)
    while (temp > end_temp):
        beta = 1 / temp
        for i in range (N_mcs):
            (energy, path, best_energy, best_path) = step_and_a_step(n_cities, beta, energy, path, best_energy, best_path, dist, steps)
            energies.append(energy)
            temperatures.append(temp)
        temp = temp*delta_t
    n_steps = len(energies)
    print_enrg(n_steps, energies, temperatures)
    print_cities(x, y, best_path)
    print ("Best path -> ", best_energy)
    return best_path

def run_tsp(N_mcs, ini_temp, end_temp, delta_t, n_runs, steps):
    xx = []
    yy = []
    data = np.genfromtxt(input_file)
    xx = [sublist[0] for sublist in data]
    yy = [sublist[1] for sublist in data]
    x = List()
    y = List()
    [x.append(s) for s in xx]
    [y.append(s) for s in yy]
    n_cities = len(x)
    dist = distances(n_cities, x, y)
    ini_path = np.zeros(n_cities, dtype=np.int16)
    for i in range(n_cities):
        ini_path[i] = i
    best_path = ini_path

    for i in range (n_runs):
        best_path = tsp(N_mcs, dist, x, y, best_path, ini_temp, end_temp, n_cities, delta_t, steps)
        ini_temp = ini_temp * 0.01


input_file="1200c.txt"

N_MCS = 100
INI_TEMP = 0.1
END_TEMP = 0.000001
DELTA_T = 0.88
N_RUNS = 1
STEPS = 1000

run_tsp(N_MCS, INI_TEMP, END_TEMP, DELTA_T, N_RUNS, STEPS)
