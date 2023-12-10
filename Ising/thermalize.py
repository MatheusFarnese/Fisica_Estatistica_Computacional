import numpy as np
import random as rd
import matplotlib.pyplot as plt
import math
from numba import jit


@jit(nopython=True)
def vizinhos(N):
#Define a tabela de vizinhos
    L=int(np.sqrt(N))
    viz = np.zeros((N,4),dtype=np.int16)
    for k in range(N):
        viz[k,0]=k+1
        if (k+1) % L == 0: viz[k,0] = k+1-L
        viz[k,1] = k+L
        if k > (N-L-1): viz[k,1] = k+L-N
        viz[k,2] = k-1
        if (k % L == 0): viz[k,2] = k+L-1
        viz[k,3] = k-L
        if k < L: viz[k,3] = k+N-L
    return viz


@jit(nopython=True)
def expos(beta):
    ex = np.zeros(5,dtype=np.float32)
    ex[0]=np.exp(8.0*beta)
    ex[1]=np.exp(4.0*beta)
    ex[2]=1.0
    ex[3]=np.exp(-4.0*beta)
    ex[4]=np.exp(-8.0*beta)
    return ex


@jit(nopython=True)
def delta_e(i):
    if (i == 0): return -8
    if (i == 1): return -4
    if (i == 2): return 0
    if (i == 3): return 4
    if (i == 4): return 8
    return 0


@jit(nopython=True)
def energia(s,viz):
#Calcula a energia da configuração representada no array s
    N=len(s)
    ener = 0
    for i in range(N):
        h = s[viz[i,0]]+s[viz[i,1]] # soma do valor dos spins a direita e acima
        ener -= s[i]*h
    return ener


@jit(nopython=True)
def vsum(s,adj):
    return (s[adj[0]] + s[adj[1]] + s[adj[2]] + s[adj[3]])


@jit(nopython=True)
def step_and_a_step(n, ex, s, vizin):
    for i in range (n):
        spin = rd.randint(0, n - 1)
        de = int(s[spin]*vsum(s,vizin[spin])*0.5+2)
        P = ex[de]
        r = rd.random()
        if (r <= P):
            s[spin] = -s[spin]
    return s


def metropolis(n_max, n_it, beta):
    s = np.zeros(n_max, dtype = int)
    for i in range(n_max - 1):
        random_value = rd.choice([1, -1])
        s[i] = random_value
    vizin = vizinhos(n_max)
    energ = energia(s, vizin)
    mag = sum(s)
    ex = expos(beta)
    states_energ = []
    states_mag = []
    states_energ.append(energ)
    states_mag.append(mag)
    for i in range (n_it):
        s = step_and_a_step(n_max, ex, s, vizin)
        energ = energia(s, vizin)
        mag = sum(s)
        states_energ.append(energ)
        states_mag.append(mag)
    return(states_energ, states_mag)


def run_metropolis(n, L, n_it, temp):
    n_max = L*L
    beta = 1/temp
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for i in range (n):
        (en, mg) = metropolis(n_max, n_it, beta)
        ax1.plot(en)
        ax2.plot(mg)
    plt.show(block=True)


#Definição de parâmetros:
rodar_n_vezes = 20
tamanho_rede = 32
passos_mc = 5000
temperatura = 1.5

#Chamada do modelo:
run_metropolis(rodar_n_vezes, tamanho_rede, passos_mc, temperatura)
