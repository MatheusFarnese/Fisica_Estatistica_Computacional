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


def thermalize(L2, n_it, beta):
    s = np.zeros(L2, dtype = int)
    for i in range(L2 - 1):
        random_value = rd.choice([1, -1])
        s[i] = random_value
    vizin = vizinhos(L2)
    ex = expos(beta)
    for i in range (n_it):
        s = step_and_a_step(L2, ex, s, vizin)
    return(s)


def ising(N_mcs, n_blocos, n_term, temp_inicial, L2, n_pontos, delta_t):
    temp = temp_inicial
    vizin = vizinhos(L2)
    heat_list = []
    heat_err = []
    sus_list = []
    sus_err = []
    energ_spin_list = []
    energ_spin_err = []
    mag_spin_list = []
    mag_spin_err = []
    for i in range (n_pontos):
        beta = 1 / temp
        s = thermalize (L2, n_term, beta)
        ex = expos(beta)
        energ = 0
        mag = 0
        blocos_energ = []
        blocos_mag = []
        heat_blocos = []
        sus_blocos = []
        for i in range (n_blocos):
            states_energ = []
            states_mag = []
            energ_spin_blocos = []
            mag_spin_blocos = []
            for j in range (int(N_mcs/n_blocos)):
                s = step_and_a_step(L2, ex, s, vizin)
                energ = energia(s, vizin)
                mag = sum(s)
                states_energ.append(energ)
                states_mag.append(abs(mag))

            E = np.array(states_energ)
            M = np.array(states_mag)

            energ_square = E**2
            mag_square = M**2
            energ_mean = E.mean()
            mag_mean = M.mean()
            energ_square_mean = energ_square.mean()
            mag_square_mean = mag_square.mean()
            energ_mean_square = energ_mean**2
            mag_mean_square = mag_mean**2
            heat = ((beta*beta) / L2) * (energ_square_mean - energ_mean_square)
            sus = (beta / L2) * (mag_square_mean - mag_mean_square)
            heat_blocos.append(heat)
            sus_blocos.append(sus)
            energ_spin_blocos.append(energ_mean / L2)
            mag_spin_blocos.append(mag_mean / L2)

        he = np.array(heat_blocos)
        su = np.array(sus_blocos)
        en = np.array(energ_spin_blocos)
        ma = np.array(mag_spin_blocos)

        heat_list.append(he.mean())
        sus_list.append(su.mean())
        energ_spin_list.append(en.mean())
        mag_spin_list.append(ma.mean())

        heat_err.append(he.std()/np.sqrt(n_blocos))
        sus_err.append(su.std()/np.sqrt(n_blocos))
        energ_spin_err.append(en.std()/np.sqrt(n_blocos))
        mag_spin_err.append(ma.std()/np.sqrt(n_blocos))

        temp = temp + delta_t

    return(heat_list, sus_list, energ_spin_list, mag_spin_list, heat_err, sus_err, energ_spin_err, mag_spin_err)


def run_ising(N_mcs, n_blocos, n_term, temp_inicial, L, n_pontos, delta_t):
    L2 = L*L
    beta = 1/temp_inicial
    (heat, sus, en, mg, heat_e, sus_e, en_e, mg_e) = ising(N_mcs, n_blocos, n_term, temp_inicial, L2, n_pontos, delta_t)
    temp_values = [0.1 * i for i in range(60)]

    plt.errorbar(temp_values, heat, yerr=heat_e, fmt='o-', capsize=3)
    plt.title('Calor específico X Temperatura')
    plt.xlabel('Temperatura')
    plt.ylabel('Calor específico')
    plt.show(block=True)

    plt.errorbar(temp_values, sus, yerr=sus_e, fmt='o-', capsize=3)
    plt.title('Susceptibilidade magnética X Temperatura')
    plt.xlabel('Temperatura')
    plt.ylabel('Susceptibilidade magnética')
    plt.show(block=True)

    plt.errorbar(temp_values, en, yerr=en_e, fmt='o-', capsize=3)
    plt.title('Energia por spin X Temperatura')
    plt.xlabel('Temperatura')
    plt.ylabel('Energia por spin')
    plt.show(block=True)

    plt.errorbar(temp_values, mg, yerr=mg_e, fmt='o-', capsize=3)
    plt.title('Magnetização por spin X Temperatura')
    plt.xlabel('Temperatura')
    plt.ylabel('Magnetização por spin')
    plt.show(block=True)

#Definição de parâmetros:
passos_mc = 100000
numero_blocos = 100
passos_termalizacao = 5000
temperatura_inicial = 0.1
tamanho_rede = 20
numero_pontos_grafico = 60
variacao_de_temperatura_a_cada_ponto = 0.1

#Chamada do modelo:
run_ising(passos_mc, numero_blocos, passos_termalizacao, temperatura_inicial, tamanho_rede, numero_pontos_grafico, variacao_de_temperatura_a_cada_ponto)
