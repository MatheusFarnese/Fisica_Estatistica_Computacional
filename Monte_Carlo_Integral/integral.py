import numpy as np
import matplotlib.pyplot as plt
import random as rm
import math

plt.rcParams['figure.figsize']  = (8, 5)
plt.rcParams['axes.labelsize']  = 20
plt.rcParams['axes.titlesize']  = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['lines.linewidth'] = 4
results_all_mean = np.zeros(9)
results_all_std = np.zeros(9)

num_it = 100
result_hist = np.zeros(1000)
result = np.zeros(num_it)
for n in range (1000):
    result = np.zeros(num_it)
    for i in range (num_it):
        result[i] = (1 - (rm.random())**2)
    result_hist[n] = result.mean()
plt.hist(result_hist)
plt.axvline(x = 0.66666666, color = 'r')
plt.show()
results_all_mean[0] = result_hist.mean()
results_all_std[0] = result_hist.std()

num_it = 1000
result_hist = np.zeros(1000)
result = np.zeros(num_it)
for n in range (1000):
    result = np.zeros(num_it)
    for i in range (num_it):
        result[i] = (1 - (rm.random())**2)
    result_hist[n] = result.mean()
plt.hist(result_hist)
plt.axvline(x = 0.66666666, color = 'r')
plt.show()
results_all_mean[1] = result_hist.mean()
results_all_std[1] = result_hist.std()

num_it = 10000
result_hist = np.zeros(1000)
result = np.zeros(num_it)
for n in range (1000):
    result = np.zeros(num_it)
    for i in range (num_it):
        result[i] = (1 - (rm.random())**2)
    result_hist[n] = result.mean()
plt.hist(result_hist)
plt.axvline(x = 0.66666666, color = 'r')
plt.show()
results_all_mean[2] = result_hist.mean()
results_all_std[2] = result_hist.std()

num_it = 100
result_hist = np.zeros(1000)
result = np.zeros(num_it)
for n in range (1000):
    result = np.zeros(num_it)
    for i in range (num_it):
        result[i] = ((math.e)**(rm.random()))
    result_hist[n] = result.mean()
plt.hist(result_hist)
plt.axvline(x = (math.e - 1), color = 'r')
plt.show()
results_all_mean[3] = result_hist.mean()
results_all_std[3] = result_hist.std()

num_it = 1000
result_hist = np.zeros(1000)
result = np.zeros(num_it)
for n in range (1000):
    result = np.zeros(num_it)
    for i in range (num_it):
        result[i] = ((math.e)**(rm.random()))
    result_hist[n] = result.mean()
plt.hist(result_hist)
plt.axvline(x = (math.e - 1), color = 'r')
plt.show()
results_all_mean[4] = result_hist.mean()
results_all_std[4] = result_hist.std()

num_it = 10000
result_hist = np.zeros(1000)
result = np.zeros(num_it)
for n in range (1000):
    result = np.zeros(num_it)
    for i in range (num_it):
        result[i] = ((math.e)**(rm.random()))
    result_hist[n] = result.mean()
plt.hist(result_hist)
plt.axvline(x = (math.e - 1), color = 'r')
plt.show()
results_all_mean[5] = result_hist.mean()
results_all_std[5] = result_hist.std()

num_it = 100
result_hist = np.zeros(1000)
result = np.zeros(num_it)
for n in range (1000):
    result = np.zeros(num_it)
    for i in range (num_it):
        result[i] = math.pi*((math.sin(rm.uniform(0, math.pi)))**2)
    result_hist[n] = result.mean()
plt.hist(result_hist)
plt.axvline(x = (math.pi/2), color = 'r')
plt.show()
results_all_mean[6] = result_hist.mean()
results_all_std[6] = result_hist.std()

num_it = 1000
result_hist = np.zeros(1000)
result = np.zeros(num_it)
for n in range (1000):
    result = np.zeros(num_it)
    for i in range (num_it):
        result[i] = math.pi*((math.sin(rm.uniform(0, math.pi)))**2)
    result_hist[n] = result.mean()
plt.hist(result_hist)
plt.axvline(x = (math.pi/2), color = 'r')
plt.show()
results_all_mean[7] = result_hist.mean()
results_all_std[7] = result_hist.std()

num_it = 10000
result_hist = np.zeros(1000)
result = np.zeros(num_it)
for n in range (1000):
    result = np.zeros(num_it)
    for i in range (num_it):
        result[i] = math.pi*((math.sin(rm.uniform(0, math.pi)))**2)
    result_hist[n] = result.mean()
plt.hist(result_hist)
plt.axvline(x = (math.pi/2), color = 'r')
plt.show()
results_all_mean[8] = result_hist.mean()
results_all_std[8] = result_hist.std()

print("Primeiro experimento(mean): ", results_all_mean[0], results_all_mean[1], results_all_mean[2])
print("Primeiro experimento (std): ", results_all_std[0], results_all_std[1], results_all_std[2])
print("Segundo experimento (mean): ", results_all_mean[3], results_all_mean[4], results_all_mean[5])
print("Segundo experimento  (std): ", results_all_std[3], results_all_std[4], results_all_std[5])
print("Terceiro experimento(mean): ", results_all_mean[6], results_all_mean[7], results_all_mean[8])
print("Terceiro experimento (std): ", results_all_std[6], results_all_std[7], results_all_std[8])

1 / (((rm.random() + rm.random())*rm.random()) + ((rm.random() + rm.random())*rm.random()) + ((rm.random() + rm.random())*rm.random()))

num_it = 10000
result_hist = np.zeros(1000)
result = np.zeros(num_it)
for n in range (1000):
    result = np.zeros(num_it)
    for i in range (num_it):
        result[i] = 1 / (((rm.random() + rm.random())*rm.random()) + ((rm.random() + rm.random())*rm.random()) + ((rm.random() + rm.random())*rm.random()))
    result_hist[n] = result.mean()
plt.hist(result_hist)
plt.show()

print(result_hist.mean())
print(result_hist.std())
