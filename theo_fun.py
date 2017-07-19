

import matplotlib.pyplot as plt
import numpy as np
import math

lamb_ts = np.linspace(0.,50.,num=20)
L = 20

funs = []
for lamb_t in lamb_ts:
    curr_fun = 0.
    for l in range(L):
        curr_fun += lamb_t**l/math.factorial(l)
    curr_fun /= np.exp(lamb_t)
    curr_fun = 1 - curr_fun
    funs.append(curr_fun)

a, b = .4, L
logits = 1./(1+np.exp(-a*(lamb_ts-b)))

plt.plot(lamb_ts, funs, 'b')
plt.plot(lamb_ts, logits, 'r')
plt.show()