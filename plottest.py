import numpy as np
import matplotlib.pyplot as plt

def calcSigmoid(x,alphaProbability=0.1,nearSensitivity=10**2):
    return [max(0.5, x) for x in ((1 / (1 + np.exp(-nearSensitivity * (x ** 2)))) - alphaProbability)]

x = np.linspace(-1, 1, 1000)
plt.plot(x, calcSigmoid(x), color='red')
plt.grid()
plt.yticks(np.arange(0, 1.1, 0.1))
plt.show()