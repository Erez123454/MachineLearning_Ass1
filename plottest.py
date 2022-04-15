import matplotlib.pyplot as plt
import numpy as np

# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
ax1.plot(x, y,label='1')
ax1.plot(x, y+4,label='1')
ax2.plot(x, y ** 2,label='1')

plt.legend()

plt.show()