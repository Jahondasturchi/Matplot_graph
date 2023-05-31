import numpy as np
import matplotlib.pyplot as plt

x = np.array([25,15,10,30,20])
mylabels = ["olma", 'banan', "kartoshka", 'piyoz', 'qalampir']

plt.pie(x, labels=mylabels)
plt.legend(title = 'Sabzavotlar')
plt.show()