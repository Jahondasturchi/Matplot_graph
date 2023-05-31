import numpy as np
import matplotlib.pyplot as plt

i = -8
a = []
b = []
while(i<=8):
    a.append(i)
    b.append(i**2)
    i=i+0.1

x = np.array(a)

y = np.array(b)
font = {'family':'arial','color':'red','size':20}
plt.title("Sport Watch data", loc='right')
plt.xlabel("Average pulse", fontdict=font)
plt.ylabel("calorie burnage",fontdict=font)
plt.plot(x, y)
plt.grid( color = 'r', linestyle='--', linewidth = 0.5)
plt.show()
