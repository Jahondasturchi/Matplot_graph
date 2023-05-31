import numpy as np
import matplotlib.pyplot as plt
x = np.array(["A", "B", "C", "D"])
y = np.array([3, 6, 1,8])
#plt.bar(x,y, width=0.2) # vertical bar
plt.barh(x,y, color = "red", height=0.1) # it is a horizontal bar

plt.show()