import numpy as np
import matplotlib.pyplot as plt
x = np.array([0,1,2,3,4,5])
y = np.array([4,5,6,7,8,9])
plt.subplot(1,2,2)
plt.plot(x,y)
plt.title('Income')


x = np.array([3,4,5,6,7])
y = np.array([7,8,9,10,11])

plt.subplot(1,2,1)
plt.plot(x,y)
plt.title('sales')


plt.suptitle("Myshop")
plt.show()