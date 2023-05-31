import numpy as np
import matplotlib.pyplot as plt
# x = np.array([2,3,4,5,6,7,8,9])

# plt.hist(x)
# plt.show()
x = np.array([35,25,25,15])
mylabels = ["A","B","C","D"]
myexplode = [0.2, 0, 0, 0]
plt.pie(x, labels=mylabels, startangle=90, explode=myexplode)
plt.show()