import matplotlib.pyplot as plt
import numpy as np
#x = np.array([0,6])
# y = np.array([0,250])
# plt.plot(x,y)
# # plt.show()
# y = np.array([3,5,6,7,8,1,3,4,5,6,7])
# plt.plot(y, marker = 'd')
# plt.show()
x = np.array([2,2,4,5,6,7,9])
a= np.array([2,5,4,8,6,7,0])
y = np.array([1,2,4,5,6,7,8])
plt.plot(x,y, a,linestyle = '--', color = 'r', linewidth = '7')
plt.plot(x)
plt.show()