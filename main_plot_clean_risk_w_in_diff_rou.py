'''
    the code to show the 3D data use the sample code show on matplotlib 
    web (https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#wireframe-plots)
'''

from readFileToArray import readTextFile

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# file_name_to_plot = '10w.txt'
file_name_to_plot = '2.txt'
# file_name_to_plot = 'clean_risk.txt' 
# file_name_to_plot = 'clean_risk_0.01.txt' 
data = readTextFile(file_name_to_plot)
print data
print 'clean data size: ', len(data)


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0.05, 0.355, 0.05)
Y = np.arange(0.05, 0.355, 0.05)

# X = np.arange(0.05, 0.35, 0.01)
# Y = np.arange(0.05, 0.35, 0.01)

X, Y = np.meshgrid(X, Y)
Z = data

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('rou_ratio')
ax.set_ylabel('rou_ratio')
ax.set_zlabel('risk')

plt.show()