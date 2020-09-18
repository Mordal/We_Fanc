'''
=================
3D wireframe plot
=================

A very basic demonstration of a wireframe plot.
'''

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
#X, Y, Z = axes3d.get_test_data(1)
X = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
Y = np.array([[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]])
Z = np.array([[-10,20,-10,-10],[-10,-10,-10,-10],[10,10,10,10],[20,20,20,20]])



# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z)

plt.show()
