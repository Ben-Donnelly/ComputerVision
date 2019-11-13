'''Visualises the data file for cs410 camera calibration assignment
To run: %run LoadCalibData.py
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def calibrateCamera3D(data):
    world_points = data[:, :3]  # Gets the first 3 points (x, y, z) of real world
    image_points = data[:, 3:]  # Gets the last two (X,Y) of 2d world
    A = []

    for i in range(len(data)):
        worldX = world_points[i][0]
        worldY = world_points[i][1]
        worldZ = world_points[i][2]

        imageX = image_points[i][0]
        imageY = image_points[i][1]

        X = [worldX, worldY, worldZ, 1, 0, 0, 0, 0, -imageX * worldX, -imageX * worldY, -imageX * worldZ, -imageX]
        Y = [0, 0, 0, 0, worldX, worldY, worldZ, 1, -imageY * worldX, -imageY * worldY, -imageY * worldZ, -imageY]

        A.append(X)
        A.append(Y)

    A = np.asarray(A)
    D, V = np.linalg.eig(A.transpose().dot(A))

    est = V[:, np.argmin(D)]
    print(est)

data = np.loadtxt('data.txt')

# fig = plt.figure()
# ax = fig.gca(projection="3d")
# ax.plot(data[:,0], data[:,1], data[:,2],'k.')
#
# fig = plt.figure()
# ax = fig.gca()
# ax.plot(data[:,3], data[:,4],'r.')

# plt.show()
calibrateCamera3D(data)

