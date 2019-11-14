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
    est = np.reshape(est, (3,4))
    return est
    # print(est)

def visualiseCameraCalibration3D(data, P):
    threeD_homogenous = np.append(data[:, :3],np.ones([len(data),1]),1)
    twoD_homogenous = np.append(data[:, 3:], np.ones([len(data), 1]), 1)

    """
    Get the re-projection matrix by getting the dot product of the Camera
    matrix, P, and the transpose of the 3D points, then transposing the
    result.
    """

    '''
    [ 8.65064148e-03  1.07863245e-03 -3.87162696e-03  9.98554504e-01
  6.27447267e-05  9.18884379e-03  4.63678386e-04 -5.20172279e-02
  6.21374751e-06  2.17125723e-06  6.39659086e-06  2.73310960e-03]

    '''

    # print(threeD_homogenous.shape, P.shape)
    # quit()
    threeD_homogenous = P.dot(threeD_homogenous.transpose())
    final_points = threeD_homogenous.transpose()

    # need to change below here
    final_points[:,0] = final_points[:,0] / final_points[:,2] # X = X/Z
    final_points[:,1] = final_points[:,1] / final_points[:,2] # Y = Y/Z

    # Plot the re-projected 2D points.
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(final_points[:,0], final_points[:,1], 'g.')
    # The reprojection againest the old 2D points.
    ax.plot(data[:,3], data[:,4],'r.')
    plt.show()

data = np.loadtxt('data.txt')
P = calibrateCamera3D(data)
visualiseCameraCalibration3D(data,P)
