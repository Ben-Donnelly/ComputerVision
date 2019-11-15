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

        A.append((X,Y))

    A = np.asarray(A)
    A = np.reshape(A, (982, 12)) # If getting error may need to seperate append above

    D, V = np.linalg.eig(A.transpose().dot(A))

    est = V[:, np.argmin(D)]

    est = np.reshape(est, (3,4)) # (12,) -> (3,4) for transpose below
    return est
    # print(est)

def visualiseCameraCalibration3D(data, P):
    threeD_homogenous = np.append(data[:, :3],np.ones([len(data),1]),1)
    # twoD_homogenous = np.append(data[:, 3:], np.ones([len(data), 1]), 1)
    image_points = data[:, 3:]
    """
    Get the re-projection matrix by getting the dot product of the Camera
    matrix, P, and the transpose of the 3D points, then transposing the
    result.
    """


    # print(threeD_homogenous.shape, P.shape)
    # quit()
    threeD_homogenous = P.dot(threeD_homogenous.transpose())
    overall = threeD_homogenous.transpose()

    # need to change below here

    overall[:,0] /= overall[:,2] # X = X/Z
    overall[:,1] /= overall[:,2] # Y = Y/Z

    # Plot the re-projected 2D points.
    fig = plt.figure("Camera calibration", figsize=(50, 50))
    ax = fig.gca() # Automatic points
    ax.plot(overall[:,0], overall[:,1], 'b.')
    # The reprojection againest the old 2D points.

    ax.plot(image_points[:, 0],  image_points[:, 1], 'r.') # could just use image points
    plt.show()

data = np.loadtxt('data.txt')
P = calibrateCamera3D(data)
visualiseCameraCalibration3D(data,P)
