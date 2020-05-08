#!/usr/bin/env python3
from numpy import array, reshape, linalg, argmin, append, ones, loadtxt, mean, subtract, var, std, max, min, delete
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def calibrate_camera_3d(data):
    world_points = data[:, :3]  # Gets the first 3 points (x, y, z) of real world
    image_points = data[:, 3:]  # Gets the last two (X,Y) of 2d world
    a = []  # Initialise A matrix

    for i in range(len(data)):
        w_x = world_points[i][0]
        w_y = world_points[i][1]
        w_z = world_points[i][2]

        i_x = image_points[i][0]
        i_y = image_points[i][1]

        x = [w_x, w_y, w_z, 1, 0, 0, 0, 0, -i_x * w_x, -i_x * w_y, -i_x * w_z, -i_x]
        y = [0, 0, 0, 0, w_x, w_y, w_z, 1, -i_y * w_x, -i_y * w_y, -i_y * w_z, -i_y]

        a.append((x, y))

    a = array(a)
    a = reshape(a, (982, 12))

    d, v = linalg.eig(a.transpose().dot(a))

    est = v[:, argmin(d)]
    est = reshape(est, (3, 4))  # (12,) -> (3,4) for transpose below

    return est


def visualise_camera_calibration_3d(data, p):
    three_d_homogenous = append(data[:, :3], ones([len(data), 1]), 1)  # Make homogenous
    image_points = data[:, 3:]

    three_d_homogenous = p.dot(three_d_homogenous.transpose())
    overall = three_d_homogenous.transpose()

    overall[:, 0] /= overall[:, 2]  # x = x/z
    overall[:, 1] /= overall[:, 2]  # y = y/z

    # Plot the re-projected 2D points.
    fig = plt.figure("Camera calibration", figsize=(50, 50))
    ax = fig.gca()  # Automatic points
    ax.plot(overall[:, 0], overall[:, 1], 'b.')  # The re-projection against the old 2D points.
    ax.plot(image_points[:, 0],  image_points[:, 1], 'r.')
    plt.show()

    return overall


def evaluate_camera_calibration_3d(data, p):

    image_points = append(data[:, 3:], ones([len(data), 1]), 1)

    dist = abs(subtract(image_points, p))
    print(f"Avg distance: {mean(dist)}")
    print(f"Variance: {var(dist)}")

    print(f"Max distance: {max(dist)}")
    print(f"Min distance: {min(dist)}")


data = loadtxt('data.txt')
p = calibrate_camera_3d(data)
p = visualise_camera_calibration_3d(data, p)
evaluate_camera_calibration_3d(data, p)
