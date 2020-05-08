#!/usr/bin/env python
""" Example of using OpenCV API to detect and draw checkerboard pattern"""
from numpy import asarray, zeros, repeat, expand_dims, linalg, argmin, float32, mgrid, asarray, asmatrix, append, ones
from cv2 import TERM_CRITERIA_EPS, imshow, VideoCapture, imread, cvtColor, COLOR_BGR2GRAY, THRESH_BINARY, \
    findChessboardCorners, cornerSubPix, TERM_CRITERIA_MAX_ITER, drawChessboardCorners, warpPerspective, threshold,\
    bitwise_and, bitwise_not, add, waitKey, destroyAllWindows


def compute_homography(fp, tp):
    # Compute homography that takes fp to tp.fp and tp should be (N,3)
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # create matrix for linear method, 2 rows for each correspondence pair
    num_corners = fp.shape[0]

    # construct constraint matrix
    a = zeros((num_corners * 2, 9))
    a[0::2, 0:3] = fp
    a[1::2, 3:6] = fp
    a[0::2, 6:9] = fp * -repeat(expand_dims(tp[:, 0], axis=1), 3, axis=1)
    a[1::2, 6:9] = fp * -repeat(expand_dims(tp[:, 1], axis=1), 3, axis=1)

    # solve using *naive* eigenvalue approach
    d, v = linalg.eig(a.transpose().dot(a))

    # normalise and return
    h = v[:, argmin(d)]

    return h.reshape((3, 3))


# termination criteria
criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)

# YOU SHOULD SET THESE VALUES TO REFLECT THE SETUP
# OF YOUR CHECKERBOARD
# NOTE: If image appears to be coming "towards you" / out of the screen
# Your numbers here could be backwards/wrong, try swap them and make sure you're using the correct numbers
# The numbers are based off the number of squares on you board
WIDTH = 9
HEIGHT = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = zeros((WIDTH * HEIGHT, 3), float32)
objp[:, :2] = mgrid[0:HEIGHT, 0:WIDTH].T.reshape(-1, 2)

vid = VideoCapture(0)

# Step 0: Load the image you wish to overlay
myImage = imread(r"C:\Users\Ben Donnelly\PycharmProjects\MyFlaskApp\Flask\uploads\testforflaskupload.jpg", 1)

# Write in pythonic way
x = [i for i in range(640, -1, -80)]
y = [i for i in range(0, 481, 96)]

# Again, more pythonic
fp = asarray([(x[i], y[j], 1) for i in range(len(x)) for j in range(len(y))], dtype=float32)

while True:
    ret, cap = vid.read()

    grey = cvtColor(cap, COLOR_BGR2GRAY)

    retu, corners = findChessboardCorners(grey, (HEIGHT, WIDTH))

    # If pattern detected, substitute cap in for squares
    if retu:
        cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)

        # Step 1a: Compute fp -- an Nx3 array of the 2D homogeneous coordinates of the
        # detected checkerboard corners
        corners = asmatrix(corners)

        tp = append(corners, ones([len(corners), 1]), 1)  # Make homogeneous

        # Step 2: Compute the homography from tp to fp
        h = asarray(compute_homography(asarray(fp), asarray(tp)))

        # Step 3: Compute warped mask image
        fp = asmatrix(fp)

        # size of any dynamic image
        warped = warpPerspective(myImage, h, dsize=(cap.shape[1], cap.shape[0]))

        # Step 4: Compute warped overlay image
        r, c = warped.shape[0], warped.shape[1]
        roi = cap[:r, :c]

        grey = cvtColor(warped, COLOR_BGR2GRAY)

        # mask and inverse
        retu, m = threshold(grey, 0, 255, THRESH_BINARY)
        mask = bitwise_not(m)

        # get roi on board
        src1 = bitwise_and(roi, roi, mask=mask)

        # place cap on roi
        src2 = bitwise_and(warped, warped, mask=m)

        # construct
        cap[:r, :c] = add(src1, src2)

        # Step 5: Compute final image by combining the warped frame with the captured frame
    imshow('Augmented picture', cap)

    if waitKey(1) == 27:  # esc
        break

destroyAllWindows()