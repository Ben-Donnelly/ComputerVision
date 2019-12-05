""" Example of using OpenCV API to detect and draw checkerboard pattern"""
import numpy as np
import cv2
from itertools import product, permutations, repeat
# These two imports are for the signal handler
import signal
import sys


#### Some helper functions #####
def reallyDestroyWindow(windowName):
    ''' Bug in OpenCV's destroyWindow method, so... '''
    ''' This fix is from http://stackoverflow.com/questions/6116564/ '''
    cv2.destroyWindow(windowName)
    for i in range(1, 5):
        cv2.waitKey(1)


def shutdown():
    ''' Call to shutdown camera and windows '''
    global cap
    cap.release()
    reallyDestroyWindow('img')


def signal_handler(signal, frame):
    ''' Signal handler for handling ctrl-c '''
    shutdown()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


##########

############## calibration of plane to plane 3x3 projection matrix

def compute_homography(fp, tp):
    tp = np.asarray(tp)
    # print(f"First fp = {fp}\n{type(fp)}\nSecond tp {tp}\n{type(tp)}")
    # quit()
    ''' Compute homography that takes fp to tp.
    fp and tp should be (N,3) '''

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # create matrix for linear method, 2 rows for each correspondence pair
    num_corners = fp.shape[0]

    # construct constraint matrix
    A = np.zeros((num_corners * 2, 9))
    A[0::2, 0:3] = fp
    A[1::2, 3:6] = fp
    A[0::2, 6:9] = fp * -np.repeat(np.expand_dims(tp[:, 0], axis=1), 3, axis=1)
    A[1::2, 6:9] = fp * -np.repeat(np.expand_dims(tp[:, 1], axis=1), 3, axis=1)

    # solve using *naive* eigenvalue approach
    D, V = np.linalg.eig(A.transpose().dot(A))

    H = V[:, np.argmin(D)].reshape((3, 3))

    # normalise and return
    return H


##############


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# YOU SHOULD SET THESE VALUES TO REFLECT THE SETUP
# OF YOUR CHECKERBOARD
WIDTH = 9
HEIGHT = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((WIDTH * HEIGHT, 3), np.float32)
objp[:, :2] = np.mgrid[0:HEIGHT, 0:WIDTH].T.reshape(-1, 2)

cap = cv2.VideoCapture(0)






## Step 0: Load the image you wish to overlay
myImage = cv2.imread(r"C:\Users\Ben Donnelly\PycharmProjects\MyFlaskApp\Flask\uploads\testforflaskupload.jpg",1)

x = []
y = []
for i in range(640, -1, -80): # 640 / 8
    x.append(i)
for i in range(0, 481, 96): # 480 / 5
    y.append(i)
# print(x, y)
d = []
for i in range(len(x)):
    # print(x[i])
    for j in range(len(y)):
        d.append([x[i], y[j], 1])
        # print()
d = np.asarray(d, dtype=np.float32)
# print(d, type(d), d.shape)
# quit()

# quit()
# grey = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
# print(grey, grey.shape)
# quit()
# cv2.imshow('image',myImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# quit()

while (True):
    # capture a frame
    ret, img = cap.read()

    ## IF YOU WISH TO UNDISTORT YOUR IMAGE YOU SHOULD DO IT HERE

    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (HEIGHT, WIDTH), None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        # print(gray, corners)
        # print(type(gray), type(corners))
        # print(gray.shape, corners.shape)
        # quit()
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, (HEIGHT, WIDTH), corners, ret)

        ## Step 1a: Compute fp -- an Nx3 array of the 2D homogeneous coordinates of the
        ## detected checkerboard corners
        corners = np.asmatrix(corners)

        # fp = zeros(shape=(len(corners),3))
        # print(fp, type(fp), fp.shape)
        # print(corners, type(corners), corners.shape)
        fp = (np.append(corners, np.ones([len(corners), 1]), 1) ) # Make homogenous
        # quit()
        # for i in range(len(corners)):
        #     print(fp[i])
        #     print(append(corners[i], 1))
        #     quit()
            # fp[i] = append(corners[i], 1) ####
        # print(fp, fp.shape, type(fp))

        # print(myImage[0:h])

        # for i in range(len(corners)):
        #     d = append(corners[i], 1)
        # print(d)
        # quit()


        ## Step 1b: Compute tp -- an Nx3 array of the 2D homogeneous coordinates of the
        # print(type(d))
        d = np.asarray(d)
        # print(type(d))
        ## samples of the image coordinates
        ## Note: this could be done outside of the loop either!



        ## Step 2: Compute the homography from tp to fp
        # quit()
        fp = np.asmatrix(fp)
        # print(fp, type(fp), fp.shape)
        # quit()
        h = compute_homography(d, fp)
        # quit()



        ## Step 3: Compute warped mask image

        d = np.asmatrix(d)

        myImage[:, 2] = 0
        h = np.asarray(h)
        wpimg = cv2.warpPerspective(myImage, h, dsize=(590, 428))
        #
        # ## Step 4: Compute warped overlay image
        #
        rows, cols, chans = wpimg.shape
        roi = img[0:rows, 0:cols]
        # # convert it to gray scale
        graysrc = cv2.cvtColor(wpimg, cv2.COLOR_BGR2GRAY)
        # # creating the mask, and the it's inverse
        ret, mask = cv2.threshold(graysrc, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # # black out the area in ROI, this is where the imposed image would go
        # print(roi, type(roi))
        # quit()
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # # i take only the region of our 'src' image
        img2_fg = cv2.bitwise_and(wpimg, wpimg, mask=mask)
        # # puts the two together and the modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        img[0:rows, 0:cols] = dst

        ## Step 5: Compute final image by combining the warped frame with the captured frame

    cv2.imshow('img', img)

    if cv2.waitKey(1)  == 27:
        break

# release everything
shutdown()

'''
[[[0 0 0]
  [0 0 0]
  [0 0 0]
  ...
  [0 0 0]
  [0 0 0]
  [0 0 0]]

 [[0 0 0]
  [0 0 0]
  [0 0 0]
  ...
  [0 0 0]
  [0 0 0]
  [0 0 0]]

 [[0 0 0]
  [0 0 0]
  [0 0 0]
  ...
  [0 0 0]
  [0 0 0]
  [0 0 0]]

 ...

 [[0 0 0]
  [0 0 0]
  [0 0 0]
  ...
  [0 0 0]
  [0 0 0]
  [0 0 0]]

 [[0 0 0]
  [0 0 0]
  [0 0 0]
  ...
  [0 0 0]
  [0 0 0]
  [0 0 0]]

 [[0 0 0]
  [0 0 0]
  [0 0 0]
  ...
  [0 0 0]
  [0 0 0]
  [0 0 0]]] <class 'numpy.ndarray'>

'''