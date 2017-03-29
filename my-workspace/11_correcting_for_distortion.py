import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in a calibration image
img = mpimg.imread('../calibration_wide/GOPR0032.jpg')
plt.imshow(img)
#plt.show()

# Arrays to store object points and image points from all the images
objpoints = [] #3D points in real world space
imgpoints = [] #2d points in image plane

# Prepare oject points, like (0,0,0), (1,0,0), (2,0,0) ...., (7,5,0)
objp = np.zeros((6*8, 3), np.float32)
objp[:,:2]  = np.mgrid[0:8, 0:6].T.reshape(-1,2) #x, y coordinates

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
# If corners are found, add object points, image points
if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)

    # draw and sisplay the corners
    img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
    plt.imshow(img)
    plt.show()
