import numpy as np
import cv2 as cv
import glob

checkerboard = (13, 9)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:checkerboard[0],0:checkerboard[1]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./camera_img/*.jpg')
for fname in images:
    print("fname : ", fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, checkerboard, None)
    # If found, add object points, image points (after refining them)
    print("ret :", ret)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, checkerboard, corners2, ret)
        cv.imwrite('./out/' + fname,img)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()

h, w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("camera matrix: ")
print(mtx)
# print("Dist : ")
# print(dist)
# print("rvecs : ")
# print(rvecs)
# print("tvecs : ")
# print(tvecs)
