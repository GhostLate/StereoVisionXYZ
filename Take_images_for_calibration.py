import os
import numpy as np
import cv2
import time
from multiprocessing import Pool

# FRAME_WIDTH = 640
# FRAME_HEIGHT = 480

FRAME_WIDTH = 1280
FRAME_HEIGHT = 960

def find_chess(gray):
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)  # Define the number of chess corners (here 9 by 6) we are looking for with the right Camera
  
    if (ret == True):
        criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        corners2 = cv2.cornerSubPix(gray,corners, (11, 11), (-1, -1), criteria)    # Refining the Position

        cv2.drawChessboardCorners(gray, (9,6), corners2, ret)
        return (1, gray)
    else:
        return (0, gray)

imgID = 0
import os
import numpy as np
import cv2

print('Starting the Calibration. Press and maintain the space bar to exit the script\n')
print('Push (s) to save the image you want and push (c) to see next frame without saving the image')

id_image=0

# termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


devpath1 = ""
devpath2 = ""

try:
    for path in os.listdir('/dev/v4l/by-id/'):
        if path.find('index0') >= 0 and path.find('299B3041') >= 0:
            devpath2 = os.path.join('/dev/v4l/by-id/', path)

        elif path.find('index0') >= 0 and path.find('299B3065') >= 0:
            devpath1 = os.path.join('/dev/v4l/by-id/', path)
except OSError as Err:
    print ("Device isn't found! Check Connection")


print (devpath1)
print (devpath2)

if devpath1 == "":
    print ("Error, devpath1 is none!")
    exit()
if devpath2 == "":
    print ("Error, devpath2 is none!")
    exit()

CamR = cv2.VideoCapture(devpath1)
CamL = cv2.VideoCapture(devpath2)

CamL.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if (CamL.isOpened() and CamR.isOpened()) == False:
    print("Cameras is opened: " + CamL.isOpened() and CamR.isOpened())
    exit()

i = int(input('2 - need to calibrate, 1 - find chess, 0 - just watch: '))
if i == 1 or i == 0:
    pool = Pool(2)
    while True:
        retR, frameR = CamR.read()
        retL, frameL = CamL.read()
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

        #cv2.namedWindow('stereo', 0)
        #cv2.resizeWindow('stereo', 1280, 480)
        cv2.imshow("stereo", np.hstack([frameL, frameR]))
        if i == 1:
            start_time = time.time()
            (data0, data1) = pool.map(find_chess, [grayL, grayR])
            #print ('Done! Time was taken: ' + format(time.time() - start_time))
            if (data0[0] == 1 and data1[0] == 1):

                #cv2.namedWindow('stereo_chess', 0)
                #cv2.resizeWindow('stereo_chess', 1280, 480)
                cv2.imshow("stereo_chess", np.hstack([data0[1], data1[1]]))

                if cv2.waitKey(0) & 0xFF == ord('s'):
                    print('Images ID: ' + str(imgID) + ' saved for right and left cameras')
                    cv2.imwrite('img/right' + str(imgID) + '.png', frameR) 
                    cv2.imwrite('img/left' + str(imgID) + '.png', frameL)
                    imgID = imgID + 1
                else:
                    print("Images hadn't saved")

        if cv2.waitKey(1) & 0xFF == ord(' '): 
            break

    pool.close()
    pool.join()
    cv2.destroyAllWindows()  

if (i != 2):    
    if int(input("Need to calibrate (1/0 - yes/no): ")) == 0:
        exit()
    
# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9 * 6, 3), np.float32)
objp[ : , : 2] = np.mgrid[0 : 9, 0 : 6].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []   # 3d points in real world space
imgpointsR = []   # 2d points in image plane
imgpointsL = []
# Start calibration from the camera
print('Starting calibration 2 cameras... ')
# Call all saved images
for i in range(0, int(input("Amount of images in img folder: ")) + 1):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    ChessImaR = cv2.imread('img/right' + str(i) + '.png', 0)    # Right side
    ChessImaL = cv2.imread('img/left' + str(i) + '.png', 0)    # Left side

    retR, cornersR = cv2.findChessboardCorners(ChessImaR, (9, 6), None)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL, (9, 6), None)  # Left side

    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)
    print ("Complete: " + str(i))

# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, ChessImaR.shape[ : : -1], None, None)
hR, wR = ChessImaR.shape[ : 2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, ChessImaL.shape[ : : -1], None, None)

hL, wL = ChessImaL.shape[ : 2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

filenameL = os.path.join("models/", "{}.npy".format("imgpointsL"))
filenameR = os.path.join("models/", "{}.npy".format("imgpointsR"))
filename_op = os.path.join("models/", "{}.npy".format("objpoints"))
filename_mtR = os.path.join("models/", "{}.npy".format("mtxR"))
filename_dR = os.path.join("models/", "{}.npy".format("distR"))
filename_mtL = os.path.join("models/", "{}.npy".format("mtxL"))
filename_dL = os.path.join("models/", "{}.npy".format("distL"))
filename_chR = os.path.join("models/", "{}.npy".format("ChessImaR"))

# Write
np.save(filenameL, imgpointsL)
np.save(filenameR, imgpointsR)
np.save(filename_op, objpoints)
np.save(filename_mtR, mtxR)
np.save(filename_dR, distR)
np.save(filename_mtL, mtxL)
np.save(filename_dL, distL)
np.save(filename_chR, ChessImaR)

print('Cameras had calibrated!') 
exit()
  
