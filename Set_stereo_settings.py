import traceback
import time
import os
import numpy as np
import cv2
from sklearn.preprocessing import normalize
from openpyxl import Workbook
from multiprocessing import Pool

FRAME_WIDTH = 1280
FRAME_HEIGHT = 960

# FRAME_WIDTH = 640
# FRAME_HEIGHT = 480

def coords_mouse_disp(event, x, y, flags, disp):
        if event == cv2.EVENT_LBUTTONDBLCLK:
                average = 0
                for u in range (-1, 2):
                        for v in range (-1, 2):
                                average += disp[y + u, x + v]
                average = average / 9.0
                dis = average
                #dis = (-593.97 * average ** (3) + 1506.8 * average ** (2) - 1373.1 * average + 522.06) * 0.335
                disZ = dis
                disX = (x - mtxL[0][2]) * disZ / mtxL[0][0]
                disY = (mtxL[1][2] - y) * disZ / mtxL[1][1]
                print ("Z: " + str(disZ))
                print ("X: " + str(disX))
                print ("Y: " + str(disY)) 

def coords_frame_zone(event, x, y, flags, param):
        if event == cv2.EVENT_MBUTTONDOWN:
                if x < FRAME_WIDTH:     
                        global point_move
                        global point_3
                        global point_4
                        point_move = True
                        point_3 = (x, y)
                        print ('Point 3:')
                        print ('pix x: '+ str(point_3[0]) + ', pix y: '+ str(point_3[1]))
                        
        elif event == cv2.EVENT_MOUSEMOVE: 
                if x < FRAME_WIDTH:     
                        if point_move == True:
                                point_4 = (x, y)
                        
        elif event == cv2.EVENT_MBUTTONUP:
                global point_move
                global FramePart_Width
                global FramePart_Height
                global FramePart_X
                global FramePart_Y

                if x < FRAME_WIDTH:     
                        print ('Point 4:')
                        print ('pix x: '+ str(point_4[0]) + ', pix y: '+ str(point_4[1]))

                        point_move = False
                        if point_3[0] < point_4[0]:
                                FramePart_Width = point_4[0] - point_3[0]
                                FramePart_X = point_3[0]
                        else:
                                FramePart_Width = point_3[0] - point_4[0]
                                FramePart_X = point_4[0]

                        if point_3[1] < point_4[1]:
                                FramePart_Height = point_4[1] - point_3[1]
                                FramePart_Y = point_3[1]
                        else:
                                FramePart_Height = point_3[1] - point_4[1]
                                FramePart_Y = point_4[1]
                        print ("Frame Width: " + str(FramePart_Width) + "; Frame Height: " + str(FramePart_Height))
                        print ("Frame X: " + str(FramePart_X) + "; Frame Y: " + str(FramePart_Y))

        elif event == cv2.EVENT_LBUTTONDBLCLK:
                if x < FRAME_WIDTH and x > 0:
                        global point_1
                        point_1 = (x, y)
                        print ('Point 1:')
                        print ('pix x: '+ str(x) + ', pix y: '+ str(y))
                else:
                        global point_2
                        point_2 = (x - FRAME_WIDTH, y)
                        print ('Point 2:')
                        print ('pix x: '+ str(x - FRAME_WIDTH) + ', pix y: '+ str(y))

                if point_1 > (0, 0) and point_2 > (0, 0):
                        disZ = (mtxL[0][0] * mtxR[0][0] * 7) / (point_1[0] * mtxR[0][0] - point_2[0] * mtxL[0][0])
                        disX = (point_1[0] - mtxL[0][2]) * disZ / mtxL[0][0]
                        disY = (mtxL[1][2] - point_1[1]) * disZ / mtxL[1][1]
                        print ("Z: " + str(disZ))
                        print ("X: " + str(disX))
                        print ("Y: " + str(disY))   

def event1(val):
        global window_size 
        window_size = val
        if i == 0 or i == 1:
                StereoCalc()
def event2(val):
        if val > 0:
                global num_disp
                num_disp = val
        if i == 0 or i == 1:
                StereoCalc()
def event3(val):
        global min_Disparity
        min_Disparity = val
        if i == 0 or i == 1:
                StereoCalc()
def event4(val):
        global block_Size
        block_Size = val
        if i == 0 or i == 1:
                StereoCalc()
def event5(val):
        global uniqueness_Ratio
        uniqueness_Ratio = val
        if i == 0 or i == 1:
                StereoCalc()
def event6(val):
        global speckle_WindowSize
        speckle_WindowSize = val
        if i == 0 or i == 1:
                StereoCalc()
def event7(val):
        global speckle_Range
        speckle_Range = val
        if i == 0 or i == 1:
                StereoCalc()
def event8(val):
        global disp12_MaxDiff
        disp12_MaxDiff = val
        if i == 0 or i == 1:
                StereoCalc()
def event9(val):
        global preFilter_Cap
        preFilter_Cap = val
        if i == 0 or i == 1:
                StereoCalc()
def event10(val):
        global mode_val
        mode_val = val
        if i == 0 or i == 1:
                StereoCalc()
def event11(val):
        global p1
        p1 = val
        if i == 0 or i == 1:
                StereoCalc()
def event12(val):
        global p2
        p2 = val
        if i == 0 or i == 1:
                StereoCalc()

def StereoCal1c(grayL, grayR, stereoParam):
        try:
                stereo = cv2.StereoSGBM_create(
                        minDisparity = stereoParam[2],
                        numDisparities = 16 * stereoParam[1],
                        blockSize = stereoParam[3],
                        uniquenessRatio = stereoParam[4], 
                        speckleWindowSize = stereoParam[5], 
                        speckleRange = stereoParam[6], 
                        disp12MaxDiff = stereoParam[7],
                        P1 = stereoParam[10] * 3 * stereoParam[0] ** 2,
                        P2 = stereoParam[11] * 3 * stereoParam[0] ** 2,  
                        preFilterCap = stereoParam[8],
                        mode = stereoParam[9])

                stereoR = cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time
                dispL = stereo.compute(grayL, grayR)
                dispR = stereoR.compute(grayR, grayL)

                return (dispL, dispR)
        except Exception as e:
                traceback.print_exc()
                raise e

def StereoCalc():
        stereo = cv2.StereoSGBM_create(
                minDisparity = min_Disparity,
                numDisparities = 16 * num_disp,
                blockSize = block_Size,
                uniquenessRatio = uniqueness_Ratio, 
                speckleWindowSize = speckle_WindowSize, 
                speckleRange = speckle_Range, 
                disp12MaxDiff = disp12_MaxDiff,
                P1 = p1 * 3 * window_size ** 2,
                P2 = p2 * 3 * window_size ** 2,  
                preFilterCap = preFilter_Cap,
                mode = mode_val)

        stereoR = cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time
        # WLS FILTER Parameters
            
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = stereo)
        wls_filter.setLambda(80000)
        wls_filter.setSigmaColor(1.8)

        disp = stereo.compute(grayL, grayR)#.astype(np.float32)/ 16
        dispL = disp
        dispR = stereoR.compute(grayR, grayL)

        filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
        filteredImg = cv2.normalize(src = filteredImg, dst = filteredImg, beta=0, alpha=255, norm_type = cv2.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)

        disp = ((disp.astype(np.float32) / 16) - min_Disparity) / (16 * num_disp) # Calculation allowing us to have 0 for the most distant object able to detect
        closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

        cv2.imshow('img', closing)

        cv2.setMouseCallback('img', coords_mouse_disp, closing)


# amg = 1
# if amg == 1:
# Arrays to store object points and image points from all images
objpoints = []   # 3d points in real world space
imgpointsR = []   # 2d points in image plane
imgpointsL = []

filenameL = os.path.join("models/", "{}.npy".format("imgpointsL"))
filenameR = os.path.join("models/", "{}.npy".format("imgpointsR"))
filename_op = os.path.join("models/", "{}.npy".format("objpoints"))
filename_mtR = os.path.join("models/", "{}.npy".format("mtxR"))
filename_dR = os.path.join("models/", "{}.npy".format("distR"))
filename_mtL = os.path.join("models/", "{}.npy".format("mtxL"))
filename_dL = os.path.join("models/", "{}.npy".format("distL"))
filename_chR = os.path.join("models/", "{}.npy".format("ChessImaR"))

# Read
imgpointsR = np.load(filenameR)
imgpointsL = np.load(filenameL)
objpoints = np.load(filename_op)
mtxR = np.load(filename_mtR)
distR = np.load(filename_dR)
mtxL = np.load(filename_mtL)
distL = np.load(filename_dL)
ChessImaR = np.load(filename_chR)

<<<<<<< HEAD
print ('Files loaded')
=======
print('Files loaded')
>>>>>>> 06cc637e5cf46ef6cfc44e695338ef7896091d12
# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

<<<<<<< HEAD
# flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
# flags |= cv2.CALIB_USE_INTRINSIC_GUESS
# flags |= cv2.CALIB_FIX_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_ASPECT_RATIO
# flags |= cv2.CALIB_ZERO_TANGENT_DIST
# flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_K3
# flags |= cv2.CALIB_FIX_K4
# flags |= cv2.CALIB_FIX_K5

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, ChessImaR.shape[::-1], criteria_stereo, flags)
# StereoRectify function
rectify_scale = 1 # if 0 image croped, if 1 image not croped
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImaR.shape[::-1], R, T, rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function

Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS,  dLS,  RL,  PL,  ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS,  dRS,  RR,  PR,  ChessImaR.shape[::-1], cv2.CV_16SC2)

# filenameLSM = os.path.join("models_1/", "{}.npy".format("Left_Stereo_Map"))
# filenameRSM = os.path.join("models_1/", "{}.npy".format("Right_Stereo_Map"))
# np.save(filenameLSM, Left_Stereo_Map)
# np.save(filenameRSM, Right_Stereo_Map)

# print('Cameras had calibrated!') 

# filenameLSM = os.path.join("models_1/", "{}.npy".format("Left_Stereo_Map"))
# filenameRSM = os.path.join("models_1/", "{}.npy".format("Right_Stereo_Map"))
# Left_Stereo_Map = np.load(filenameLSM)
# Right_Stereo_Map = np.load(filenameRSM)


Processes = 8

window_size = 0 #5 
num_disp = 10 
min_Disparity = 0 
block_Size = 0 #5
uniqueness_Ratio = 15 
speckle_WindowSize = 50
speckle_Range = 2
disp12_MaxDiff = 1
preFilter_Cap = 63 
mode_val = 1
p1 = 8
p2 = 32


FramePart_Height = FRAME_HEIGHT
FramePart_Width = FRAME_WIDTH

FramePart_Y = 0
FramePart_X = 0

point_1 = (0, 0)
point_2 = (0, 0)

point_3 = (0, 0)
point_4 = (FRAME_WIDTH, FRAME_HEIGHT)
point_move = False

trackbar_name1 = 'window_size: %d' % window_size
trackbar_name2 = 'num_disp: %d' % num_disp
trackbar_name3 = 'min_Disparity: %d' % min_Disparity
trackbar_name4 = 'block_Size: %d' % block_Size
trackbar_name5 = 'uniqueness_Ratio: %d' % uniqueness_Ratio
trackbar_name6 = 'speckle_WindowSize: %d' % speckle_WindowSize
trackbar_name7 = 'speckle_Range: %d' % speckle_Range
trackbar_name8 = 'disp12_MaxDiff: %d' % disp12_MaxDiff
trackbar_name9 = 'preFilter_Cap: %d' % preFilter_Cap
trackbar_name10 = 'mode_val: %d' % mode_val
trackbar_name11 = 'p1: %d' % p1
trackbar_name12 = 'p2: %d' % p2


i = int(input('1 - use in "test" folder, 0 - just cameras, 2 - real-time, 3 - real-time(multiCPU): '))

cv2.namedWindow('img', 0)
cv2.namedWindow("stereo", 0)
cv2.resizeWindow("stereo", FRAME_WIDTH, FRAME_HEIGHT / 2)

cv2.createTrackbar(trackbar_name1, 'img', window_size, 50, event1)
cv2.createTrackbar(trackbar_name4, 'img', block_Size, 50, event4)
cv2.createTrackbar(trackbar_name2, 'img', num_disp, 50, event2)
cv2.createTrackbar(trackbar_name3, 'img', min_Disparity, 50, event3)
cv2.createTrackbar(trackbar_name5, 'img', uniqueness_Ratio, 50, event5)
cv2.createTrackbar(trackbar_name6, 'img', speckle_WindowSize, 200, event6)
cv2.createTrackbar(trackbar_name7, 'img', speckle_Range, 50, event7)
cv2.createTrackbar(trackbar_name8, 'img', disp12_MaxDiff, 200, event8)
cv2.createTrackbar(trackbar_name8, 'img', disp12_MaxDiff, 50, event8)
cv2.createTrackbar(trackbar_name9, 'img', preFilter_Cap, 100, event9)
cv2.createTrackbar(trackbar_name10, 'img', mode_val, 3, event10)
cv2.createTrackbar(trackbar_name11, 'img', p1, 50, event11)
cv2.createTrackbar(trackbar_name12, 'img', p2, 100, event12)

if i == 1:
        frameL = cv2.imread('test/left.png', 0)    # Right side
        frameR = cv2.imread('test/right.png', 0)    # Left side

        grayR = frameR
        grayL = frameL

        cv2.imshow("stereo", np.hstack([cv2.resize(frameL, (640, 480)), cv2.resize(frameR, (640, 480))]))

else:
        devpath1 = ""
        devpath2 = ""

        try:
                for path in os.listdir('/dev/v4l/by-id/'):
                        if path.find('index0') >= 0 and path.find('299B3041') >= 0:
                                devpath2 = os.path.join('/dev/v4l/by-id/',path)

                        elif path.find('index0') >= 0 and path.find('299B3065') >= 0:
                                devpath1 = os.path.join('/dev/v4l/by-id/',path)
        except OSError as Err:
                print ('Device Not Found!! Check Connection')

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
                print("Cameras opened: " + CamL.isOpened() and CamR.isOpened())
                exit()

        if i == 0:
                j = int(input('1 - save photos, 0 - not: '))

        pool = Pool(Processes)   
        while CamL.isOpened() and CamR.isOpened():
                retR, frameR = CamR.read()
                retL, frameL = CamL.read()

                #frameL = cv2.warpAffine(frameL, np.float32([ [1,0,0], [0,1,7] ]), frameL.shape[:2])

                Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
                Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0) 

                grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
                grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
                
                grayR = cv2.GaussianBlur(Right_nice, (5, 3), 1) 
                grayL = cv2.GaussianBlur(Left_nice, (5, 3), 1) 

                stereo = np.hstack([grayL, grayR])

                cv2.rectangle(stereo, point_3, point_4,(255, 255, 255), 2)

                cv2.imshow("stereo", stereo)

                cv2.setMouseCallback("stereo", coords_frame_zone, stereo)

                if (cv2.waitKey(1) & 0xFF == ord('s')) and (i == 0) and (j == 1):
                        cv2.imwrite('test/right.png', grayR) # Save the image in the file where this Programm is located
                        cv2.imwrite('test/left.png', grayL)

                elif i == 2:
                        StereoCalc()
                        if (cv2.waitKey(1) & 0xFF == ord(' ')):   
                                break
                elif i == 3:
                        start_time = time.time()
                        parts = Processes
                        data = []

                        stereoParam = (window_size, num_disp, min_Disparity, block_Size, uniqueness_Ratio, speckle_WindowSize, speckle_Range, disp12_MaxDiff, preFilter_Cap, mode_val, p1, p2)
                        
                        for n in range(parts):
                                if n == 0:
                                        data.append(   (grayL[FramePart_Y : int(FramePart_Height / parts) + block_Size + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], 
                                                        grayR[FramePart_Y : int(FramePart_Height / parts) + block_Size + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], stereoParam))
                                elif n == parts - 1:
                                        data.append(   (grayL[int(FramePart_Height / parts) * n - block_Size + FramePart_Y : FramePart_Height + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], 
                                                        grayR[int(FramePart_Height / parts) * n - block_Size + FramePart_Y : FramePart_Height + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], stereoParam))
                                else:
                                        data.append(   (grayL[int(FramePart_Height / parts) * n - block_Size + FramePart_Y : int(FramePart_Height / parts) * (n + 1) + block_Size + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], 
                                                        grayR[int(FramePart_Height / parts) * n - block_Size + FramePart_Y : int(FramePart_Height / parts) * (n + 1) + block_Size + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], stereoParam))
                        
                        d_slices = pool.starmap(StereoCal1c, data)
                        
                        for m in range(parts):
                                if m == 0:
                                        DisparityL = d_slices[m][0][0 : int(FramePart_Height / parts), 0 : FramePart_Width - 1]
                                        DisparityR = d_slices[m][1][0 : int(FramePart_Height / parts), 0 : FramePart_Width - 1]
                                elif m == parts - 1:
                                        DisparityL = np.vstack((DisparityL, d_slices[m][0][block_Size + 1 : int(FramePart_Height / parts) + block_Size, 0 : FramePart_Width - 1]))
                                        DisparityR = np.vstack((DisparityR, d_slices[m][1][block_Size + 1 : int(FramePart_Height / parts) + block_Size, 0 : FramePart_Width - 1]))
                                else:
                                        DisparityL = np.vstack((DisparityL, d_slices[m][0][block_Size : int(FramePart_Height / parts) + block_Size, 0 : FramePart_Width - 1]))
                                        DisparityR = np.vstack((DisparityR, d_slices[m][1][block_Size : int(FramePart_Height / parts) + block_Size, 0 : FramePart_Width - 1]))

                        stereo = cv2.StereoSGBM_create(
                                minDisparity = stereoParam[2],
                                numDisparities = 16 * stereoParam[1],
                                blockSize = stereoParam[3],
                                uniquenessRatio = stereoParam[4], 
                                speckleWindowSize = stereoParam[5], 
                                speckleRange = stereoParam[6], 
                                disp12MaxDiff = stereoParam[7],
                                P1 = stereoParam[10] * 3 * stereoParam[0] ** 2,
                                P2 = stereoParam[11] * 3 * stereoParam[0] ** 2,  
                                preFilterCap = stereoParam[8],
                                mode = stereoParam[9])

                        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left = stereo)
                        wls_filter.setLambda(80000)
                        wls_filter.setSigmaColor(1.8)

                        disp = DisparityL
                        dispL = np.int16(DisparityL)
                        dispR = np.int16(DisparityR)

                        filteredImg = wls_filter.filter(dispL, grayL[FramePart_Y : FramePart_Y + FramePart_Height, FramePart_X : FramePart_X + FramePart_Width], None, dispR) #, None, dispR
                        filteredImg = cv2.normalize(src = filteredImg, dst = filteredImg, beta=0, alpha=255, norm_type = cv2.NORM_MINMAX)
                        filteredImg = np.uint8(filteredImg)

                        disp = ((disp.astype(np.float32) / 16) - min_Disparity) / (16 * num_disp)

                        closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
                        # print ('Done! Time taken: ' + format(time.time() - start_time))

                        cv2.imshow('img', closing)
                        
                        cv2.setMouseCallback('img', coords_mouse_disp, closing)

                        if (cv2.waitKey(1) & 0xFF == ord(' ')):   
                                break
                elif i != 0:
                        break
        pool.close()
        pool.join() 

cv2.destroyAllWindows()  
exit() 
