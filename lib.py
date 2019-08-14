import traceback
import time
import os
import numpy as np
import cv2
from sklearn.preprocessing import normalize
from openpyxl import Workbook
from multiprocessing import Pool

def GetCords2points(point_Left, point_Right, mtxL, mtxR):
        disZ = (mtxL[0][0] * mtxR[0][0] * 7) / (point_Left[0] * mtxR[0][0] - point_Right[0] * mtxL[0][0])
        disX = (point_Left[0] - mtxL[0][2]) * disZ / mtxL[0][0]
        disY = (mtxL[1][2] - point_Left[1]) * disZ / mtxL[1][1]
        return (disX, disY, disZ)

def GetDisparityImage(Img_nice, FramePartFormat, FramePartCords, stereoParam, Processes):
        (Left_nice, Right_nice) = Img_nice
        (FramePart_Width, FramePart_Height) = FramePartFormat
        (FramePart_X, FramePart_Y) = FramePartCords

        grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
        
        grayR = cv2.GaussianBlur(Right_nice, (5, 3), 1) 
        grayL = cv2.GaussianBlur(Left_nice, (5, 3), 1) 

        data = []

        for n in range(Processes):
                if n == 0:
                        data.append(   (grayL[FramePart_Y : int(FramePart_Height / Processes) + stereoParam[3] + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], 
                                        grayR[FramePart_Y : int(FramePart_Height / Processes) + stereoParam[3] + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], stereoParam))
                elif n == Processes - 1:
                        data.append(   (grayL[int(FramePart_Height / Processes) * n - stereoParam[3] + FramePart_Y : FramePart_Height + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], 
                                        grayR[int(FramePart_Height / Processes) * n - stereoParam[3] + FramePart_Y : FramePart_Height + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], stereoParam))
                else:
                        data.append(   (grayL[int(FramePart_Height / Processes) * n - stereoParam[3] + FramePart_Y : int(FramePart_Height / Processes) * (n + 1) + stereoParam[3] + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], 
                                        grayR[int(FramePart_Height / Processes) * n - stereoParam[3] + FramePart_Y : int(FramePart_Height / Processes) * (n + 1) + stereoParam[3] + FramePart_Y, FramePart_X : FramePart_Width - 1 + FramePart_X], stereoParam))
        
        pool = Pool(Processes)   
        d_slices = pool.starmap(StereoCal1c, data)
        pool.close()
        pool.join() 
        
        for m in range(Processes):
                if m == 0:
                        DisparityL = d_slices[m][0][0 : int(FramePart_Height / Processes), 0 : FramePart_Width - 1]
                        DisparityR = d_slices[m][1][0 : int(FramePart_Height / Processes), 0 : FramePart_Width - 1]
                elif m == Processes - 1:
                        DisparityL = np.vstack((DisparityL, d_slices[m][0][stereoParam[3] + 1 : int(FramePart_Height / Processes) + stereoParam[3], 0 : FramePart_Width - 1]))
                        DisparityR = np.vstack((DisparityR, d_slices[m][1][stereoParam[3] + 1 : int(FramePart_Height / Processes) + stereoParam[3], 0 : FramePart_Width - 1]))
                else:
                        DisparityL = np.vstack((DisparityL, d_slices[m][0][stereoParam[3] : int(FramePart_Height / Processes) + stereoParam[3], 0 : FramePart_Width - 1]))
                        DisparityR = np.vstack((DisparityR, d_slices[m][1][stereoParam[3] : int(FramePart_Height / Processes) + stereoParam[3], 0 : FramePart_Width - 1]))

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

        disp = ((disp.astype(np.float32) / 16) - stereoParam[2]) / (16 * stereoParam[1])
        closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

        return closing
        # return filteredImg

def GetCordsFromDispImg(pointForDispImg, Img_nice, FramePartFormat, FramePartCords, mtxL, stereoParam, Processes):
        diparity = GetDisparityImage(Img_nice, FramePartFormat, FramePartCords, stereoParam, Processes)

        average = 0
        for u in range (-1, 2):
                for v in range (-1, 2):
                        average += diparity[pointForDispImg[1] + u, pointForDispImg[0] + v]
        average = average / 9.0
        dis = average
        dis = (-593.97 * average ** (3) + 1506.8 * average ** (2) - 1373.1 * average + 522.06) * 0.335
        disZ = dis
        disX = (pointForDispImg[0] - mtxL[0][2]) * disZ / mtxL[0][0]
        disY = (mtxL[1][2] - pointForDispImg[1]) * disZ / mtxL[1][1]

        return (disX, disY, disZ)

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

def Main(frameL, frameR, FramePartFormat, FramePartCords, pointForDispImg, point_Left, point_Right, Processes):
        objpoints = []   
        imgpointsR = []   
        imgpointsL = []

        filenameL = os.path.join("models/", "{}.npy".format("imgpointsL"))
        filenameR = os.path.join("models/", "{}.npy".format("imgpointsR"))
        filename_op = os.path.join("models/", "{}.npy".format("objpoints"))
        filename_mtR = os.path.join("models/", "{}.npy".format("mtxR"))
        filename_dR = os.path.join("models/", "{}.npy".format("distR"))
        filename_mtL = os.path.join("models/", "{}.npy".format("mtxL"))
        filename_dL = os.path.join("models/", "{}.npy".format("distL"))
        filename_chR = os.path.join("models/", "{}.npy".format("ChessImaR"))

        imgpointsR = np.load(filenameR)
        imgpointsL = np.load(filenameL)
        objpoints = np.load(filename_op)
        mtxR = np.load(filename_mtR)
        distR = np.load(filename_dR)
        mtxL = np.load(filename_mtL)
        distL = np.load(filename_dL)
        ChessImaR = np.load(filename_chR)

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC

        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, ChessImaR.shape[::-1], criteria_stereo, flags)
        rectify_scale = 1 # if 0 image croped, if 1 image not croped
        RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImaR.shape[::-1], R, T, rectify_scale,(0,0))  

        Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS,  dLS,  RL,  PL,  ChessImaR.shape[::-1], cv2.CV_16SC2)  
        Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS,  dRS,  RR,  PR,  ChessImaR.shape[::-1], cv2.CV_16SC2)

        Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
        Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0) 

        window_size = 2 #5 
        num_disp = 10 
        min_Disparity = 0 
        block_Size = 3 #5
        uniqueness_Ratio = 0 
        speckle_WindowSize = 0
        speckle_Range = 2
        disp12_MaxDiff = 1
        preFilter_Cap = 63 
        mode_val = 1
        p1 = 8
        p2 = 32 
        
        stereoParam = (window_size, num_disp, min_Disparity, block_Size, uniqueness_Ratio, speckle_WindowSize, speckle_Range, disp12_MaxDiff, preFilter_Cap, mode_val, p1, p2)

        (X, Y, Z) = GetCordsFromDispImg(pointForDispImg, (Left_nice, Right_nice), FramePartFormat, FramePartCords, mtxL, stereoParam, Processes)

        (X1, Y1, Z1) = GetCords2points(point_Left, point_Right, mtxL, mtxR)

        return [(X, Y, Z), (X1, Y1, Z1)]

FRAME_WIDTH = 1280
FRAME_HEIGHT = 960

Processes = 8  

FramePartFormat = (700, 500)
FramePartCords = (300, 400)

pointForDispImg = (500, 400)
point_Left = (500, 400)
point_Right = (320, 400)

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

while CamL.isOpened() and CamR.isOpened():
        retR, frameR = CamR.read()
        retL, frameL = CamL.read()

        stereo = np.hstack([frameL, frameR])

        cv2.imshow("stereo", stereo)

        if cv2.waitKey(1) & 0xFF == ord('s'):
                data = Main(frameL, frameR, FramePartFormat, FramePartCords, pointForDispImg, point_Left, point_Right, Processes)
                print(data)

        elif (cv2.waitKey(1) & 0xFF == ord(' ')): 
                cv2.destroyAllWindows()
                exit()