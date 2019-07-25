import os
import numpy as np
import cv2
from sklearn.preprocessing import normalize
from openpyxl import Workbook

def coords_mouse_disp(event,x,y,flags,disp):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9.0
        dis = average
        dis = (-593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06)*0.01*0.335
        #disZ= np.around(dis,decimals=2)
        #disX =  np.around(dis,decimals=2)
        #disY =  np.around(dis,decimals=2)
        disZ = dis
        disX = dis*0.0021
        disY = dis*0.0021
        print('Distance Z: '+ str(disZ)+' m')
        print('Distance X: '+ str((x-375)*disX)+' m')
        print('Distance Y: '+ str((y-325)*disY)+' m')
        print('pix X: '+ str(x-375))
        print('pix Y: '+ str(y-325))

def event1(val):
    global window_size 
    window_size = val
    global trackbar_name1
    if i == 0 or i == 1:
        StereoCalc()
def event2(val):
    if val > 0:
        global num_disp
        num_disp = val
    global trackbar_name2
    trackbar_name2 = 'num_disp: %d' % num_disp
    if i == 0 or i == 1:
        StereoCalc()
def event3(val):
    global min_Disparity
    min_Disparity = val
    global trackbar_name3
    trackbar_name3 = 'min_Disparity: %d' % min_Disparity
    if i == 0 or i == 1:
        StereoCalc()
def event4(val):
    global block_Size
    block_Size = val
    global trackbar_name4
    trackbar_name4 = 'block_Size: %d' % block_Size
    if i == 0 or i == 1:
        StereoCalc()
def event5(val):
    global uniqueness_Ratio
    uniqueness_Ratio = val
    global trackbar_name5
    trackbar_name5 = 'uniqueness_Ratio: %d' % uniqueness_Ratio
    if i == 0 or i == 1:
        StereoCalc()
def event6(val):
    global speckle_WindowSize
    speckle_WindowSize = val
    global trackbar_name6
    trackbar_name6 = 'speckle_WindowSize: %d' % speckle_WindowSize
    if i == 0 or i == 1:
        StereoCalc()
def event7(val):
    global speckle_Range
    speckle_Range = val
    global trackbar_name7
    trackbar_name7 = 'speckle_Range: %d' % speckle_Range
    if i == 0 or i == 1:
        StereoCalc()
def event8(val):
    global disp12_MaxDiff
    disp12_MaxDiff = val
    global trackbar_name8
    trackbar_name8 = 'disp12_MaxDiff: %d' % disp12_MaxDiff
    if i == 0 or i == 1:
        StereoCalc()
def event9(val):
    global preFilter_Cap
    preFilter_Cap = val
    global trackbar_name9
    trackbar_name9 = 'preFilter_Cap: %d' % preFilter_Cap
    if i == 0 or i == 1:
        StereoCalc()
def event10(val):
    global mode_val
    mode_val = val
    global trackbar_name10
    trackbar_name10 = 'mode_val: %d' % mode_val
    if i == 0 or i == 1:
        StereoCalc()
def event11(val):
    global p1
    p1 = val
    global trackbar_name11
    trackbar_name11 = 'p1: %d' % p1
    if i == 0 or i == 1:
        StereoCalc()
def event12(val):
    global p2
    p2 = val
    global trackbar_name12
    trackbar_name12 = 'p2: %d' % p2 
    if i == 0 or i == 1:
        StereoCalc()

def StereoCalc():
    stereo = cv2.StereoSGBM_create(
        minDisparity = min_Disparity,
        numDisparities = 16*num_disp,
        blockSize = block_Size,
        uniquenessRatio = uniqueness_Ratio, 
        speckleWindowSize = speckle_WindowSize, 
        speckleRange = speckle_Range, 
        disp12MaxDiff = disp12_MaxDiff,
        P1 = p1*3*window_size**2,
        P2 = p2*3*window_size**2,  
        #preFilterCap = preFilter_Cap,
        mode = mode_val
        )
    stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time
    # WLS FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
    dispL= disp
    dispR= stereoR.compute(grayR,grayL)
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)

    filteredImg = wls_filter.filter(dispL,grayL,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    disp = ((disp.astype(np.float32)/ 16)-min_Disparity)/(16*num_disp) # Calculation allowing us to have 0 for the most distant object able to detect

    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 
    
    cv2.imshow('img', closing)
    # Mouse click
    cv2.setMouseCallback('img',coords_mouse_disp,closing)
# Mouseclick callback
wb=Workbook()
ws=wb.active  

# Prepare object points
# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

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

print('Files loaded')
# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
#flags |= cv2.CALIB_USE_INTRINSIC_GUESS
#flags |= cv2.CALIB_FIX_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_ASPECT_RATIO
#flags |= cv2.CALIB_ZERO_TANGENT_DIST
#flags |= cv2.CALIB_RATIONAL_MODEL
#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_K3
#flags |= cv2.CALIB_FIX_K4
#flags |= cv2.CALIB_FIX_K5
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria_stereo,
                                                          flags)

# StereoRectify function
rectify_scale= 1 # if 0 image croped, if 1 image not croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
print (MLS)
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)

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


i = int(input('Photos: (1 - use in "test", 0 - from cameras, 2 - real-time)'))

cv2.namedWindow('img', 0)

cv2.createTrackbar(trackbar_name1, 'img', window_size, 50, event1)
cv2.createTrackbar(trackbar_name4, 'img', block_Size, 50, event4)
cv2.createTrackbar(trackbar_name2, 'img', num_disp, 50, event2)
cv2.createTrackbar(trackbar_name3, 'img', min_Disparity, 50, event3)
cv2.createTrackbar(trackbar_name5, 'img', uniqueness_Ratio, 50, event5)
cv2.createTrackbar(trackbar_name6, 'img', speckle_WindowSize, 200, event6)
cv2.createTrackbar(trackbar_name7, 'img', speckle_Range, 50, event7)
cv2.createTrackbar(trackbar_name8, 'img', disp12_MaxDiff, 50, event8)
cv2.createTrackbar(trackbar_name9, 'img', preFilter_Cap, 100, event9)
cv2.createTrackbar(trackbar_name10, 'img', mode_val, 3, event10)
cv2.createTrackbar(trackbar_name11, 'img', p1, 50, event11)
cv2.createTrackbar(trackbar_name12, 'img', p2, 100, event12)

if i == 1:
    frameL= cv2.imread('test/left.png', 0)    # Right side
    frameR= cv2.imread('test/right.png', 0)    # Left side
    grayR= frameR
    grayL= frameL
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

    CamR= cv2.VideoCapture(devpath1)
    CamL= cv2.VideoCapture(devpath2)

    while True:
        retR, frameR= CamR.read()
        retL, frameL= CamL.read()

        Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the kalibration parameters founds during the initialisation
        Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
        grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
        cv2.imshow("stereo", np.hstack([cv2.resize(Left_nice, (640, 480)), cv2.resize(Right_nice, (640, 480))]))

        if (cv2.waitKey(1) & 0xFF == ord('s')) and (i == 0 or i == 1):   
            break
        elif i == 2:
            StereoCalc()
            if (cv2.waitKey(1) & 0xFF == ord(' ')):   
                break

print('Test photos loaded')

cv2.waitKey() 