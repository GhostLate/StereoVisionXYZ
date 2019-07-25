###################################
##### Authors:                #####
##### Stephane Vujasinovic    #####
##### Frederic Uhrweiller     ##### 
#####                         #####
##### Creation: 2017          #####
###################################
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
      devpath2 = os.path.join('/dev/v4l/by-id/',path)

    elif path.find('index0') >= 0 and path.find('299B3065') >= 0:
      devpath1 = os.path.join('/dev/v4l/by-id/',path)
      #if devpath1 == "":
      #  devpath1 = os.path.join('/dev/v4l/by-id/',path)
      #else:
      #  devpath2 = os.path.join('/dev/v4l/by-id/',path)
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

# Call the two cameras
CamR= cv2.VideoCapture(devpath1)   # Wenn 0 then Right Cam and wenn 2 Left Cam
CamL= cv2.VideoCapture(devpath2)

while True:
    retR, frameR= CamR.read()
    retL, frameL= CamL.read()
    grayR= cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retR, cornersR = cv2.findChessboardCorners(grayR,(9,6),None)  # Define the number of chess corners (here 9 by 6) we are looking for with the right Camera
    retL, cornersL = cv2.findChessboardCorners(grayL,(9,6),None)  # Same with the left camera
    cv2.imshow('imgR',frameR)
    cv2.imshow('imgL',frameL)

    # If found, add object points, image points (after refining them)
    if (retR == True) & (retL == True):
        corners2R= cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)    # Refining the Position
        corners2L= cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(grayR,(9,6),corners2R,retR)
        cv2.drawChessboardCorners(grayL,(9,6),corners2L,retL)
        cv2.imshow('VideoR',grayR)
        cv2.imshow('VideoL',grayL)

        if cv2.waitKey(0) & 0xFF == ord('s'):   # Push "s" to save the images and "c" if you don't want to
            str_id_image= str(id_image)
            print('Images ' + str_id_image + ' saved for right and left cameras')
            cv2.imwrite('img/right'+str_id_image+'.png',frameR) # Save the image in the file where this Programm is located
            cv2.imwrite('img/left'+str_id_image+'.png',frameL)
            id_image=id_image+1
        else:
            print('Images not saved')

    # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):   # Push the space bar and maintan to exit this Programm
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()    
