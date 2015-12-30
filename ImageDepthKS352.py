'''
Created on Dec 29, 2015

@author: Shuyi Wang
'''

if __name__ == '__main__':
    pass

import numpy as np
import cv2
import time
import shelve

def printTextOnImage(img, text="Hello World"):
    cv2.putText(img, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA, False)

def readGrayScaleImage(camera):
    ret, frame = camera.read()

    if frame is None:
        print("No frame captured") 
        print((ret,frame))
        return None, None
    height, width, depth = frame.shape
    
    left = np.zeros((height,width,depth), dtype=np.uint8)
    right = np.zeros((height,width,depth), dtype=np.uint8)
    
    BLUE = 0
    GREEN = 1
    RED= 2
    
    left[:, :, RED] = frame[:, :, RED]
    right[:, :, BLUE] = frame[:, :, BLUE]
    right[:, :, GREEN] = frame[:, :, GREEN]
    
    grayLeft = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) 
    grayRight = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
    return cv2.equalizeHist(grayLeft), cv2.equalizeHist(grayRight)
    
def calibrateCamera(camera):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpointsL = []  # 2d points in image plane.
    imgpointsR = []  # 2d points in image plane.
    
    calibrateStart = time.clock();
    iterStart = time.clock();
    while len(objpoints) < 15:
        iterEnd = time.clock()
        shouldCapture = iterEnd - iterStart > 3.0;
        
        grayL, grayR = readGrayScaleImage(camera)
    
        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, (7, 6), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (7, 6), None)
        
        disPlayStr = str(3-int(iterEnd - iterStart))
        displayRgbL = cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR) 
        displayRgbR = cv2.cvtColor(grayR, cv2.COLOR_GRAY2BGR) 
    
        # If found, add object points, image points (after refining them)
        if retL == True and retR == True :
            if shouldCapture:
                objpoints.append(objp)
        
                cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
                imgpointsL.append(cornersL)
                
                cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
                imgpointsR.append(cornersR)
                
                disPlayStr = "Capture succeeded"
    
            # Draw and display the corners
            cv2.drawChessboardCorners(displayRgbL, (7, 6), cornersL, retL)
            cv2.drawChessboardCorners(displayRgbR, (7, 6), cornersR, retR)
        printTextOnImage(displayRgbL, disPlayStr)
        printTextOnImage(displayRgbR, disPlayStr)
        cv2.imshow("Left", displayRgbL)
        cv2.imshow("Right", displayRgbR)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if shouldCapture:
            iterStart = time.clock()
#         time.sleep(0.5)
    
        
    h, w = grayL.shape[:2]
    print(grayL.shape[::-1]);
    print(grayL.shape[:2]);
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, None, None, None, None, (w, h))
#     newcameramtxL, roiL = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distCoeffsL, (w, h), 1, (w, h))
#     newcameramtxR, roiR = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distCoeffsR, (w, h), 1, (w, h))

    
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, (w, h), R, T, flags=0)
    
    mapLx, mapLy = cv2.initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, (w, h), cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, (w, h), cv2.CV_32FC1)
    
    print("validPixROI1: ")
    print(validPixROI1)
    print("validPixROI2: ")
    print(validPixROI2)

    
    
    return (cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, mapLx, mapLy, mapRx, mapRy, validPixROI1, validPixROI2, h, w)
    
def calibrateCameraAndDisplayResult(cameraIndex, reCalibrate):
    camera = cv2.VideoCapture(cameraIndex)
    
    db = shelve.open(r"C:\ComputerVision\CalibrationResult.db")
    if(reCalibrate):
        (cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, mapLx, mapLy, mapRx, mapRy, validPixROI1, validPixROI2, h, w) = calibrateCamera(camera)
        db["Result"] = (cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, mapLx, mapLy, mapRx, mapRy, validPixROI1, validPixROI2, h, w);
    else:
        (cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, mapLx, mapLy, mapRx, mapRy, validPixROI1, validPixROI2, h, w) = db["Result"] 
    
    
#     stereo = cv2.StereoBM_create(numDisparities=96, blockSize=51)
#     stereo.setROI1(validPixROI1)
#     stereo.setROI2(validPixROI2)
#     stereo.setPreFilterSize(51)
#     stereo.setPreFilterCap(31)
#     stereo.setTextureThreshold(5)
#     stereo.setUniquenessRatio(15)
    stereo = cv2.StereoSGBM_create(minDisparity=16*3, 
                                   numDisparities=16*6, 
                                   blockSize=11,
                                   P1=8*11*11,
                                   P2=32*11*11,
                                   preFilterCap=63,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32)
    
    while True:
        start = time.clock()
        imgL, imgR = readGrayScaleImage(camera)
        
        # undistort
#         dstL = cv2.undistort(imgL, cameraMatrixL, distCoeffsL, None, newcameramtxL)
#         dstR = cv2.undistort(imgR, cameraMatrixR, distCoeffsR, None, newcameramtxR)
        dstL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
        dstR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
        
        # crop the image
#         x, y, ww, hh = validPixROI1
#         dstL = dstL[y:y + hh, x:x + ww]
#         x, y, ww, hh = validPixROI2
#         dstR = dstR[y:y + hh, x:x + ww]
        
        grayL = dstL
        grayR = dstR
        
        fps = 1 / (time.clock() - start)
        printTextOnImage(dstL, "fps: " + str(int(fps)))
        printTextOnImage(dstR, "fps: " + str(int(fps)))
        (rectifiedHeight, rectifiedWidth) = dstL.shape[:2]
        if (rectifiedHeight <= 0.5 * h or rectifiedWidth <= 0.5 * w):
            print("Error in calib, recalibrating")
            (cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, mapLx, mapLy, mapRx, mapRy, validPixROI1, validPixROI2, h, w) = calibrateCamera(camera)
            db["Result"] = (cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, mapLx, mapLy, mapRx, mapRy, validPixROI1, validPixROI2, h, w);
            continue
        
        disparity = stereo.compute(grayR, grayL)
        disparity += 16;
        disparity = disparity.astype(np.uint8);
        
        
        
        cv2.imshow("Left", dstL)
        cv2.imshow("Right", dstR)
        cv2.imshow('DepthMap', disparity)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
    db.close()


calibrateCameraAndDisplayResult(1, False)