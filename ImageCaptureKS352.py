'''
Created on Dec 29, 2015

@author: Shuyi Wang
'''

if __name__ == '__main__':
    pass

import cv2
import numpy as np
cameraIndex = 1
print("Camera index: " + str(cameraIndex))

video_capture = cv2.VideoCapture(cameraIndex)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if frame is None:
        print("No frame captured") 
        print((ret,frame))
        continue;
    height, width, depth = frame.shape
    
    if height<=0 or width <=0: continue;
    
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
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Display the resulting frame
    # cv2.equalizeHist(img) or clahe.apply(img)
    cv2.imshow('Left', cv2.equalizeHist(grayLeft))
    cv2.imshow('Right', cv2.equalizeHist(grayRight))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()