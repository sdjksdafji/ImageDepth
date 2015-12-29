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
        print((ret,frame))
        continue;
    height, width, depth = frame.shape
    print((height,width))
    if height<=0 or width <=0: continue;
    
    left = np.zeros((height,width,depth), dtype=np.uint8)
    right = np.zeros((height,width,depth), dtype=np.uint8)
    
    RED= 2
    BLUE = 0
    GREEN = 1
    
    for h in range(height):
        for w in range(width):
            left[h, w, RED] = frame[h, w, RED]
            right[h, w, BLUE] = frame[h, w, BLUE]
            right[h, w, GREEN] = frame[h, w, GREEN]
            
    # Display the resulting frame
    cv2.imshow('Left', cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
    cv2.imshow('Right', cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()