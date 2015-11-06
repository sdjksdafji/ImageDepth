'''
Created on Nov 11, 2015

@author: Shuyi Wang
'''

if __name__ == '__main__':
    pass

import cv2
cameraIndex = -1
print("Camera index: " + str(cameraIndex))

video_capture = cv2.VideoCapture(cameraIndex)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if frame is None: 
        print((ret,frame))
        continue;
    h, w = frame.shape[:2]
    print((h,w))
    if h<=0 or w <=0: continue;
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
