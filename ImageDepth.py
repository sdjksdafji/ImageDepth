'''
Created on Aug 21, 2015

@author: Shuyi Wang
'''

import numpy as np
import cv2

imgL = cv2.imread(r"C:\Users\Shuyi Wang\Documents\workspace-sts\ImageDepth\tsukuba_l.png", 0)
imgR = cv2.imread(r"C:\Users\Shuyi Wang\Documents\workspace-sts\ImageDepth\tsukuba_r.png", 0)     
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
disparity += 16;
disparity = disparity.astype(np.uint8);
cv2.imshow('Video', disparity)
cv2.waitKey(0)
