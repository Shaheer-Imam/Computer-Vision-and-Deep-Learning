"""
    SIFT and SURF have been removed from opencv default. To get access from SIFT and SURF you
    need to pull down opencv and opencv_contrib repositories from GitHub and then compile 
    and install Opencv from source
"""

import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT 
keypoints = sift.detect(gray, None)
print("Number of keypoints Detected: ", len(keypoints))
image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - SIFT', image)
cv2.waitKey(0)
cv2.destroyAllWindows()