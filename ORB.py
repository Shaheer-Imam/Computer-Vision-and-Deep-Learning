import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fast = cv2.FastFeatureDetector()
keypoints = fast.detect(gray, None)
print("Number of keypoints Detected: ", len(keypoints))
image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - FAST', image)
cv2.waitKey()
cv2.destroyAllWindows()