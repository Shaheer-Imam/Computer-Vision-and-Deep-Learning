import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./images/ex2.jpg')
rows,cols,ch = image.shape

cv2.imshow('Original', image)
cv2.waitKey(0)

points_A = np.float32([[320,15], [700,215], [85,610]])
points_B = np.float32([[0,0], [420,0], [0,594]])
M = cv2.getAffineTransform(points_A, points_B)

warped = cv2.warpAffine(image, M, (cols, rows))
 
cv2.imshow('warpPerspective', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()