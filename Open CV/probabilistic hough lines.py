import cv2
import numpy as np

image = cv2.imread('./images/soduku.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize = 3)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, 5, 10)
print(lines.shape)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(image, (x1, y1), (x2, y2),(0, 255, 0), 3)

cv2.imshow('Probabilistic Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()