import cv2
import numpy as np

image = cv2.imread('images/input.jpg')

M = np.ones(image.shape, dtype = "uint8") * 175 

added = cv2.add(image, M)
cv2.imshow("Added", added)

subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)

M = np.ones(image.shape, dtype = "uint8") * 75 
print(M)

cv2.waitKey(0)
cv2.destroyAllWindows()