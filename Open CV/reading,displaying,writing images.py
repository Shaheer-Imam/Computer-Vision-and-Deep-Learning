import cv2
import numpy as np 

input = cv2.imread('./images/input.jpg')
cv2.imshow('Hello World', input)


print(input.shape)
print('Height of Image:', int(input.shape[0]), 'pixels')
print('Width of Image: ', int(input.shape[1]), 'pixels')

cv2.waitKey()
cv2.destroyAllWindows()
