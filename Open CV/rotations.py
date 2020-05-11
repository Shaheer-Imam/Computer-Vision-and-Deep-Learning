import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
height, width = image.shape[:2]

# Divide by two to rototate the image around its centre
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, .5)

rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()

#We could now crop the image as we can calculate it's new size (we haven't learned cropping yet!).

#But here's another method for simple rotations that uses the cv2.transpose function

img = cv2.imread('images/input.jpg')

rotated_image = cv2.transpose(img)

cv2.imshow('Rotated Image - Method 2', rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()

# A horizontal flip.
flipped = cv2.flip(image, 1)
cv2.imshow('Horizontal Flip', flipped) 
cv2.waitKey()
cv2.destroyAllWindows()