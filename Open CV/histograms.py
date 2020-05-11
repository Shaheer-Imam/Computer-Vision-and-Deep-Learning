import cv2
import numpy as np

from matplotlib import pyplot as plt
image = cv2.imread('images/input.jpg')
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.hist(image.ravel(), 256, [0, 256]); plt.show()

color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color = col)
    plt.xlim([0,256])
    
plt.show()

image2 = cv2.imread('images/tobago.jpg') 

for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image2], [i], None, [256], [0, 256])
    plt.plot(histogram2, color = col)
    plt.xlim([0,256])
    
plt.show()