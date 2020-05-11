import cv2
import numpy as np

def sketch(image):

    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    canny_edges = cv2.Canny(img_blur,40,70)
    
    ret,mask=cv2.threshold(canny_edges,70,255,cv2.THRESH_BINARY_INV)
    
    return mask

cap = cv2.VideoCapture(0);

while (True):
    ret, frame = cap.read()
    #image = cv2.imread("images/robert.jpeg")
    cv2.imshow("Output",sketch(frame))
    cv2.imshow("Original",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()