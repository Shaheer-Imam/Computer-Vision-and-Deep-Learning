import cv2

cap = cv2.VideoCapture(0);

while (True):
    ret, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_blur = cv2.blur(img_invert, (10, 10))
    def dodgeV2(image, mask):
        return cv2.divide(image, 255 - mask, scale=256)

    final_img = dodgeV2(img_gray, img_blur)

    cv2.imshow("Output",final_img)
    cv2.imshow("Original",frame)


    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()