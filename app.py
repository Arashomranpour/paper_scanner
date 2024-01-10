import cv2
import numpy as np

widthimage = 640
heightimage = 480

cap = cv2.VideoCapture(1)
cap.set(3, widthimage)
cap.set(4, heightimage)
cap.set(10, 150)


def process(img):
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgblur = cv2.GaussianBlur(imggray, (5, 5), 1)
    imgcanny = cv2.Canny(imgblur, 200, 200)
    kernel = np.ones((5, 5))
    imgdial = cv2.dilate(imgcanny, kernel, iterations=2)
    imgtheresh = cv2.erode(imgdial, kernel, iterations=1)

    return imgtheresh


def contours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000:
            cv2.drawContours(imgContours, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContours, biggest, -1, (255, 240, 0), 25)

    return biggest


def orderthem(mypoints):
    mypoints = mypoints.reshape((4, 2))
    mypointnew = np.zeros((4, 1, 2), np.int32)
    add = mypoints.sum(1)
    mypointnew[0] = mypoints[np.argmin(add)]
    mypointnew[3] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints, axis=1)
    mypointnew[1] = mypoints[np.argmin(diff)]
    mypointnew[2] = mypoints[np.argmax(diff)]
    return mypointnew


def getwrap(img, biggest):
    print(biggest.shape)
    biggest = orderthem(biggest)
    pst1 = np.float32(biggest)
    pst2 = np.float32(
        [[0, 0], [widthimage, 0], [0, heightimage], [widthimage, heightimage]]
    )
    matrix = cv2.getPerspectiveTransform(pst1, pst2)
    imgout = cv2.warpPerspective(img, matrix, (widthimage, heightimage))
    imgcrop = imgout[20 : imgout.shape[0] - 20, 20 : imgout.shape[1] - 20]
    imgcrop = cv2.resize(imgcrop, (widthimage, heightimage))
    return imgcrop


while True:
    success, img = cap.read()
    img = cv2.flip(img, 2)
    cv2.resize(img, (widthimage, heightimage))

    imgContours = img.copy()

    imgthres = process(img)

    biggest = contours(imgthres)

    cv2.imshow("Res", imgthres)

    if biggest.size != 0:
        imgwrap = getwrap(img, biggest)
        cv2.imshow("Res", imgwrap)
        cv2.imwrite("./output_image.jpg", imgwrap)

    else:
        cv2.imshow("no biggest", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
