# import the necessary packages
from transform import order_points
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def findcoordinates(image):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # show the original image and the edge detected image
    print("STEP 1: Edge Detection")

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
            cv2.imshow("Outline", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return screenCnt.reshape(4, 2) * ratio
    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    return -1
