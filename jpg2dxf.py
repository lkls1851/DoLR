import cv2
import numpy as np
import ezdxf

image = cv2.imread('figure_2_0.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

binary = cv2.bitwise_not(binary)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

doc = ezdxf.new(dxfversion='R2010')

msp = doc.modelspace()
for contour in contours:
    points = [tuple(point[0]) for point in contour]
    msp.add_lwpolyline(points)

doc.saveas('check.dxf')