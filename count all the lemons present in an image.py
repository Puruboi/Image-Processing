# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:27:47 2020
@author: Anonymous
"""
#importing required libraries
#pip install opencv-python
#pip install numpy
import cv2   # open cv library 
import numpy as np   # numpy library 
import time
print("Libraries imported successfully")

# reading image path 
image_path = r"C:\Users\Anonymous\Desktop\drive-download-20200904T145408Z-001\lemon1.jpg"

# imread reads the image 
image = cv2.imread(image_path)

"""cvtColor(...)
    cvtColor(src, code[, dst[, dstCn]]) -> dst
    brief Converts an image from one color space to another."""
#converting color image to BGR image
image_1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#cv2.imshow('BGR_image',image_1)  #plotting BGR format image
#medianBlur(...)
#medianBlur(src, ksize[, dst]) -> dst; @brief Blurs an image using the median filter
image_2 = cv2.medianBlur(image_1,1)

"""
HoughCircles(...)
    HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    @brief Finds circles in a grayscale image using the Hough transform.
    The function finds circles in a grayscale image using a modification of the Hough transform. 
    Example:
    include snippets/imgproc_HoughLinesCircles.cpp   
    note Usually the function detects the centers of circles well. However, it may fail to find correct
    radii. You can assist to the function by specifying the radius range ( minRadius and maxRadius ) if
    you know it. Or, in the case of #HOUGH_GRADIENT method you may set maxRadius to a negative number
    to return centers only without radius search, and find the correct radius using an additional procedure.
"""

circles=cv2.HoughCircles(image_2,cv2.HOUGH_GRADIENT,1, 30, np.array([]), 80, 20, 3, 50)
# detection of circles 
circles = np.uint16(np.around(circles))
start = time.time()
# counting for lemons
for i in circles[0,:]:
    cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
    
end = time.time()
'''Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.'''

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image,str(circles.shape[1])+' lemons detected in the image',(20,50), font, 0.6,(250,250,250),2,cv2.LINE_AA)
cv2.imshow('Number of detected lemons',image)
print ("Seconds taken for prediction: {}".format(end - start))
cv2.waitKey(0)
cv2.destroyAllWindows()