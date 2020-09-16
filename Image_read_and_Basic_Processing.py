# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:13:35 2020

@author: Anonymous
"""
"""Computer Vision - Image Basics with OpenCV and Python
Now we will learn how to work with OpenCV for Computer Vision. We will accomplish it by completing each task in the project:

Open an Image with Matplotlib
Open an Image with OpenCV
Learn the main points to remember for OpenCV
Change the Image's color, size, orientation
"""


#Open an Image with Matplotlib 
#PIL
#Import Libraries

import numpy as np 
import matplotlib.pyplot as plt 

from PIL import Image 
path = r""
img = Image.open(path)
#img.rotate(90)
print(type(img))
# Turn img to array 

img_array = np.asarray(img)
print(type(img_array))

# height width and channels
print("height width and channels : {}".format(img_array.shape))

plt.imshow(img_array)

# RGB channels
"""Red channel is in position No 0
   Green channel is in position No 1
   Blue channel is in position No 2
   
   The colour values vary from 0 == no colour from the channel, to 255 == full colour from the channel 
"""
img_test = img_array.copy()
# only red channel
plt.imshow(img_test[:,:,0])
# Scale Red channel to Gray
plt.imshow(img_test[:,:,0], cmap = 'gray')

# only green channel
plt.imshow(img_test[:,:,1])
# Scale green channel to Gray
plt.imshow(img_test[:,:,0], cmap = 'gray')

# only Blue channel
plt.imshow(img_test[:,:,2])
# Scale blue channel to Gray
plt.imshow(img_test[:,:,2], cmap = 'gray')

# Remove red 
img_test[:,:,0] = 0 
plt.imshow(img_test)

# Remove green  
img_test[:,:,1] = 0 
plt.imshow(img_test)

# Remove Blue
img_test[:,:,2] = 0 
plt.imshow(img_test)

#Open an Image with OpenCV
#import Libraries 
import cv2
import numpy as np 
import matplotlib.pyplot as plt

path_2 = r""
img_1 = cv2.imread(path_2)
type(img_1)
print(img_1.shape)
plt.imshow(img_1)

"""Until now we were working with Matplotlib and RGB
   opencv is reading the channel as BGR.
"""
# We will convert opencv to the channels of the photo
img_1_fix = cv2.cvtColor(img_1,
                         cv2.COLOR_BGR2RGB)

# scale to gray and check the Shape
img_gray = cv2.imread(path_2,cv2.IMREAD_GRAYSCALE)
print(img_gray.shape)
plt.imshow(img_gray, cmap = 'gray')

# resize the image 

img_new = cv2.resize(img_1_fix,(1000,400))
plt.imshow(img_new)

# resize with Ratio
width_ratio = 0.5
height_ratio = 0.5

img2 = cv2.resize(img_fix,(0,0),width_ratio,height_ratio)
plt.imshow(img2)

# Flip img horizontal
img3 = cv2.flip(img_1_fix,0)
plt.imshow(img3)

# Flip img vertical
img_3 = cv2.flip(img_1_fix,1)
plt.imshow(img_3)

# Flip img horizontal and vertical axis
img4 = cv2.flip(img_1_fix,-1)
plt.imshow(img4)

# change the size of our canva
last_img = plt.figsize = (10,7)
ilp = last_img.add_subplot(111)
ilp.imshow(img_fix)










