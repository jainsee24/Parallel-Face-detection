# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 00:48:50 2019

@author: sj
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:27:13 2019

@author: sj
"""

import concurrent.futures
import Time   
import time
import cv2 as cv
import multiprocessing
import numpy as np
from numpy import array
from scipy import misc
import datetime
import sys
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
num_cores = multiprocessing.cpu_count()




face = misc.imread('nm/nm1.jpg', cv.IMREAD_UNCHANGED)
print (face.shape)
face_cascade = cv.CascadeClassifier('C:/Users/sj/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
img = cv.imread('nm/nm1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
(x, y, w, h)=(0,0,0,0)
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
frame = img[y:y+h, x:x+w]
cv.imwrite('nm/facecircledimg.jpg', img)

cv.imwrite('nm/face.jpg', frame)

time1=time.time()

img = cv.imread("nm/face.jpg")
H,W = img.shape[:2]
gray = np.zeros((H,W), np.uint8)
for i in range(H):
    for j in range(W):
        gray[i,j] = np.clip(0.07 * img[i,j,0]  + 0.72 * img[i,j,1] + 0.21 * img[i,j,2], 0, 255)
cv.imwrite('nm/facegray.jpg', gray)



face = misc.imread('nm/nm2.jpg', cv.IMREAD_UNCHANGED)
print (face.shape)
face_cascade = cv.CascadeClassifier('C:/Users/sj/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
img = cv.imread('nm/nm2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
frame = img[y:y+h, x:x+w]
cv.imwrite('nm/Face2.jpg', frame)
cv.imwrite('nm/facecircledimg1.jpg', img)


img = cv.imread("nm/Face2.jpg")
H,W = img.shape[:2]
gray = np.zeros((H,W), np.uint8)
for i in range(H):
    for j in range(W):
        gray[i,j] = np.clip(0.07 * img[i,j,0]  + 0.72 * img[i,j,1] + 0.21 * img[i,j,2], 0, 255)
cv.imwrite('nm/Facegray2.jpg', gray)
cv.waitKey()

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

def compare_images(img1, img2):
    img1 = normalize(img1)
    img2 = normalize(img2)
    diff = img1 - img2 
    m_norm = sum(abs(diff))
    z_norm = norm(diff.ravel(), 0) 
    return (m_norm, z_norm)

img1=cv.imread("nm/Facegray2.jpg")
img2=cv.imread("nm/Facegray.jpg")
x=len(np.array(img1))
y=len(np.array(img2))

width = int(img1.shape[1]*y/x)
height = int(img1.shape[0]*y/x)
dim = (width, height)
# resize image
resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
cv.imwrite('nm/NewSizedImageGray.jpg',resized) 

img1=cv.imread("nm/NewSizedImageGray.jpg")
n_m, n_0 = compare_images(img1, img2)
print("Photo norm:", n_m, "/ per pixel:", n_m/img1.size)
print("Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size)

if(n_m/img1.size*100<25):
    print("Images Are of same Person!!!!!Hurray...!!!")
else:
    print("Differnt People")

time2=time.time()

print("Time For Serial Execution - ",Time.timediffparsr(time2,time1))


