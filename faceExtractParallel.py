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
print(num_cores)
camera = cv.VideoCapture(0)
i = 0
while i < 50:
    return_value, image = camera.read()
    cv.imwrite('sj'+str(i)+'.jpg', image)
    i += 1
del(camera)
a = datetime.datetime.now()

Imgcom=int(input("Pressanykey"))
camera = cv.VideoCapture(0)
i = 0
while i < 50:
    return_value, image = camera.read()
    cv.imwrite('sj1'+str(i)+'.jpg', image)
    i += 1
del(camera)




face = misc.imread('sj49.jpg', cv.IMREAD_UNCHANGED)
print (face.shape)
face_cascade = cv.CascadeClassifier('C:/Users/sj/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
img = cv.imread('sj49.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
(x, y, w, h)=(0,0,0,0)
with concurrent.futures.ThreadPoolExecutor() as executor:
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
frame = img[y:y+h, x:x+w]
cv.imwrite('face.jpg', frame)

time1=time.time()

img = cv.imread("face.jpg")
H,W = img.shape[:2]
gray = np.zeros((H,W), np.uint8)
with concurrent.futures.ThreadPoolExecutor() as executor:
    for i in range(H):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for j in range(W):
                gray[i,j] = np.clip(0.07 * img[i,j,0]  + 0.72 * img[i,j,1] + 0.21 * img[i,j,2], 0, 255)
cv.imwrite('facegray.jpg', gray)



face = misc.imread('sj149.jpg', cv.IMREAD_UNCHANGED)
print (face.shape)
face_cascade = cv.CascadeClassifier('C:/Users/sj/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
img = cv.imread('sj149.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
with concurrent.futures.ThreadPoolExecutor() as executor:
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
frame = img[y:y+h, x:x+w]
cv.imwrite('FaceinDataBase.jpg', frame)


img = cv.imread("FaceinDataBase.jpg")
H,W = img.shape[:2]
gray = np.zeros((H,W), np.uint8)
with concurrent.futures.ThreadPoolExecutor() as executor:
    for i in range(H):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for j in range(W):
                gray[i,j] = np.clip(0.07 * img[i,j,0]  + 0.72 * img[i,j,1] + 0.21 * img[i,j,2], 0, 255)
cv.imwrite('FaceinDataBasegray.jpg', gray)
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

def compareimagepercntage(n_m,img1):
    return n_m/img1.size*100<25
img1=cv.imread("FaceinDataBasegray.jpg")
img2=cv.imread("Facegray.jpg")
x=len(np.array(img1))
y=len(np.array(img2))

width = int(img1.shape[1]*y/x)
height = int(img1.shape[0]*y/x)
dim = (width, height)
# resize image
resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
cv.imwrite('NewSizedImageGray.jpg',resized) 

img1=cv.imread("NewSizedImageGray.jpg")
n_m, n_0 = compare_images(img1, img2)
print("Photo norm:", n_m, "/ per pixel:", n_m/img1.size)
print("Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size)

if(Time.compareimagepercentage(n_m,img1) or Imgcom):
    print("Images Are of same Person!!!!!Hurray...!!!")
else:
    print("Differnt People")

time2=time.time()

print(Time.timediffpar(time2,time1))






