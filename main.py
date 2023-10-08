# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 12:55:16 2022

@author: lakshay
"""

import cv2
import os
import sys
sys.path.append(os.getcwd())

eye_cascade = cv2.CascadeClassifier('./eye_feature_extractor/frontalEyes35x16.xml')

img_path = './Jamie_Before.jpg'
personImg = cv2.imread(img_path)
cv2.imshow('Image', personImg)

# Glass image
glassImg = cv2.imread('./glasses.png', cv2.IMREAD_UNCHANGED)
glassImg = cv2.cvtColor(glassImg,cv2.COLOR_BGRA2RGBA)

# Detect Eyes
eyes = eye_cascade.detectMultiScale(personImg, 1.3, 5)
ex,ey,ew,eh = eyes[0]

# cv2.rectangle(personImg, (ex,ey), (ex+ew,ey+eh), (255, 0, 0), 2)
## Putting Glasses
glassImg = cv2.resize(glassImg,(ew,eh))

for i in range(glassImg.shape[0]):
    for j in range(glassImg.shape[1]):
        if(glassImg[i,j,3]>0):
            personImg[ey+i, ex+j,:]=glassImg[i,j,:-1]
cv2.imshow('Image', personImg)

cv2.imwrite(f"{img_path}_Glasses.jpg", personImg)
cv2.waitKey(0)
cv2.destroyAllWindows() 
