# -*- coding: utf-8 -*-
"""
Created on Oct 2023

@author: lakshay
"""

import cv2
import os
import sys
sys.path.append(os.getcwd())

# Load 
eye_cascade = cv2.CascadeClassifier('./eye_feature_extractor/frontalEyes35x16.xml')

# Glass image
glassImg = cv2.imread('./glasses.png', cv2.IMREAD_UNCHANGED)
glassImg = cv2.cvtColor(glassImg,cv2.COLOR_BGRA2RGBA)

## Video stream
cam = cv2.VideoCapture(0) # O means use the default webcam

while True:
    
    bolVal, frame = cam.read()
    # grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    
    if bolVal == False:     # If frame is not capture then bolVal is False
        continue
    
    # Show captured frame
    cv2.imshow('Frame', frame)

    # When Detect Eyes in a frame
    eyes = eye_cascade.detectMultiScale(frame, 1.3, 5)
    try:
        if len(eyes[0]) == 4:
            ex,ey,ew,eh = eyes[0]

            # cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (255, 0, 0), 2)
            ## Putting Glasses
            glassImg = cv2.resize(glassImg, (ew,eh))

            for i in range(glassImg.shape[0]):
                for j in range(glassImg.shape[1]):
                    if(glassImg[i,j,3]>0):
                        frame[ey+i, ex+j,:]=glassImg[i,j,:-1]
            cv2.imshow('Glasses Overlay', frame)  
            cv2.imwrite(f"./Frame_Glasses.jpg", frame)  
            cv2.waitKey(0)
            cv2.destroyAllWindows() 
            break
    except:
        pass
