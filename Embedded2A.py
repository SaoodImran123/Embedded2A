# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:11:33 2021

@author: Saood
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
# Opening image
start = time.time()
vidfr = cv2.VideoCapture('video.mp4')
if vidfr.isOpened():
    success,image = vidfr.read()
count = 0
while success and image is not None:
    cv2.imwrite("frame%d.jpg" % count, image)
    sucess,image = vidfr.read()
    count += 1
all_frames = []
all_framesRGB = []
frame  = cv2.imread("frame0.jpg")
frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
all_frames.append(frame)
all_framesRGB.append(frameRGB)
stop_data = cv2.CascadeClassifier("upper_body.xml")

for vars in range(1,count):
    filename = "frame" + str(vars) + ".jpg"
    img = cv2.imread(filename)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_frames.append(img)
    all_framesRGB.append(imgrgb)
    found = stop_data.detectMultiScale(all_framesRGB[vars],minSize=(20,20))
    amount_found = len(found)
    if amount_found != 0:
        for (x,y,width,height) in found:
            all_framesRGB[vars] = cv2.rectangle(all_framesRGB[vars], (x,y),(x+height,y+width),(0,255,0),5)



height,width,layers = all_frames[0].shape
size = (width,height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter("newmovie.mp4", fourcc, 20.0, (width, height))
for i in range(len(all_framesRGB)):
    out.write(all_framesRGB[i])
vidfr.release()
out.release()
cv2.destroyAllWindows()
end = time.time()
print(f"Runtime totalled: {end - start}")
