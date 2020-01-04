import cv2
import shutil
from random import shuffle
import glob
import matplotlib.pyplot as plt
import matplotlib._png as png
import numpy as np
path = "/home/mak/Desktop/Projects/Stroboscopy/Vocal Nodule/Manasa 1/*.png"
dest = "/home/mak/Desktop/Projects/Stroboscopy/Vocal Nodule/Manasa 1/Cropped/"
# read addresses and labels from the 'train' folder
addrs = glob.glob(path)

cropping = False
 
x_start, y_start, x_end, y_end = 0, 0, 0, 0

done =  False

def cropImage(img,cropping,nm):
 
    image = img
    oriImage = image.copy()
     
     
    def mouse_crop(event, x, y, flags, param):
        # grab references to the global variables
        global x_start, y_start, x_end, y_end, cropping,done
     
        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True
     
        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping == True:
                x_end, y_end = x, y
     
        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            x_end, y_end = x, y
            cropping = False # cropping is finished
            done = True
     
            refPoint = [(x_start, y_start), (x_end, y_end)]
     
            if len(refPoint) == 2: #when two points were found
                roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv2.imshow("Cropped", roi)
                print "Cropping"
                roi = cv2.resize(roi,(256,256))
                cv2.imwrite(nm,roi)
     
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)
     
    while True:
     
        i = image.copy()
     
        if not cropping:
            cv2.imshow("image", image)
     
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)

        if done==True:
            print "Image done "
            break
     
        cv2.waitKey(1)
     
    # close all open windows
    cv2.destroyAllWindows()


for i in range(0,len(addrs)):
    addr = addrs[i]
    print addr
    img =  cv2.imread(addr)
    nm = dest +"cropped" + str(i)+".png"
    # cv2.imshow("image",img)
    # cv2.waitKey(1)
    cropImage(img,False,nm)
    cropping = False
 
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    done =  False
