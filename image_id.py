#%%
import numpy as np
import cv2
import dlib
import time
from scipy.spatial import distance as dist
from imutils import face_utils
import csv
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd


def cal_yawn(shape):
	top_lip = shape[50:53]
	top_lip = np.concatenate((top_lip, shape[61:64]))

	low_lip = shape[56:59]
	low_lip = np.concatenate((low_lip, shape[65:68]))

	top_mean = np.mean(top_lip, axis=0)
	low_mean = np.mean(low_lip, axis=0)

	distance = dist.euclidean(top_mean,low_mean)
	return distance

def cal_eye(shape):
    distance=[]
    top_left_eye=shape[36:40]
    bottom_left_eye=shape[41:42]
    top_left_mean=np.mean(top_left_eye,axis=0)
    bottom_left_mean=np.mean(bottom_left_eye,axis=0)

    distance.append(dist.euclidean(top_left_mean,bottom_left_mean))


    top_right_eye=shape[43:46]
    bottom_right_eye=shape[47:48]
    top_right_mean=np.mean(top_right_eye,axis=0)
    bottom_right_mean=np.mean(bottom_right_eye,axis=0)

    distance.append(dist.euclidean(top_right_mean,bottom_right_mean))

    left_eye_aspect_ratio=0
    right_eye_aspect_ratio=0

    A=dist.euclidean(shape[37],shape[41])
    B=dist.euclidean(shape[38],shape[40])
    C=dist.euclidean(shape[36],shape[39])
    left_eye_aspect_ratio=(A+B)/(2*C)
    distance.append(left_eye_aspect_ratio)

    A=dist.euclidean(shape[43],shape[47])
    B=dist.euclidean(shape[44],shape[46])
    C=dist.euclidean(shape[42],shape[45])
    right_eye_aspect_ratio=(A+B)/(2*C)
    distance.append(right_eye_aspect_ratio)

    return distance

cam = cv2.imread("eyes_open.jpeg", cv2.IMREAD_COLOR)
#cam=cv2.imread("eyes_closed.jpeg", cv2.IMREAD_COLOR)


def drawgraph(i):
	global count
	count=count+1
	x.append(count)
	y.append(eye_dist)

	plt.cla()
	plt.plot(x,y)
        
eye_dist=0
count=0
x=[]
y=[]


anima=animation.FuncAnimation(plt.gcf(),drawgraph,interval=1000)

plt.show()


face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

yawn_thresh = 35
ptime = 0
img_gray = cv2.cvtColor(cam,cv2.COLOR_BGR2GRAY)
faces = face_model(img_gray)

for face in faces:
    shapes = landmark_model(img_gray,face)
    shape = face_utils.shape_to_np(shapes)
    
    #Left eye points have been identified from shapes[36:42]
    left_eye = shape[36:42]
    #right eye points have been identified from shapes[42:47]
    right_eye=shape[42:48]
    mouth = shape[48:59]
    cv2.drawContours(cam,[left_eye,right_eye,mouth],-1,(205, 0, 0),thickness=2)
    

    

lip_dist = cal_yawn(shape)
print("\n---- Parameters regarding mouth ----")
print("\n Distance between top and bottom lip : "+str(lip_dist))
eye_dist=cal_eye(shape)
print("\n---- Parameters regarding both eyes ----\n")
print("Distance between top and bottom left eye lids : "+str(eye_dist[0]))
print("Distance between top and bottom right eye lids : "+str(eye_dist[1]))
print("EAR left eye : "+str(eye_dist[2]))
print("EAR right eye : "+str(eye_dist[3]))

earThreshold=(eye_dist[2]+eye_dist[3])/2

# From data acquired from a research article 
if(earThreshold)<0.3:
    print("Eyes closed")
print("\n")
	
if lip_dist > yawn_thresh :
		#cv2.putText(cam, f'User Yawning!',(cam.shape[1]//2 - 170 ,cam.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200),2)
        print("Person yawning")

cv2.imshow('Webcam' , cam)
cv2.waitKey(15000)
cv2.destroyAllWindows()
# %%
