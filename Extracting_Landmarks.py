#!/usr/bin/env python
# coding: utf-8

# ## Extracting Keypoints or Landmarks to create custom Dataset using MediaPipe

# #### 1. Install and Import all the Required Libraries

# In[1]:


import tensorflow as tf
print(tf.__version__)

# In[2]:


import mediapipe as mp
print(mp.__version__)

# In[3]:


import cv2 
import numpy as np 
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp

# ### 2. Capturing Keypoints using MP Holistic

# In[4]:


# using MediaPipe holistic

#Importing mediapipe holistic model to make landmark detection
mp_holistic = mp.solutions.holistic 

#importing MediaPipe Drawing Utitlities to draw landmarks 
mp_drawing = mp.solutions.drawing_utils

# In[5]:


#Defining mediapipe detection function 

def mediapipe_detection(image , model):
  
  image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
  image.flags.writeable = False   #making sure accidental changes to the image data is not made
  results = model.process(image)    # Make landmark predictions
  image.flags.writeable = True
  image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)

  return image , results

# In[6]:


def drawing_landmarks(image , results):
 mp_drawing.draw_landmarks(image , results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
 mp_drawing.draw_landmarks(image , results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
 mp_drawing.draw_landmarks(image , results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
 mp_drawing.draw_landmarks(image , results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# In[7]:


def drawing_styled_landmarks(image , results):
 mp_drawing.draw_landmarks(image , results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
 mp_drawing.DrawingSpec(color=(80,110,10) , thickness= 1 , circle_radius=1),
 mp_drawing.DrawingSpec(color=(80,250,121) , thickness= 1 , circle_radius=1)
 )
 mp_drawing.draw_landmarks(image , results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
 mp_drawing.DrawingSpec(color=(80,22,10) , thickness= 2 , circle_radius=4),
 mp_drawing.DrawingSpec(color=(80,44,121) , thickness= 2 , circle_radius=2)
 )
 mp_drawing.draw_landmarks(image , results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
 mp_drawing.DrawingSpec(color=(80,22,10) , thickness= 2 , circle_radius=4),
 mp_drawing.DrawingSpec(color=(80,44,121) , thickness= 2 , circle_radius=2)
 )
 mp_drawing.draw_landmarks(image , results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
 mp_drawing.DrawingSpec(color=(80,117,10) , thickness= 2 , circle_radius=4),
 mp_drawing.DrawingSpec(color=(80,66,121) , thickness= 2 , circle_radius=2
 ))



def extract_keypoints(results):

 pose = np.array([[point.x , point.y , point.z , point.visibility] for point in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)

 lh = np.array([[point.x , point.y , point.z] for point in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

 rh = np.array([[point.x , point.y , point.z] for point in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

 face = np.array([[point.x , point.y , point.z] for point in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)

 #Concatenating and returning to represent all the keypoints in a single array

 return np.concatenate([pose , face , lh, rh])


# Expected output = (1404 + 63 + 63 + 132) = 1662

# In[ ]:



