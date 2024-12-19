#!/usr/bin/env python
# coding: utf-8

# ## Creating a Dataset
# - Setting up folders to store the landmarks
# - Recording the sign languages using a camera

# In[1]:


import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import time
import mediapipe as mp


# 

# In[2]:


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# In[3]:


import sys
print(sys.executable)


# In[4]:


#Converted Extracting_Landmarks into Python script to import functions defined in it
from Extracting_Landmarks import mediapipe_detection , drawing_landmarks , drawing_styled_landmarks , extract_keypoints


# ### 1. Setup Folders for Collections

# Defining the File Path

# In[5]:

# File Path to store the landmark data

DATA_PATH = os.path.join("LANDMARK_DATA")

# Defining the actions 

# File Path to store the landmark data

DATA_PATH = os.path.join("LANDMARK_DATA")

# Defining the actions 
actions = np.array(['hello' , 'thankyou' , 'My' , 'sorry', 'Name' , 'You' , 'I am' , 'Nice' ,'Meet' , 'Fine', 'I'])
# Defining the number of videos per action

no_of_videos = 100

# Defining the number of frames per video

no_of_frames = 30



