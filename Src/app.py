# Libraries
import os
import cv2
import base64
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib as plt

# Navigation Bar
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://www.linkedin.com/in/pravin-muralidharan-data-scientist/" target="_blank">Real Time Facial Expression Recognition</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://www.linkedin.com/in/pravin-muralidharan-data-scientist/" target="_blank">LinkedIn</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)


# Getting current working directory
cwd = os.getcwd()     
print(cwd)             

# Creating Containers
projectInfo = st.container()
datasetDesc = st.container()


# Sidebar
st.sidebar.title('About')
st.sidebar.write("Get to Know about the author")
st.sidebar.title("Pravin Muralidharan")
st.sidebar.markdown("Big Data Developer | Cognizant Technology Solutions | AL & ML Practitioner | Machine Learning | CNN | MLOps |")
st.sidebar.title("Summary")
st.sidebar.markdown("""
  Pravin Muralidharan is a seasoned engineer with the right blend of data science, deep learning and software engineering. Based out of Tamil Nadu, India.  
  He is a passionate learner and loves to explore and build new things which would contribute as much as possible to society. His skills focused on building enterprise-ready AI solutions.
  Currently working as Big Data Engineer (Programmer Analyst Trainee) in Cognizant Technology Solution, Chennai. 
  Holding Bachelors Degree in Mechanical Engineering with skills focused on CAD and CAE.
""")
st.sidebar.title("Write us")
st.sidebar.write("mrpravin2000@gmail.com")

# Project Information/Overview
with projectInfo:
    st.title('Real Time Facial Expression Recognition')
    st.markdown(""" 
    ## Problem Statement """)
    st.markdown(""" 
    This is a few shot learning live face emotion detection system. The model should be able to real-time
identify the emotions of students in a live class.   """)
    
# Dataset Description
with datasetDesc:
  st.markdown("""
  ## Dataset Description
  Fer2013 dataset consists of 48x48 pixel grayscale images of faces.
  * 0 Happy,
  * 1 Angry,
  * 2 Surprise,
  * 3 Sad,
  * 4 Neutral,
  * 5 Fear,
  * 6 Disgust.

We define a CNN model and compare the predicted results with given labels.""")


# Real Time Facial Expression Detection Demo
st.markdown("""
## Real-Time Demo""")
col1, col2 = st.columns([17,5])
with col1:
  start_demo = st.button('Start Demo')      # Start Demo Button
with col2:
  stop_demo = st.button('Stop Demo')        # Stop Demo Button
FRAME_WINDOW = st.image([])               # Defining Frame Window
if True:
  cap = cv2.VideoCapture(0)               # Accessing Secondary Camera
else:
  cap = cv2.VideoCapture(1)               # Accessing Primary Camera



path = cwd + "\haarcascade_frontalface_default.xml"   # haarcascade file
model_path = cwd + "\Vinnet.h5"                       # Trained Model

# Real-Time Demo                                      
while start_demo:                                      # Starting Demo
  font_scale = 1.5
  font = cv2.FONT_HERSHEY_DUPLEX
  ret, frame = cap.read()
  # Loading trained model and haarcasade
  vinnet_model = tf.keras.models.load_model(model_path)
  faceCascade = cv2.CascadeClassifier(path)
  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray, 1.1,4)
  for x,y,w,h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 2)
    faces = faceCascade.detectMultiScale(roi_gray)
    
  final_image = cv2.resize(roi_color, (224,224))
  final_image = np.expand_dims(final_image, axis = 0)
  final_image = final_image/255.0
  Predictions = vinnet_model(final_image)                            # Prediction
  # Status = Angry
  if (np.argmax(Predictions) == 0):
    status = "Angry"
    x1,y1,w1,h1 = 0,0,175,75
    cv2.putText(frame, status, (100, 150), font, 1.5, (0,0,255),2,cv2.LINE_4)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)
  # Status = Disgust
  elif (np.argmax(Predictions) == 1):
    status = "Disgust"
    x1,y1,w1,h1 = 0,0,175,75
    cv2.putText(frame, status, (100, 150), font, 1.5, (0,128,128),2,cv2.LINE_4)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)
  # Status = Fear
  elif (np.argmax(Predictions) == 2):
    status = "Fear"
    x1,y1,w1,h1 = 0,0,175,75
    cv2.putText(frame, status, (100, 150), font, 1.5, (0,0,128),2,cv2.LINE_4)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)
  # Status = Happy
  elif (np.argmax(Predictions) == 3):
    status = "Happy"
    x1,y1,w1,h1 = 0,0,175,75
    cv2.putText(frame, status, (100, 150), font, 1.5, (0,255,0),2,cv2.LINE_4)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)
  # Status = Sad
  elif (np.argmax(Predictions) == 5):
    status = "Sad"
    x1,y1,w1,h1 = 0,0,175,75
    cv2.putText(frame, status, (100, 150), font, 1.5, (255,0,0),2,cv2.LINE_4)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)
  # Status = Surprise
  elif (np.argmax(Predictions) == 6):
    status = "Surprise"
    x1,y1,w1,h1 = 0,0,175,75
    cv2.putText(frame, status, (100, 150), font, 1.5, (0,255,255),2,cv2.LINE_4)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)
  # Status = Neutral
  else:
    status = "Neutral"
    x1,y1,w1,h1 = 0,0,175,75
    cv2.putText(frame, status, (100, 150), font, 1.5, (255,255,255),2,cv2.LINE_4)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)

  frames = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # Converting BGR to RGB
  FRAME_WINDOW.image(frames)                         # Showing results in the frame window
  

if stop_demo:                                        # Stopping the demo
  cap.release()                                      # Releasing the camera
  cv2.destroyAllWindows()


