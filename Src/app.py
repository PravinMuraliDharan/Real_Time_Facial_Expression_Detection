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
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode


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
  * 0 Angry (train - 3995/test - 958),
  * 1 Disgust (train - 436/test - 111),
  * 2 Fear (train - 4097/test - 1024),
  * 3 Happy (train - 7215/test - 1774),
  * 4 Neutral (train - 4965 /test - 1233),
  * 5 Sad (train - 4830 /test - 1247),
  * 6 Surprise (train - 3171/test -  831).

We define a CNN model and compare the predicted results with given labels.""")


# Real Time Facial Expression Detection Demo
st.markdown("""
## Real-Time Demo""")
 
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})



# load model
path = cwd + "/Src/haarcascade_frontalface_default.xml"   # haarcascade file
model_path = cwd + "/Src/Vinnet.h5"                       # Trained Model
vinnet_model = tf.keras.models.load_model(model_path)
faceCascade = cv2.CascadeClassifier(path)


class Face_Expression_Detection(VideoTransformerBase):
    @st.cache(allow_output_mutation=True, max_entries=20, ttl=4600)
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_DUPLEX

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(image=img_gray, scaleFactor=1.1, minNeighbors=4)
        roi_color = None
        for (x, y, w, h) in faces:
          roi_gray = img_gray[y:y+h, x:x+w]
          roi_color = img[y:y+h, x:x+w]
          cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 2)
        if roi_color is not None:
          final_image = cv2.resize(roi_color, (224,224))
          final_image = np.expand_dims(final_image, axis = 0)
          final_image = final_image/255.0
          Predictions = vinnet_model(final_image)                            # Prediction
        # Status = Angry
          if (np.argmax(Predictions) == 0):
            status = "Angry"
            x1,y1,w1,h1 = 0,0,175,75
            cv2.putText(img, status, (100, 150), font, 1.5, (0,0,255),2,cv2.LINE_4)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
        # Status = Disgust
          elif (np.argmax(Predictions) == 1):
            status = "Disgust"
            x1,y1,w1,h1 = 0,0,175,75
            cv2.putText(img, status, (100, 150), font, 1.5, (0,128,128),2,cv2.LINE_4)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
        # Status = Fear
          elif (np.argmax(Predictions) == 2):
            status = "Fear"
            x1,y1,w1,h1 = 0,0,175,75
            cv2.putText(img, status, (100, 150), font, 1.5, (0,0,128),2,cv2.LINE_4)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
        # Status = Happy
          elif (np.argmax(Predictions) == 3):
            status = "Happy"
            x1,y1,w1,h1 = 0,0,175,75
            cv2.putText(img, status, (100, 150), font, 1.5, (0,255,0),2,cv2.LINE_4)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
        # Status = Sad
          elif (np.argmax(Predictions) == 5):
            status = "Sad"
            x1,y1,w1,h1 = 0,0,175,75
            cv2.putText(img, status, (100, 150), font, 1.5, (255,0,0),2,cv2.LINE_4)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
        # Status = Surprise
          elif (np.argmax(Predictions) == 6):
            status = "Surprise"
            x1,y1,w1,h1 = 0,0,175,75
            cv2.putText(img, status, (100, 150), font, 1.5, (0,255,255),2,cv2.LINE_4)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
        # Status = Neutral
          else:
            status = "Neutral"
            x1,y1,w1,h1 = 0,0,175,75
            cv2.putText(img, status, (100, 150), font, 1.5, (255,255,255),2,cv2.LINE_4)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
          
        return img


webrtc_streamer(key="opencv-filter", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,video_processor_factory=Face_Expression_Detection)


