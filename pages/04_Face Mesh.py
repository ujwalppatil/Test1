import streamlit as st

import mediapipe as mp
import cv2


import numpy as np
import tempfile
from PIL import Image
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS

#Lets try to integrate streamlit and mediapipe

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# DEMO_VIDEO = 'demo.mp4'
# OUTM = 'output.mp4'
# DEMO_IMAGE = 'demo.jpg'


st.title('Face Mesh Application using MediaPipe')

# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
#         width: 350px;
#     }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
#         width: 350px;
#         margin-left: -350px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


st.sidebar.title('Face Mesh Application using MediaPipe')
st.sidebar.subheader('Parameters')

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized




# app_mode = st.sidebar.selectbox('Choose the App mode',
# ['About App','Run on Video','Run on Image']
# )

# if app_mode =='About App':

#     st.markdown('In this application we are using **MediaPipe** from Google for creating a Face Mesh on a video')  

#     st.text('Demo output for Face mesh')
#     st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
#         width: 350px;
#     }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
#         width: 350px;
#         margin-left: -350px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
#     )
#     dem = open(OUTM,'rb')
#     out_vid = dem.read()

#     st.video(out_vid)

#     st.markdown('''
#           # About Author \n 
#              Hey this is ** Pavan Kunchala ** I hope you like the application \n
#             I am looking for ** Collabration ** or ** Freelancing ** in the field of ** Deep Learning ** and 
#             ** Computer Vision ** \n
#             I am also looking for ** Job opportunities ** in the field of** Deep Learning ** and ** Computer Vision** 
#             if you are interested in my profile you can check out my resume from 
#             [here](https://drive.google.com/file/d/16aKmdHryldvx3OPNwmHhxW-DAoQOypvX/view?usp=sharing)
#             If you're interested in collabrating you can mail me at ** pavankunchalapk@gmail.com ** \n
#             You can check out my ** Linkedin ** Profile from [here](https://www.linkedin.com/in/pavan-kumar-reddy-kunchala/) \n
#             You can check out my ** Github ** Profile from [here](https://github.com/Pavankunchala) \n
#             You can also check my technicals blogs in ** Medium ** from [here](https://pavankunchalapk.medium.com/) \n
#             If you are feeling generous you can buy me a cup of ** coffee ** from [here](https://www.buymeacoffee.com/pavankunchala)
             
#             ''')
# if app_mode =='Run on Video':

st.subheader('We are applying Face Mesh on a video')
# st.set_option('deprecation.showfileUploaderEncoding', False)

# use_webcam = st.sidebar.button('Use Webcam')
    
    # st.sidebar.text('Params For video')
    # st.markdown(
    # """
    # <style>
    # [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    #     width: 400px;
    # }
    # [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    #     width: 400px;
    #     margin-left: -400px;
    # }
    # </style>
    # """,
    # unsafe_allow_html=True,
    #     )
detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    #max faces
max_faces = st.sidebar.number_input('Maximum Number of Faces',value =1,min_value= 1)

st.markdown(' ## Output')
stframe = st.empty()
    

# video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])

    

tfflie = tempfile.NamedTemporaryFile(delete=False)


# if not video_file_buffer:
#     if use_webcam:
#         vid = cv2.VideoCapture(0)
    # else:
    #     vid = cv2.VideoCapture(DEMO_VIDEO)
    #     tfflie.name = DEMO_VIDEO
    
# else:
#     tfflie.write(video_file_buffer.read())
#     vid = cv2.VideoCapture(tfflie.name)
vid = cv2.VideoCapture(0)


    

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = 120
#codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
# codec = cv2.VideoWriter_fourcc('V','P','0','9')
# out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))


# st.sidebar.text('Input Video')
# st.sidebar.video(tfflie.name)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    

with mp_face_mesh.FaceMesh(
min_detection_confidence=detection_confidence,
min_tracking_confidence=tracking_confidence , 
max_num_faces = max_faces) as face_mesh:




    while vid.isOpened():

        ret, frame = vid.read()

        if not ret:
            continue
                
            

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        

        results = face_mesh.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # print("Face Landmarks :",face_landmarks)
                # cv2.putText
                # x,y,z = face_landmarks
                # cv2.putText(frame, (x,y,z), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                mp_drawing.draw_landmarks(
                image = frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        # print(results.multi_face_landmarks)
        # landmarks = {
        #     x,
        #     y,
        #     z
        # }
        # landmarks = results.multi_face_landmarks
        fr = vid.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fr))
        cv2.putText(frame, "FPS :""{0}".format(fr), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        # out.write(frame)    
        frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)

        frame = image_resize(image = frame, width = 640)
            
        stframe.image(frame,channels = 'BGR',use_column_width=True)

    

st.text('Video Processed')

# output_video = open('output1.webm','rb')
# out_bytes = output_video.read()
# st.video(out_bytes)


        





vid.release()
# out.release()


