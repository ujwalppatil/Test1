import streamlit as st
st.write("hello feature one")

import cv2 
import matplotlib.pyplot as plt

weights = cv2.dnn.readNetFromTensorflow("yolo\graph_opt.pb")
Width = 1280
Height = 720
th = 0.7

PARTS = { "Nose": 0, "Neck": 1, "RightShoulder": 2, "RightElbow": 3, "RightWrist": 4,
             "LeftShoulder": 5, "LeftElbow": 6, "LeftWrist": 7, "RightHip": 8, "RightKnee": 9,
             "RightAnkle": 10, "LeftHip": 11, "LeftKnee": 12, "LeftAnkle": 13, "RightEye": 14,
             "LeftEye": 15, "RightEar": 16, "LeftEar": 17, "Background": 18 }

PAIRS = [ ["Neck", "RightShoulder"], ["Neck", "LeftShoulder"], ["RightShoulder", "RightElbow"],
             ["RightElbow", "RightWrist"], ["LeftShoulder", "LeftElbow"], ["LeftElbow", "LeftWrist"],
             ["Neck", "RightHip"], ["RightHip", "RightKnee"], ["RightKnee", "RightAnkle"], ["Neck", "LeftHip"],
             ["LeftHip", "LeftKnee"], ["LeftKnee", "LeftAnkle"], ["Neck", "Nose"], ["Nose", "RightEye"],
             ["RightEye", "RightEar"], ["Nose", "LeftEye"], ["LeftEye", "LeftEar"] ]


st.title("Webcam Live Feed")
run = st.checkbox('Run', value=True)
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
camera.set(60, 30)
camera.set(3, 1280)
camera.set(4, 720)

# if not camera.isOpened():
#         camera = cvv.VideoCapture(0)
# if not camera.isOpened():
#         raise IOError("Cannot Open Webcam")

while cv2.waitKey(1) < 0:
    _, frame = camera.read()

    # _, image = capture.read()
    if not _:
        cv2.waitKey()
        break


    IWidth = frame.shape[1]
    IHeight = frame.shape[0]
    weights.setInput(cv2.dnn.blobFromImage(frame, 1.0, (Width, Height), (127.5, 127.5, 127.5), swapRB = True, crop = False))
    o = weights.forward()
    o = o[:, :19, :, :]
    assert(len(PARTS) == o.shape[1])

    pnts = []
    for i in range(len(PARTS)):
        Map = o[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(Map)
        X = (IWidth * point[0]) / o.shape[3]
        Y = (IHeight * point[1]) / o.shape[2]
        pnts.append((int(X), int(Y)) if conf > th else None)

    for pair in PAIRS:
        partF = pair[0]
        partT = pair[1]
        assert(partF in PARTS)
        assert(partT in PARTS)

        idF = PARTS[partF]
        idT = PARTS[partT]

        if pnts[idF] and pnts[idT]:
            cv2.line(frame, pnts[idF], pnts[idT], (0, 255, 0), 3)
            cv2.ellipse(frame, pnts[idF], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, pnts[idT], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t, _ = weights.getPerfProfile()
    frequency = cv2.getTickFrequency() / 1000
    cv2.putText(frame, '%.2fms' % (t / frequency), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # cv2.imshow('Pose Estimation Using Webcam', frame) 
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')
