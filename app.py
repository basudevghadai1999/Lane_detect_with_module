import streamlit as st
import numpy as np
import cv2
import pickle
import tempfile
from warp import perspective_warp,inv_perspective_warp
from sliding_window_and_get_hist import sliding_window
from  get_curve import get_curve
from pipeline import pipeline,draw_lanes
import socket




def get_hist(img):
    hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
    return hist


left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []







def vid_pipeline(img):
    global running_avg
    global index
    img_ = pipeline(img)
    img_ = perspective_warp(img_)
    out_img, curves, lanes, ploty = sliding_window(img_, draw_windows=True)
    curverad = get_curve(img, curves[0], curves[1])
    lane_curve = np.mean([curverad[0], curverad[1]])
    img = draw_lanes(img, curves[0], curves[1])

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontColor = (0, 255, 0)
    fontSize = 0.80

    cv2.putText(img, 'Lane Curvature: {:.0f} m'.format(lane_curve), (900, 50), font, fontSize, fontColor, 2)
    cv2.putText(img, 'Vehicle offset: {:.4f} m'.format(curverad[2]), (900, 100), font, fontSize, fontColor, 2)
    #if curverad[2] > .25:
        #cv2.putText(img, 'Turn Left', (940, 175), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 0), 3)
    #elif curverad[2] < -.25:
        #cv2.putText(img, 'Turn Right', (940, 175), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 0), 3)
    #else:
        #cv2.putText(img, 'Go Stright', (940, 175), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 3)
    #cv2.putText(img, "Temp: " + str(weather("bhubaneswar")[0]) + "'F" + " Sky: " + str(weather("bhubaneswar")[1]),
                #(885, 395), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 0), 1)
    cv2.putText(img, "Project By-", (20, 20), font, .4, (0, 0, 0), 1)
    cv2.putText(img, "ADIT-IBM Team-72 ", (20, 50), font, .95, (0, 0, 0), 2)
    return img


def dashboard(img):
    shapes = np.zeros_like(img, np.uint8)
    cv2.rectangle(shapes, (875, 10), (1270, 220), (100, 100, 100), cv2.FILLED)

    out = img.copy()
    alpha = 0.5
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]

    return out




# Streamlit Code Start from Here

st.set_page_config(layout="wide")
c1, c2, c3 = st.columns([3, 6, 2])
with c2:
    st.title("Self Driving Car Lane Detection System")
c4, c5 = st.columns(2)
with c4:
    f = st.file_uploader("Upload file")
if f is not None:
    if st.button("Submit"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())
        c6, c7 = st.columns(2)
        with c6:
            st.video(f, format="video/mp4", start_time=0)
        with c7:
            frame_window = st.image([])
        if frame_window is None:
            st.write("empty")
        kill = st.button("Stop")

        cap = cv2.VideoCapture(tfile.name)

        while f:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame1 = dashboard(frame)
                frame2 = vid_pipeline(frame1)
                #frames1= car(frames)

                # cv2.imshow('frame', frames2)
                frame_window.image(frame2)
                if kill:
                    break
        cap.release()
        cv2.destroyAllWindows()



