import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration

from landmark_detector import FaceLandmarkDetector
from makeup_renderer import (apply_makeup, MakeupParams, LIPS_PRESETS, BLUSH_PRESETS, EYEBROW_PRESETS,)


st.title("Make-up filter")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ]
    }
)

def show_preset_name(preset):
    return preset[0]

def choose_color(label, presets, index=0):
    choice = st.sidebar.selectbox(
        label,
        presets,
        index=index,
        format_func=show_preset_name,
    )
    return choice[1]


enabled = st.sidebar.checkbox("Uključi šminku", value=True)

params = MakeupParams(
    lips_color=choose_color("Boja usana", LIPS_PRESETS),
    lips_intensity=st.sidebar.slider("Intenzitet usana", 0.0, 1.0, 0.75),

    blush_color=choose_color("Boja rumenila", BLUSH_PRESETS, 1),
    blush_intensity=st.sidebar.slider("Intenzitet rumenila", 0.0, 0.5, 0.35),

    eyebrow_color=choose_color("Boja obrva", EYEBROW_PRESETS, 1),
    eyebrow_intensity=st.sidebar.slider("Intenzitet obrva", 0.0, 1.0, 0.35),
)


class MakeupProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = FaceLandmarkDetector()
        self.params = MakeupParams()
        self.enabled = True

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        landmarks = self.detector.process_frame(image)

        if self.enabled:
            image = apply_makeup(image, landmarks, self.params)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


camera = webrtc_streamer(
    key="makeup-camera",
    video_processor_factory=MakeupProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION,
)

if camera.video_processor:
    camera.video_processor.params = params
    camera.video_processor.enabled = enabled