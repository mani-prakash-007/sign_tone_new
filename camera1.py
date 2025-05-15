import cv2
import numpy as np
import mediapipe as mp
import csv
import os
from PIL import Image

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

DATA_DIR = "static/hand_gesture_data"
os.makedirs(DATA_DIR, exist_ok=True)

class VideoCamera1(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, frame = self.video.read()
        frame = cv2.flip(frame, 1)

        with open("static/label.txt", "r") as ff:
            glabel = ff.read().strip()
        with open("static/label1.txt", "r") as ff:
            gfile = ff.read().strip()

        GESTURE_LABEL = glabel
        FILE_NAME = os.path.join(DATA_DIR, f"{gfile}")

        with open(FILE_NAME, mode="a", newline="") as f:
            writer = csv.writer(f)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract landmark data
                    landmark_list = []
                    for lm in hand_landmarks.landmark:
                        landmark_list.extend([lm.x, lm.y, lm.z])

                    writer.writerow(landmark_list)
                    cv2.putText(frame, f"Recording: {GESTURE_LABEL} (Hand {i+1})", (10, 50 + i * 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
