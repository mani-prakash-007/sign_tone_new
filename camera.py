import joblib
import cv2
import numpy as np
import mediapipe as mp
import os

# Load model, gesture map, and scaler
model = joblib.load("gesture_model.pkl")
gesture_map = joblib.load("gesture_map.pkl")
scaler = joblib.load("gesture_scaler.pkl")  # ✅ Load the scaler

expected_features = model.n_features_in_

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract features
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                if len(landmark_list) == expected_features:
                    # ✅ Apply scaler to feature vector
                    X = np.array(landmark_list).reshape(1, -1)
                    X_scaled = scaler.transform(X)

                    # Predict with scaled features
                    prediction = model.predict(X_scaled)[0]
                    gesture_name = gesture_map.get(prediction, "Unknown")

                    # Class label resolution
                    gn = ""
                    with open("static/class2.txt", "r") as f:
                        cna = f.read().split("|")
                    with open("static/class1.txt", "r") as f:
                        hna = f.read().split("|")

                    for idx, cname in enumerate(cna):
                        if cname == gesture_name and idx < len(hna):
                            gn = hna[idx]
                            break

                    if gn:
                        with open("static/detect.txt", "w") as f:
                            f.write(gn)
                        # ✅ Draw label per hand
                        cv2.putText(frame, f"{gn} (Hand {i+1})", (10, 50 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
