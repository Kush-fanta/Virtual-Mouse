import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import random
from pynput.mouse import Button, Controller
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Utility Functions from Util.py ---
def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

# --- Setup ---
mouse = Controller()
screen_width, screen_height = pyautogui.size()
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# --- Gesture Detection Logic ---
def is_index_bent(lm): return get_angle(lm[5], lm[6], lm[8]) < 50
def is_middle_bent(lm): return get_angle(lm[9], lm[10], lm[12]) < 50
def is_ring_bent(lm): return get_angle(lm[13], lm[14], lm[16]) < 50
def is_pinky_bent(lm): return get_angle(lm[17], lm[18], lm[20]) < 50
def is_thumb_near_any(lm, t=50):
    return (get_distance([lm[4], lm[5]]) < t or
            get_distance([lm[4], lm[9]]) < t or
            get_distance([lm[4], lm[13]]) < t)

# --- Streamlit UI ---
st.title("🖐️ Virtual Mouse Control using Hand Gestures")
st.markdown("""
**How to Use:**
- 👉 Move Cursor: Move index finger.
- 🟢 Left Click: Bend Index Finger + Thumb Near.
- 🔴 Right Click: Bend Middle Finger + Thumb Near.
- 🔄 Double Click: Bend Index + Middle Fingers + Thumb Near.
- 📸 Screenshot: Bend All Fingers.
- ✋ Pause Pointer: Bring Thumb near landmark 5, 9, or 13.
""")

screenshot_path = st.empty()

# --- Webcam Stream Processing ---
class VirtualMouse(VideoTransformerBase):
    def __init__(self):
        self.last_screenshot = None

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = hands.process(rgb)

        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            lm = [(pt.x, pt.y) for pt in hand_landmarks.landmark]

            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)

            if len(lm) >= 21:
                thumb_near = is_thumb_near_any(lm)

                if not thumb_near:
                    ix, iy = lm[8]
                    x = int(ix * screen_width)
                    y = int(iy * screen_height)
                    pyautogui.moveTo(x, y)

                # Finger statuses
                idx_bent = is_index_bent(lm)
                mid_bent = is_middle_bent(lm)
                ring_bent = is_ring_bent(lm)
                pinky_bent = is_pinky_bent(lm)

                # Screenshot
                if idx_bent and mid_bent and ring_bent and pinky_bent:
                    self.last_screenshot = f"screenshot_{random.randint(1000,9999)}.png"
                    pyautogui.screenshot(self.last_screenshot)
                    cv2.putText(image, "Screenshot", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

                elif idx_bent and mid_bent and thumb_near:
                    pyautogui.doubleClick()
                    cv2.putText(image, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                elif idx_bent and not mid_bent and thumb_near:
                    mouse.press(Button.left)
                    mouse.release(Button.left)
                    cv2.putText(image, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                elif mid_bent and not idx_bent and thumb_near:
                    mouse.press(Button.right)
                    mouse.release(Button.right)
                    cv2.putText(image, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return image

ctx = webrtc_streamer(key="virtual-mouse", video_transformer_factory=VirtualMouse)

# --- Screenshot Download ---
if ctx.video_transformer and ctx.video_transformer.last_screenshot:
    with open(ctx.video_transformer.last_screenshot, "rb") as file:
        st.download_button(
            label="📥 Download Screenshot",
            data=file,
            file_name=ctx.video_transformer.last_screenshot,
            mime="image/png"
        )
