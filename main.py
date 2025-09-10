import time
import cv2
import mediapipe as mp

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# ===================== Config =====================
MAX_HANDS = 2
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5

SWIPE_THRESHOLD_PX = 20
PINCH_THRESHOLD_PX = 40
SWIPE_COOLDOWN_S = 2.0
HAND_MISSING_THRESHOLD_S = 0.5

WIN_NAME = "Vision Control"

URL = "" 

# ===================== Setup =====================
def build_driver():
    try:
        driver = webdriver.Chrome()
        driver.get(URL)
        return driver
    except Exception:
        return None

def build_face_detector():
    casc = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(casc)

def draw_hands(image, hand_landmarks, mp_hands, mp_draw, draw_spec):
    mp_draw.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=draw_spec,
        connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
    )

# ===================== Gestures =====================
def only_index_up(hand_landmarks, mp_hands):
    pairs = [
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
    ]
    up = []
    for tip_idx, pip_idx in pairs:
        tip_y = hand_landmarks.landmark[tip_idx].y
        pip_y = hand_landmarks.landmark[pip_idx].y
        up.append(tip_y < pip_y)
    return up == [False, True, False, False, False]

def pinch(hand_landmarks, image, mp_hands):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    h, w, _ = image.shape
    p1 = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    p2 = (int(index_tip.x * w), int(index_tip.y * h))
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5 < PINCH_THRESHOLD_PX, p1, p2

# ===================== Actions =====================
def send_scroll(driver, direction):
    if not driver:
        return
    try:
        body = driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.PAGE_DOWN if direction == "down" else Keys.PAGE_UP)
    except Exception:
        pass

# ===================== Main =====================
def main():
    driver = build_driver()
    face_cascade = build_face_detector()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    draw_spec = mp_draw.DrawingSpec(thickness=2, circle_radius=2, color=(0, 0, 255))

    cap = cv2.VideoCapture(0)
    prev_y = None
    swipe_up = False
    swipe_down = False
    last_swipe_up = 0.0
    last_swipe_down = 0.0
    hand_detected_time = None
    hand_missing_time = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            now = time.time()

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    draw_hands(image, hl, mp_hands, mp_draw, draw_spec)

                    is_pinch, p1, p2 = pinch(hl, image, mp_hands)
                    if is_pinch:
                        cv2.circle(image, p1, 10, (0, 255, 255), cv2.FILLED)
                        cv2.circle(image, p2, 10, (3, 146, 112), cv2.FILLED)
                        cv2.line(image, p1, p2, (0, 255, 255), 2)
                        cv2.putText(image, "Pinch", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    if only_index_up(hl, mp_hands):
                        tip = hl.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        h, w, _ = image.shape
                        cx, cy = int(tip.x * w), int(tip.y * h)
                        cv2.circle(image, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

                        if prev_y is not None:
                            dy = cy - prev_y
                            if dy > SWIPE_THRESHOLD_PX:
                                swipe_down = True
                            elif dy < -SWIPE_THRESHOLD_PX:
                                swipe_up = True
                        prev_y = cy
                    else:
                        prev_y = None

                hand_missing_time = None
                if hand_detected_time is None:
                    hand_detected_time = now
            else:
                if hand_missing_time is None:
                    hand_missing_time = now
                elif now - hand_missing_time > HAND_MISSING_THRESHOLD_S:
                    hand_detected_time = None
                    prev_y = None
                    hand_missing_time = None

            if swipe_down and (now - last_swipe_down > SWIPE_COOLDOWN_S):
                send_scroll(driver, "up")
                cv2.putText(image, "Swipe Down", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                swipe_down = False
                last_swipe_down = now

            if swipe_up and (now - last_swipe_up > SWIPE_COOLDOWN_S):
                send_scroll(driver, "down")
                cv2.putText(image, "Swipe Up", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                swipe_up = False
                last_swipe_up = now

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

            cv2.imshow(WIN_NAME, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    try:
        if driver:
            driver.quit()
    except Exception:
        pass

if __name__ == "__main__":
    main()
