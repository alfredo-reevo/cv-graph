import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)

print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

vertices = np.empty((0, 2), dtype=int)
is_pinching = False
eraser_radius = 25

mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    for x, y in vertices:
        cv2.circle(frame, (x, y), 12, (255, 0, 0), cv2.FILLED)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

            # Retrieve coordinates for index finger tip and thumb tip
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Eraser functionality
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

            # Find the palm (centre point of the eraser)
            palm = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            palm_x, palm_y = int(palm.x * frame.shape[1]), int(palm.y * frame.shape[0])

            open_hand = (
                index_tip.y < index_pip.y and
                middle_tip.y < middle_pip.y and
                ring_tip.y < ring_pip.y and
                pinky_tip.y < pinky_pip.y
            )

            if open_hand:
                cv2.circle(frame, (palm_x, palm_y), eraser_radius, (0, 255, 255), cv2.FILLED)
                
                eraser_centre = np.array([palm_x, palm_y])

                vertices = [v for v in vertices if np.linalg.norm(v - eraser_centre) > eraser_radius]

                is_pinching = False
            else:
                thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

                length = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array([index_x, index_y]))

                cx, cy = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2 

                if length < 25:
                    # pinch gesture detected
                    if not is_pinching:
                        vertices = np.append(vertices, [[cx, cy]], axis=0) if vertices.size else np.array([[cx, cy]])
                        is_pinching = True
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                else:
                    # no pinch gesture
                    is_pinching = False
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)



    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key != 255:
        print(f"Key pressed: {chr(key)} (ASCII: {key})")

    if key == ord('q'):
        break
    elif key == ord('c'):
        vertices = np.array([])

cap.release()
cv2.destroyAllWindows()