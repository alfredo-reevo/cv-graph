import cv2
import numpy as np
import mediapipe as mp
import threading
from webcam import Webcam

cap = Webcam(0).start()

print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

vertices = np.empty((0, 2), dtype=int)
edges = np.empty((0, 2, 2), dtype=int)
is_pinching = False
source_vertex = None
edge_snap_radius = 30
eraser_radius = 25

mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    for edge in edges:
        cv2.line(frame, tuple(edge[0]), tuple(edge[1]), (255, 255, 255), 2)

    for x, y in vertices:
        cv2.circle(frame, (x, y), 12, (255, 0, 0), cv2.FILLED)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(
            #     frame, 
            #     hand_landmarks, 
            #     mp_hands.HAND_CONNECTIONS
            # )

            # Retrieve coordinates for index finger tip and thumb tip
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            lm = hand_landmarks.landmark
            
            ##* Eraser functionality *##
            
            # Find the palm (centre point of the eraser)
            palm = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            palm_x, palm_y = int(palm.x * frame.shape[1]), int(palm.y * frame.shape[0])

            open_hand = (
                lm[8].y < lm[7].y < lm[6].y < lm[5].y and  # Index finger
                lm[12].y < lm[11].y < lm[10].y < lm[9].y and  # Middle finger
                lm[16].y < lm[15].y < lm[14].y < lm[13].y and  # Ring finger
                lm[20].y < lm[19].y < lm[18].y < lm[17].y  # Pinky finger
            )

            if open_hand:
                # Eraser enabled
                cv2.circle(frame, (palm_x, palm_y), eraser_radius, (0, 255, 255), cv2.FILLED)
                eraser_centre = np.array([palm_x, palm_y])

                if len(edges) > 0:
                    dist_v1 = np.linalg.norm(edges[:, 0] - eraser_centre, axis=1)
                    dist_v2 = np.linalg.norm(edges[:, 1] - eraser_centre, axis=1)

                    edge_mask = (dist_v1 > eraser_radius) & (dist_v2 > eraser_radius)
                    edges = edges[edge_mask]

                if len(vertices) > 0:
                    distances = np.linalg.norm(vertices - eraser_centre, axis=1)
                    mask = distances > eraser_radius
                    vertices = vertices[mask]

                is_pinching = False
                source_vertex = None
            else:
                # Pinch detection (vertex creation and edge snapping)
                thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

                length = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array([index_x, index_y]))

                cx, cy = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2 

                if length < 25:
                    # pinch gesture detected
                    if not is_pinching:
                        is_pinching = True

                        # Detect if the pinch is close to an existing vertex
                        if len(vertices) > 0:
                            distances = np.linalg.norm(vertices - np.array([cx, cy]), axis=1)
                            closest_vertex = np.argmin(distances)

                            if distances[closest_vertex] < edge_snap_radius:
                                source_vertex = tuple(vertices[closest_vertex])
                            else:
                                vertices = np.append(vertices, [[cx, cy]], axis=0)
                        else:
                            vertices = np.append(vertices, [[cx, cy]], axis=0)

                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                    if source_vertex is not None:
                        cv2.line(frame, source_vertex, (cx, cy), (255, 255, 255), 2)
                else:
                    # no pinch gesture
                    if is_pinching and source_vertex is not None:
                        if len(vertices) > 0:
                            distances = np.linalg.norm(vertices - np.array([cx, cy]), axis=1)
                            closest_vertex = np.argmin(distances)

                            if distances[closest_vertex] < edge_snap_radius:
                                target_vertex = tuple(vertices[closest_vertex])
                                if target_vertex != source_vertex:
                                    new_edge = np.array([[source_vertex, target_vertex]])
                                    edges = np.append(edges, new_edge, axis=0)
                    
                    is_pinching = False
                    source_vertex = None
                    
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)



    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key != 255:
        print(f"Key pressed: {chr(key)} (ASCII: {key})")

    if key == ord('q'):
        break
    elif key == ord('c'):
        vertices = np.array([])

cap.stop()
cv2.destroyAllWindows()