import mediapipe as mp
import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

folder = "C:/Users/sujatha/Desktop/Emotion/"  # Use forward slashes or escape the backslashes
counter = 0
facemesh = mp.solutions.face_mesh
face = facemesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.6, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils

while True:
    _, frm = cap.read()

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    op = face.process(rgb)
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            print(i.landmark[0].y * 480)
            draw.draw_landmarks(frm, i, facemesh.FACEMESH_CONTOURS, landmark_drawing_spec=draw.DrawingSpec(color=(0, 0, 255), circle_radius=1))

    cv2.imshow("window", frm)

    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        timestamp = time.time()
        cv2.imwrite(f"{folder}image_{counter}_{timestamp}.jpg", frm)
        time.sleep(0.5)  # Wait for 0.5 seconds to avoid multiple captures for the same timestamp

    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
