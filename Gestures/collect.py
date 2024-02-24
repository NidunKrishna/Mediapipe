import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import math

cap = cv2.VideoCapture(0)

facemesh = mp.solutions.face_mesh
face = facemesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.6, min_detection_confidence=0.6)
draw_face = mp.solutions.drawing_utils

hands = mp.solutions.hands
hands_mesh = hands.Hands(static_image_mode=True, min_detection_confidence=0.7)
draw_hand = mp.solutions.drawing_utils

# Load MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet', include_top=True)

while True:
    _, frm = cap.read()

    rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    result = face.process(rgb_frame)
    rgb_face = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op_face = face.process(rgb_face)
    
    for face_landmarks in result.multi_face_landmarks:
            
        left_eye_landmarks = face_landmarks.landmark[56:77]
        right_eye_landmarks = face_landmarks.landmark[363:474]

        height, width, _ = frm.shape
        left_eye_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in left_eye_landmarks]
        right_eye_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in right_eye_landmarks]

        left_eye_x, left_eye_y, left_eye_w, left_eye_h = cv2.boundingRect(np.array(left_eye_points))
        right_eye_x, right_eye_y, right_eye_w, right_eye_h = cv2.boundingRect(np.array(right_eye_points))

        left_eye_roi = frm[left_eye_y:left_eye_y + left_eye_h, left_eye_x:left_eye_x + left_eye_w]
        right_eye_roi = frm[right_eye_y:right_eye_y + right_eye_h, right_eye_x:right_eye_x + right_eye_w]

        # Preprocess the image for classification
        left_eye_img = cv2.resize(left_eye_roi, (224, 224))  # MobileNetV2 input size
        left_eye_img = image.img_to_array(left_eye_img)
        left_eye_img = np.expand_dims(left_eye_img, axis=0)
        left_eye_img = preprocess_input(left_eye_img)

        # Make predictions for left eye
        left_eye_predictions = model.predict(left_eye_img)
        left_eye_label = decode_predictions(left_eye_predictions, top=1)[0][0][1]

        cv2.putText(frm, f"Left Eye: {left_eye_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Left Eye", left_eye_roi)

        # Repeat the same process for the right eye if needed

    if op_face.multi_face_landmarks:
        for i in op_face.multi_face_landmarks:
            print(i.landmark[0].y * 480)
            draw_face.draw_landmarks(frm, i, facemesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=draw_face.DrawingSpec(color=(0, 0, 255), circle_radius=1))

    rgb_hand = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op_hand = hands_mesh.process(rgb_hand)
    
    if op_hand.multi_hand_landmarks:
        for i in op_hand.multi_hand_landmarks:
            draw_hand.draw_landmarks(frm, i, hands.HAND_CONNECTIONS, 
                landmark_drawing_spec=draw_hand.DrawingSpec(color=(255, 0, 0), circle_radius=4, thickness=3),
                connection_drawing_spec=draw_hand.DrawingSpec(thickness=3, color=(0, 0, 255)))

    height, width, _ = frm.shape
    aspect_ratio = height / width

    if aspect_ratio > 1:
        k = frm.shape[0] / height
        w_cal = math.ceil(k * width)
        frm = cv2.resize(frm, (w_cal, frm.shape[0]))
    cv2.imshow("Combined Window", frm)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
