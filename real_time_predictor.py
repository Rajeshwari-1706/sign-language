import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

MODEL_PATH = 'best_static_asl_model.h5'  
ACTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please check that 'best_static_asl_model.h5' is in the current directory.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Converts MediaPipe results into the 126-feature vector format expected by the DNN."""
    # Initialize with zeros for fixed 126 features (63 per hand)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    
    if results.multi_hand_landmarks:
        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_idx].classification[0].label
            
            # Flatten the (x, y, z) coordinates for 21 landmarks
            keypoints = np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()

            # Assign based on MediaPipe's label (Right means the signer's left hand)
            if handedness == 'Left':

                lh = keypoints
            elif handedness == 'Right':
               
                rh = keypoints
                
    # Concatenate to form the final 126-dimension input vector
    return np.concatenate([lh, rh])


print("Starting webcam feed...")
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip the frame horizontally for a mirrored, more intuitive view
        frame = cv2.flip(frame, 1)

        # Recolor image to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection (This is the slow part)
        results = hands.process(image)

        # Recolor back to BGR for OpenCV rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        predicted_sign = "WAITING..."
        confidence = 0.0

        if results.multi_hand_landmarks:
            
            # Draw hand landmarks (optional visual feedback)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
            # --- Keypoint Extraction & Prediction ---
            keypoints = extract_keypoints(results)
            
            # Only predict if keypoints are NOT all zeros (i.e., hands were detected)
            if np.any(keypoints):
                # Reshape keypoints to (1, 126) for model input
                input_data = np.expand_dims(keypoints, axis=0) 
                
                # Predict
                res = model.predict(input_data, verbose=0)[0]
                predicted_index = np.argmax(res)
                predicted_sign = ACTIONS[predicted_index]
                confidence = res[predicted_index]

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        
        display_text = f"Sign: {predicted_sign} | Conf: {confidence:.2f}"
        cv2.putText(image, display_text, (5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the processed image
        cv2.imshow('ASL Real-Time Detector', image)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed and resources released.")
