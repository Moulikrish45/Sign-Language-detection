import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hands module for static image mode
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory where data is stored
DATA_DIR = './data'

# Initialize data and labels list
data = []
labels = []

# Loop through each directory (representing a label/class)
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image file in the directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        # Read the image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hands
        results = hands.process(img_rgb)
        
        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract and normalize hand landmark coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize landmarks by setting min values to zero
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append processed data and corresponding label
            data.append(data_aux)
            labels.append(dir_)

# Save processed data and labels into a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data preparation completed. Processed data saved to 'data.pickle'.")
