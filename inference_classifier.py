import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up the hands model for video mode
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# Label dictionary for predicted gestures
labels_dict = {0: 'HELLO', 1: 'I LOVE YOU', 2: 'PLEASE', 3: 'GOODBYE', 4: 'SORRY', 5: 'THANK YOU', 6: 'NO', 7: 'YES'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert the BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the detected hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract hand landmark coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize landmarks by setting minimum values to zero
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Ensure we only extract 42 features: 21 landmarks * 2 (x and y coordinates)
            if len(data_aux) > 42:
                data_aux = data_aux[:42]

        # Calculate bounding box for the detected hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Make a prediction using the trained model
        prediction = model.predict([np.asarray(data_aux)])

        # Get the predicted gesture
        predicted_character = labels_dict[int(prediction[0])]

        # Draw a bounding box around the hand and display the predicted gesture
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Sign Language Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
