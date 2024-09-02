import cv2
import numpy as np
import imgaug.augmenters as iaa
import mediapipe as mp
import os

gestures = ["HELLO", "I LOVE YOU", "PLEASE", "GOODBYE", "SORRY", "THANK YOU", "NO", "YES"]

# Create directories for each gesture
for gesture in gestures:
    if not os.path.exists(f'./data/{gesture}'):
        os.makedirs(f'./data/{gesture}')

# Define Image Augmentations
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Affine(rotate=(-25, 25)),  # rotate images
    iaa.Multiply((0.8, 1.2)),  # change brightness
    iaa.GaussianBlur(sigma=(0, 3.0))  # blur images
])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

# Initialize Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

print("Press '1' to record HELLO")
print("Press '2' to record I LOVE YOU")
print("Press '3' to record PLEASE")
print("Press '4' to record GOODBYE")
print("Press '5' to record SORRY")
print("Press '6' to record THANK YOU")
print("Press '7' to record NO")
print("Press '8' to record YES")
print("Press 'q' to quit")

gesture_dirs = {
    ord('1'): 'HELLO',
    ord('2'): 'I LOVE YOU',
    ord('3'): 'PLEASE',
    ord('4'): 'GOODBYE',
    ord('5'): 'SORRY',
    ord('6'): 'THANK YOU',
    ord('7'): 'NO',
    ord('8'): 'YES'
}

current_gesture = None
image_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if current_gesture:
            dataset_dir = f'./data/{current_gesture}'

            # Save original frame
            img_filename = os.path.join(dataset_dir, f'image_{image_count}.jpg')
            cv2.imwrite(img_filename, frame)

            # Image augmentation
            augmented_images = seq(images=[frame])
            for i, aug_img in enumerate(augmented_images):
                aug_img_filename = os.path.join(dataset_dir, f'image_{image_count}_aug_{i}.jpg')
                cv2.imwrite(aug_img_filename, aug_img)

            image_count += 1
            print(f"Captured and augmented image {image_count} for gesture {current_gesture}")

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key in gesture_dirs:
        current_gesture = gesture_dirs[key]
        print(f"Recording gesture: {current_gesture}")
        image_count = 0  # Reset image count for new gesture
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
