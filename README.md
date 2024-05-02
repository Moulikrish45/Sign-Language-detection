# Real-time Hand Gesture Recognition

This project demonstrates real-time hand gesture recognition using a pre-trained machine learning model and live video input from a camera. It includes scripts for training the model, collecting data, and performing inference.

## Scripts Overview

1. `dataset_collection.py`: Collects hand gesture data by capturing images from a camera and extracting hand landmarks using the MediaPipe library. The data is saved in a pickle file.

2. `training_classifier.py`: Trains a Random Forest classifier on the collected hand gesture data. It splits the data into training and testing sets, trains the model, evaluates its performance, and saves the trained model.

3. `inference_classifier.py`: Performs real-time inference using the trained model. It captures live video from the camera, detects hand landmarks, predicts the hand gesture using the trained model, and overlays the predicted gesture on the video stream.

## Setup and Requirements

To run these scripts, you need:

- Python 3.x
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy
- scikit-learn (`sklearn`)

## Sign Language Translation For American Alphabets

This project can be extended for sign language translation of American ABCD. By training the model with hand gestures representing American sign language alphabet, real-time inference can be used to translate hand gestures into corresponding letters. This can aid communication for individuals who are deaf or hard of hearing.

## Usage

1.Data Collection: Run dataset_collection.py to capture hand gesture data. Make sure to adjust the DATA_DIR variable to specify the directory where the data will be saved.
2.Model Training: Run training_classifier.py to train a Random Forest classifier on the collected data. The trained model will be saved in a file named 'model.p'.
3.Real-time Inference: Run inference_classifier.py to perform real-time hand gesture recognition using the trained model. Ensure that the camera index specified in the script (2 by default) corresponds to the correct camera device.

You can install the required Python packages using pip:

```bash
pip install -r requirements.txt