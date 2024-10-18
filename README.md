# Fitness Tracker with Pose Estimation 🏋️‍♂️
[![Watch the video](https://github.com/Sakshi13t/Fitness_Tracker_with_pose_estimation/blob/main/output%20videos/output_video%20(1).mp4)

## Overview

This project is a Fitness Tracker app leveraging pose estimation technology to analyze and track workouts. The app can predict the type of exercise from a video input. The supported exercises are:

- Push-ups
- Squats
- Jumping Jacks
- Pull-ups
- Lunges

## Features

1. **Pose Estimation with Mediapipe**: Detects keypoints in video frames to analyze body posture and movements.
2. **Feature Extraction**: Calculates angles and distances between body keypoints to create a dataset.
3. **Machine Learning Model**: Trained a Random Forest Classifier with hyperparameter tuning to predict exercise types.
4. **Streamlit UI**: Provides a user-friendly interface where users can upload a video and get real-time predictions and feedback.
5. **AWS Integration**: Uses Boto3 for uploading and downloading data to/from an S3 bucket.

## Dataset Creation

1. **Frame Extraction**: Extracted frames from collected exercise videos.
2. **Feature Calculation**: Calculated angles and distances between keypoints for each frame.
3. **Dataset Storage**: Stored the extracted features in a CSV file.

## Model Training

1. **Random Forest Classifier**: Trained the model using scikit-learn's RandomForestClassifier.
2. **Hyperparameter Tuning**: Optimized model performance through hyperparameter tuning.

## Dependencies

- `numpy`
- `scikit-learn`
- `pandas`
- `mediapipe`
- `matplotlib`
- `opencv-python`
- `streamlit`
- `os`
- `tempfile`
- `boto3`
- `joblib`

## AWS Integration

- **Uploading Data to S3**: The app uses Boto3 to upload videos and extracted features to an S3 bucket.
- **Downloading Data from S3**: Similarly, it can download data from the S3 bucket for further processing.

## Acknowledgments

- **Mediapipe**: For the pose estimation solutions.
- **Streamlit**: For the easy-to-use web app framework.
- **AWS**: For providing robust cloud storage solutions.
