import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import joblib
from anyio import key
from sklearn.ensemble import RandomForestClassifier

mp_drawing = mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

model_path = 'D:/project/infosys_project/model/Trained-model.pkl'

try:
    model = joblib.load(model_path)
    if not isinstance(model, RandomForestClassifier):
        raise TypeError("Expected a RandomForestClassifier object. Check your model file.")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def detect_keypoints(frame):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z if hasattr(landmark, 'z') else None,
                    'visibility': landmark.visibility,
                })
        return keypoints

def calculate_distances(keypoints):
    distances = {
        'Distance Left Shoulder-Elbow': np.linalg.norm(
            np.array([keypoints[11]['x'], keypoints[11]['y']]) - np.array(
                [keypoints[13]['x'], keypoints[13]['y']])),
        'Distance Left Elbow-Wrist': np.linalg.norm(
            np.array([keypoints[13]['x'], keypoints[13]['y']]) - np.array(
                [keypoints[15]['x'], keypoints[15]['y']])),
        'Distance Right Shoulder-Elbow': np.linalg.norm(
            np.array([keypoints[12]['x'], keypoints[12]['y']]) - np.array(
                [keypoints[14]['x'], keypoints[14]['y']])),
        'Distance Right Elbow-Wrist': np.linalg.norm(
            np.array([keypoints[14]['x'], keypoints[14]['y']]) - np.array(
                [keypoints[16]['x'], keypoints[16]['y']])),
        'Distance Left Hip-Knee': np.linalg.norm(
            np.array([keypoints[23]['x'], keypoints[23]['y']]) - np.array(
                [keypoints[25]['x'], keypoints[25]['y']])),
        'Distance Left Knee-Ankle': np.linalg.norm(
            np.array([keypoints[25]['x'], keypoints[25]['y']]) - np.array(
                [keypoints[27]['x'], keypoints[27]['y']])),
        'Distance Right Hip-Knee': np.linalg.norm(
            np.array([keypoints[24]['x'], keypoints[24]['y']]) - np.array(
                [keypoints[26]['x'], keypoints[26]['y']])),
        'Distance Right Knee-Ankle': np.linalg.norm(
            np.array([keypoints[26]['x'], keypoints[26]['y']]) - np.array(
                [keypoints[28]['x'], keypoints[28]['y']])),
    }
    return distances

def calculate_angle(point1, point2, point3):
    angle_rad = np.arctan2(point3['y'] - point2['y'], point3['x'] - point2['x']) - np.arctan2(
        point1['y'] - point2['y'], point1['x'] - point2['x'])
    angle_deg = np.abs(np.degrees(angle_rad))
    return angle_deg

def calculate_angles(keypoints):
    angles = {
        'Angle Left Shoulder-Elbow-Wrist': calculate_angle(keypoints[11], keypoints[13], keypoints[15]),
        'Angle Right Shoulder-Elbow-Wrist': calculate_angle(keypoints[12], keypoints[14], keypoints[16]),
        'Angle Left Knee-Hip-Ankle': calculate_angle(keypoints[25], keypoints[23], keypoints[27]),
        'Angle Left Hip-Knee-Ankle': calculate_angle(keypoints[23], keypoints[25], keypoints[27]),
    }
    return angles

def process_video(video_path):
    frames = extract_frames(video_path)
    data = []

    for frame in frames:
        keypoints = detect_keypoints(frame)
        if keypoints:
            distances = calculate_distances(keypoints)
            angles = calculate_angles(keypoints)
            data.append({
                'Frame': frame,
                'Keypoints': keypoints,
                'Distances': distances,
                'Angles': angles,
            })

    return data

def predict_on_frames(frames_data, model):
    predictions = []
    for frame_data in frames_data:
        distances = frame_data['Distances']
        angles = frame_data['Angles']

        features = np.array([distances[key] for key in sorted(distances.keys())] +
                            [angles[key] for key in sorted(angles.keys())]).reshape(1, -1)

        prediction = model.predict(features)
        predictions.append(prediction[0])
        final_prediction = determine_final_prediction(predictions)

    return final_prediction
def determine_final_prediction(predictions):

    from collections import Counter
    counts = Counter(predictions)
    final_prediction = counts.most_common(1)[0][0]

    return final_prediction


# Streamlit app code
mp_drawing = mp.solutions.drawing_utils

st.title('Fitness Tracker with Pose Estimation')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Fitness-Tracker Sidebar')
st.sidebar.subheader('Parameters')

#choosing the app mode
app_mode = st.sidebar.selectbox('Choose the App Mode', ['About the App', 'Run on Video'])

# User Profile
st.sidebar.subheader('User Profile')
# Placeholder for user profile input fields
st.sidebar.text_input('Name')
st.sidebar.number_input('Age', min_value=0)
st.sidebar.number_input('Weight (kg)', min_value=0)
st.sidebar.number_input('Height (cm)', min_value=0)

@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


with st.sidebar.expander('Fitness Recommendations'):
    st.write("""
    1. **Warm-Up and Cool-Down:** Always start your workout with a warm-up and end with a cool-down.
    2. **Progressive Overload:** Gradually increase the intensity of your workouts.
    3. **Rest and Recovery:** Allow your muscles time to recover.
    4. **Balanced Routine:** Include a balance of strength training, cardio, and flexibility exercises.
    5. **Hydration and Nutrition:** Stay hydrated and eat a balanced diet.
    6. **Proper Form:** Focus on maintaining proper form during exercises.
    7. **Consistency is Key:** Stay consistent with your workout routine.
    """)

if app_mode == 'About the App':
    st.markdown('This Fitness Tracker app leverages advanced pose estimation technology using **Mediapipe** to analyze workout videos. By uploading a video, the app detects keypoints, calculates distances and angles between body parts, and provides real-time feedback on exercise form. Additionally, it offers personalized fitness recommendations to enhance your workout routine and ensure proper technique.')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 350px;
            margin-left: -350px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.video('https://youtu.be/oaK74yozU9g?si=aYKuhdLX3jyzM87g')

elif app_mode == 'Run on Video':
    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 350px;
            margin-left: -350px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    use_webcam = st.sidebar.checkbox('Use Webcam')
    if use_webcam:
        st.write("Webcam feature coming soon!")  # Placeholder for webcam integration
    else:
        # Unique key for the file uploader widget
        uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video_uploader")

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            st.sidebar.video(uploaded_video)

            # Load your machine learning model
            model_path = 'D:/project/infosys_project/model/Trained-model.pkl'  # Replace with your model path
            model = joblib.load(model_path)
            if isinstance(model, RandomForestClassifier):  # Adjust based on your model type
                print("Model loaded successfully.")
            else:
                raise TypeError("Expected a RandomForestClassifier object. Check your model file.")

            st.write("Processing video...")
            frames_data = process_video(video_path)
            predictions = predict_on_frames(frames_data, model)

            st.write("Predictions:")
            st.write(predictions)