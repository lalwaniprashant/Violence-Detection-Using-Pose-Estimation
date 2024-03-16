import cv2
import tensorflow as tf
import mediapipe as mp
import torch
import numpy as np
import os
from threading import Thread
from flask import Flask, render_template, request, send_from_directory, jsonify
app = Flask(__name__, static_folder='static')

# one timers
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# frame_skip = 5
seq_len = 10

# path_model = r"C:\Users\prash\Desktop\Academic\FYP_Venv\final analysis\LSTM_model & kaggle_only_data\csv3\normal_data\csv3_normal_data_kaggle_only_lstm_model.h5"

path_model = "./csv3_normal_data_kaggle_only_lstm_model.h5"


def EachVideo(linkVideo):
    X = []
    video_path = r"" + linkVideo
    cap = cv2.VideoCapture(video_path)

    frame_skip = 5
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        result = yolo_model(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
            c_lm = []
            with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
                resulta = pose.process(
                    frame[int(ymin):int(ymax), int(xmin):int(xmax):])
                if resulta.pose_landmarks and clas == 0:
                    for (id, lm) in enumerate(resulta.pose_landmarks.landmark):
                        if id > 10 and id not in [17, 18, 19, 20, 21, 22] and id not in [29, 30, 31, 32]:
                            c_lm.append(lm.x)
                            c_lm.append(lm.y)
                            c_lm.append(lm.z)
            if len(c_lm) > 0:
                X.append(c_lm)

    return X
# end of one timer


# separation
current_directory = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(current_directory, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def process_model(video_link):
    # processing
    processed_array = EachVideo(video_link)
    processed_array = np.array(processed_array)

    x_1 = []

    len_processed_array = len(processed_array)

    for i in range(seq_len, len_processed_array):
        x_1.append(processed_array[i - seq_len:i, 1:])

    x_1 = np.array(x_1)

    model = tf.keras.models.load_model(path_model)

    y_pred = model.predict(x_1)

    y_pred_binary = (y_pred >= 0.5).astype(int)

    num_ones = np.sum(y_pred_binary == 1)
    num_zeroes = np.sum(y_pred_binary == 0)

    prediction_output = ""

    if (num_ones > num_zeroes):
        prediction_output = "Violence"
    else:
        prediction_output = "Non Violence"

    print("Predicted output is:", prediction_output)

    confidence = round(np.mean(y_pred) * 100, 2)
    confidence = max(confidence, 100-confidence)

    print("Confidence is:", confidence)
    # end of processing

    return prediction_output, confidence


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global FULL_FILEPATH
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    VIDEO_FILENAME = file.filename
    FULL_FILEPATH = os.path.abspath(os.path.join(
        app.config['UPLOAD_FOLDER'], file.filename))
    print(f"Absolute path: {FULL_FILEPATH}")
    return render_template('play.html', video_path=f"/uploads/{VIDEO_FILENAME}")


@app.route('/get_result', methods=['GET'])
def get_result():
    # Simulated result
    result1, result2 = process_model(FULL_FILEPATH)
    return jsonify({"result1": result1, "result2": result2})


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
