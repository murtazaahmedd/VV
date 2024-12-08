import os
import cv2
import sqlite3
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_file, Response,url_for,send_from_directory,redirect
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the models
CROWD_MODEL = tf.saved_model.load('CC-Model/model')
MASK_MODEL = YOLO("Mask-Model/best.pt")
names = MASK_MODEL.model.names

# Create directories for uploaded and processed videos
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Initialize SQLite database
DB_NAME = 'VV.db'

def initialize_database():
    """Initialize the SQLite database and create the CrowdControl table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CrowdControl (
            Detection_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Camera_ID TEXT NOT NULL,
            Timestamp TEXT NOT NULL,
            No_of_Detections INTEGER NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Mask_Detection (
            Detection_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Camera_ID TEXT NOT NULL,
            Timestamp TEXT NOT NULL,
            No_of_Detections INTEGER NOT NULL,
            Image BLOB      
        )
    ''')
    conn.commit()
    conn.close()

def log_crowd_detection_to_db(camera_id, no_of_detections):
    """Insert a detection log into the CrowdControl table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO CrowdControl (Camera_ID, Timestamp, No_of_Detections)
        VALUES (?, ?, ?)
    ''', (camera_id, timestamp, no_of_detections))
    conn.commit()
    conn.close()

def log_mask_detection_to_db(camera_id, no_of_detections,image_data):
    """Log a detection event into the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO Mask_Detection (Camera_ID, Timestamp, No_of_Detections, Image)
        VALUES (?, ?, ?, ?)
    ''', (camera_id, timestamp, no_of_detections, image_data))
    print("Called")
    conn.commit()
    conn.close()

# Call this once to initialize the database
initialize_database()

#CC MODELS FUNCTIONS
def CC_process_video_alternative(video_path, model, output_path, threshold=0.25, frame_skip=10, detection_threshold=15):
    """Efficient frame-by-frame video processing, skipping frames periodically, with people detection."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use the 'mp4v' codec for MP4 files
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, Resolution: {frame_width}x{frame_height}, FPS: {fps}")

    frame_count = 0
    processed_frame_count = 0
    total_people_detected = 0
    camera_id = os.path.basename(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames based on frame_skip
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_count += 1
        processed_frame_count += 1

        # Convert BGR frame (OpenCV) to RGB for TensorFlow
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)[tf.newaxis, ...]

        # Detect objects
        results = model(input_tensor)

        # Draw bounding boxes for detections
        boxes = results['detection_boxes'].numpy()[0]
        classes = results['detection_classes'].numpy()[0]
        scores = results['detection_scores'].numpy()[0]

        frame_people_detected = 0

        # Count the number of people detected in this frame
        for i in range(int(results['num_detections'][0])):
            if classes[i] == 1 and scores[i] > threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                left, top, right, bottom = (int(xmin * frame_width), int(ymin * frame_height),
                                            int(xmax * frame_width), int(ymax * frame_height))
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                frame_people_detected += 1

        total_people_detected += frame_people_detected

        # If the detection threshold is exceeded, log to the database
        if frame_people_detected >= detection_threshold:
            log_crowd_detection_to_db(camera_id, frame_people_detected)

        # Write processed frame to output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video processing completed. Output saved at: {output_path}")
    print(f"Total people detected: {total_people_detected}")
    return total_people_detected


def CC_process_webcam_feed(model, threshold=0.25,detection_threshold=15):
    """Process live webcam feed for people detection."""
    cap = cv2.VideoCapture(0)  # Open webcam
    total_people_detected = 0
    frame_count = 0
    frame_skip = 10

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return -1

    print("Processing live webcam feed...")
    while cap.isOpened():
        total_people_detected_in_frame = 0
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to reduce computation
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_count += 1

        # Convert BGR frame (OpenCV) to RGB for TensorFlow
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)[tf.newaxis, ...]

        # Detect objects
        results = model(input_tensor)

        # Extract detections
        boxes = results['detection_boxes'].numpy()[0]
        classes = results['detection_classes'].numpy()[0]
        scores = results['detection_scores'].numpy()[0]

        # Count the number of people detected in this frame
        for i in range(int(results['num_detections'][0])):
            if classes[i] == 1 and scores[i] > threshold:
                total_people_detected += 1
                total_people_detected_in_frame +=1

        # Draw bounding boxes for detections
        for i in range(int(results['num_detections'][0])):
            if classes[i] == 1 and scores[i] > threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                left, top, right, bottom = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]),
                                            int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Show the processed frame
        cv2.imshow('Webcam Feed', frame)

        if total_people_detected_in_frame >= detection_threshold:
            log_crowd_detection_to_db("Webcam", total_people_detected_in_frame)        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Normalize total people detected by the number of processed frames
    normalized_count = total_people_detected / (frame_count / frame_skip)
    print(f"Total people detected from webcam: {normalized_count}")
    return normalized_count

#MASK MODELS FUNCTIONS
def MASK_detect_objects_from_webcam(model):
    count = 0
    cap = cv2.VideoCapture(0)  # 0 for the default webcam
    camera_id = "Laptop Webcam"  # Replace with your camera ID
    no_mask_detections = 0

    # Cache to store track IDs of recently detected "No Queue" persons
    recent_detections = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue

        # Resize the frame to (1020, 600)
        frame = cv2.resize(frame, (1020, 600))

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                label = names[class_id]
                x1, y1, x2, y2 = box

                # Draw bounding box and label on the frame
                color = (0, 255, 0) if label.lower() != "mask_weared_incorrect" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Check for "No Queue" class (Assume class 1 = "No Queue")
                if label.lower() == "mask_weared_incorrect" and track_id not in recent_detections:
                    recent_detections.add(track_id)  # Add to cache
                    no_mask_detections += 1

                    # Save the frame with bounding boxes as an image
                    _, image_buffer = cv2.imencode('.jpg', frame)
                    image_data = image_buffer.tobytes()

                    # Log detection to the database
                    log_mask_detection_to_db(camera_id, 1, image_data)

        # Optionally, clear recent detections after a certain period
        if len(recent_detections) > 100:  # Limit cache size
            recent_detections.clear()

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def MASK_process_video_for_detections(video_path,model):
    print("process_video_for_detections")
    """Process a video for mask detections and save snapshots to the database."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    camera_id = os.path.basename(video_path)  # Use the video filename as the camera ID

    # Cache to store track IDs of recently detected "No Queue" persons
    recent_detections = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue

        # Resize the frame to (1020, 600)
        frame = cv2.resize(frame, (1020, 600))

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                label = names[class_id]
                x1, y1, x2, y2 = box

                # Draw bounding box and label on the frame
                color = (0, 255, 0) if label.lower() != "mask_weared_incorrect" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Check for "No Queue" class (Assume class 1 = "No Queue")
                if label.lower() == "mask_weared_incorrect" and track_id not in recent_detections:
                    recent_detections.add(track_id)  # Add to cache

                    # Save the frame with bounding boxes as an image
                    _, image_buffer = cv2.imencode('.jpg', frame)
                    image_data = image_buffer.tobytes()

                    # Log detection to the database
                    log_mask_detection_to_db(camera_id, 1, image_data)

    cap.release()

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video uploads and processing for selected models."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    selected_models = request.form.getlist('models')  # Get selected models
    if not selected_models:
        return jsonify({'error': 'No model selected'}), 400

    if file:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)  # Save uploaded file

        responses = {}

        if 'crowd' in selected_models:
            output_path_crowd = os.path.join(app.config['OUTPUT_FOLDER'], 'crowd_' + file.filename)
            total_detections_crowd = CC_process_video_alternative(
                input_path, CROWD_MODEL, output_path_crowd, threshold=0.25, frame_skip=10)
            responses['crowd_count'] = {
                'total_detections': total_detections_crowd,
                'download_link': f'/download?filename={os.path.basename(output_path_crowd)}'
            }

        if 'mask' in selected_models:
            MASK_process_video_for_detections(input_path,MASK_MODEL)
            responses['mask_detection'] = 'Mask detection logged successfully.'

        return jsonify({'message': 'Processing completed', 'results': responses})

@app.route('/download')
def download_file():
    """Download the processed video."""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path, as_attachment=True)

@app.route('/webcam', methods=['GET'])
def webcam_feed():
    """Handle live webcam feed processing for selected models."""
    selected_models = request.args.getlist('models')  # Get selected models
    if not selected_models:
        return jsonify({'error': 'No model selected'}), 400

    responses = {}

    if 'crowd' in selected_models:
        total_detections_crowd = CC_process_webcam_feed(CROWD_MODEL, threshold=0.25)
        if total_detections_crowd == -1:
            return jsonify({'error': 'Unable to access the webcam for crowd detection'}), 500
        responses['crowd_count'] = {'total_detections': total_detections_crowd}

    if 'mask' in selected_models:
        return Response(MASK_detect_objects_from_webcam(MASK_MODEL), mimetype='multipart/x-mixed-replace; boundary=frame')

    return jsonify({'message': 'Webcam feed processed', 'results': responses})

if __name__ == '__main__':
    app.run(debug=True)