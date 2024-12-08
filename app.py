import os
import cv2
import sqlite3
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_file, Response,url_for,send_from_directory,redirect
from datetime import datetime
from ultralytics import YOLO
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load the models
CROWD_MODEL = tf.saved_model.load('CC-Model/model')
MASK_MODEL = YOLO("Mask-Model/best.pt")
QUEUE_MODEL = YOLO("Queue-Model/best.pt")
SMOKE_MODEL = YOLO("Smoke-Model/best.pt")
Mask_names = MASK_MODEL.model.names
Queue_names = QUEUE_MODEL.model.names
Smoke_name = SMOKE_MODEL.model.names

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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Queue_Detection (
            Detection_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Camera_ID TEXT NOT NULL,
            Timestamp TEXT NOT NULL,
            No_of_Detections INTEGER NOT NULL,
            Image BLOB      
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Smoking_Detection (
            Detection_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Camera_ID TEXT NOT NULL,
            Timestamp TEXT NOT NULL,
            No_of_Detections INTEGER NOT NULL,
            Image BLOB      
        )
    ''')  
    # New Camera Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Camera (
            Camera_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            IP_Address TEXT NOT NULL,
            Location_ID INTEGER NOT NULL,
            FOREIGN KEY (Location_ID) REFERENCES Location(Location_ID)
        )
    ''')

    # New Location Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Location (
            Location_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Location_Name TEXT NOT NULL UNIQUE
        )
    ''')

    # New Alerts Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Alerts (
            Alert_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Camera_ID INTEGER NOT NULL,
            Location_Name TEXT NOT NULL,
            Alert_Type TEXT NOT NULL,
            Detected_Value INTEGER NOT NULL,
            Timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()



def log_alert(camera_id, location_name, alert_type, detected_value):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO Alerts (Camera_ID, Location_Name, Alert_Type, Detected_Value, Timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (camera_id, location_name, alert_type, detected_value, timestamp))
    conn.commit()
    conn.close()

#LOGGING DATA IN DB
def log_crowd_detection_to_db(camera_id, no_of_detections):
    """Insert a detection log into the CrowdControl table."""
    print("CC LOGGED CALLED")
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

def log_queue_detection_to_db(camera_id, no_of_detections,image_data):
    """Log a detection event into the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO Queue_Detection (Camera_ID, Timestamp, No_of_Detections, Image)
        VALUES (?, ?, ?, ?)
    ''', (camera_id, timestamp, no_of_detections, image_data))
    print("Called")
    conn.commit()
    conn.close()

def log_smoke_detection_to_db(camera_id, no_of_detections,image_data):
    """Log a detection event into the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO Smoking_Detection (Camera_ID, Timestamp, No_of_Detections, Image)
        VALUES (?, ?, ?, ?)
    ''', (camera_id, timestamp, no_of_detections, image_data))
    print("Called")
    conn.commit()
    conn.close()

# Call this once to initialize the database
initialize_database()

#CC MODELS FUNCTIONS
def CC_process_video_alternative(video_path, model, output_path, threshold=0.25, frame_skip=10, detection_threshold=5):
    """Efficient frame-by-frame video processing, skipping frames periodically, with people detection."""
    print("CC UPLOAD CALLED")
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
            location_name = "MainHall"  # Fetch location dynamically if needed
            log_alert(camera_id, location_name, "Crowd", frame_people_detected)


        # Write processed frame to output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video processing completed. Output saved at: {output_path}")
    print(f"Total people detected: {total_people_detected}")
    return total_people_detected

frame_count = 0  # Global counter for frame skipping

def CC_process_webcam_feed(frame, model, threshold=0.25, detection_threshold=0, frame_skip=10):
    """Process a single frame for people detection."""
    print("Processing frame for crowd detection...")
    
    # Initialize counters
    total_people_detected_in_frame = 0
    
    # Skip frames to reduce computation
    global frame_count
    if frame_count % frame_skip != 0:
        frame_count += 1
        return "crowd", 0  # Skip processing and return zero detections

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
            total_people_detected_in_frame += 1

    print(f"Total people detected in current frame: {total_people_detected_in_frame}")

    # Draw bounding boxes for detections
    for i in range(int(results['num_detections'][0])):
        if classes[i] == 1 and scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left, top, right, bottom = (
                int(xmin * frame.shape[1]),
                int(ymin * frame.shape[0]),
                int(xmax * frame.shape[1]),
                int(ymax * frame.shape[0]),
            )
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Display the processed frame
    #cv2.imshow('Webcam Feed', frame)

    # Log detections if threshold is met
    if total_people_detected_in_frame >= detection_threshold:
        log_crowd_detection_to_db("Webcam", total_people_detected_in_frame)
        location_name = "Webcam Location"  # Replace with dynamic location
        log_alert("Webcam", location_name, "Crowd", total_people_detected_in_frame)


    return "crowd", total_people_detected_in_frame


def MASK_detect_objects_from_webcam(frame, model):
    """Process a single frame for mask-wearing detection without duplicate logging."""
    no_mask_detections = 0
    frame = cv2.resize(frame, (1020, 600))  # Resize the frame

    results = model.track(frame, persist=True)
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            label = Mask_names[class_id]
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if label.lower() not in ["without_mask", "mask_weared_incorrect"] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Check and log detection
            if label.lower() in ["without_mask", "mask_weared_incorrect"]:
                if track_id not in recent_detections_cache['mask']:
                    recent_detections_cache['mask'].add(track_id)
                    no_mask_detections += 1

                    _, image_buffer = cv2.imencode('.jpg', frame)
                    image_data = image_buffer.tobytes()
                    log_mask_detection_to_db("Webcam", 1, image_data)
                    location_name = "MainHall"  # Replace dynamically
                    log_alert("Webcam", location_name, "Mask", 1)

    return no_mask_detections


def MASK_process_video_for_detections(video_path,model):
    print("MASK UPLOAD CALLED")
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
                label = Mask_names[class_id]
                x1, y1, x2, y2 = box

                # Draw bounding box and label on the frame
                color = (0, 255, 0) if label.lower() not in ["without_mask", "mask_weared_incorrect"] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Check for "No Queue" class (Assume class 1 = "No Queue")
                if label.lower() in ["without_mask", "mask_weared_incorrect"] and track_id not in recent_detections:
                    recent_detections.add(track_id)  # Add to cache

                    # Save the frame with bounding boxes as an image
                    _, image_buffer = cv2.imencode('.jpg', frame)
                    image_data = image_buffer.tobytes()

                    # Log detection to the database
                    log_mask_detection_to_db(camera_id, 1, image_data)
                    location_name = "MainHall"  # Fetch dynamically if needed
                    log_alert(camera_id, location_name, "Mask", 1)


    cap.release()


def QUEUE_detect_objects_from_webcam(frame, model):
    """Process a single frame for queue detection without duplicate logging."""
    no_queue_detections = 0
    frame = cv2.resize(frame, (1020, 600))

    results = model.track(frame, persist=True)
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            label = Queue_names[class_id]
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if label.lower() != "no-queue" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Check and log detection
            if label.lower() == "no-queue":
                if track_id not in recent_detections_cache['queue']:
                    recent_detections_cache['queue'].add(track_id)
                    no_queue_detections += 1

                    _, image_buffer = cv2.imencode('.jpg', frame)
                    image_data = image_buffer.tobytes()
                    log_queue_detection_to_db("Webcam", 1, image_data)
                    location_name = "MainHall"  # Replace dynamically
                    log_alert("Webcam", location_name, "Queue", 1)

    return no_queue_detections


def QUEUE_process_video_for_detections(video_path,model):
    """Process a video for mask detections and save snapshots to the database."""
    print("FUNC1 Called")
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
                label = Queue_names[class_id]
                x1, y1, x2, y2 = box

                # Draw bounding box and label on the frame
                color = (0, 255, 0) if label.lower() != "no-queue" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Check for "No Queue" class (Assume class 1 = "No Queue")
                if label.lower() == "no-queue" and track_id not in recent_detections:
                    recent_detections.add(track_id)  # Add to cache

                    # Save the frame with bounding boxes as an image
                    _, image_buffer = cv2.imencode('.jpg', frame)
                    image_data = image_buffer.tobytes()

                    # Log detection to the database
                    log_queue_detection_to_db(camera_id, 1, image_data)
                    location_name = "MainHall"  # Fetch dynamically
                    log_alert(camera_id, location_name, "Queue", 1)


    cap.release()

def SMOKE_detect_objects_from_webcam(frame, model):
    """Process a single frame for smoke detection without duplicate logging."""
    smoke_detections = 0
    frame = cv2.resize(frame, (1020, 600))  # Resize the frame

    # Run YOLO model tracking on the frame
    results = model.track(frame, persist=True)
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            label = Smoke_name[class_id]
            x1, y1, x2, y2 = box

            # Set color based on detection type
            color = (0, 255, 0) if label.lower() not in ["cigarette", "smoke"] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Check and log detection
            if label.lower() in ["cigarette", "smoke"]:
                if track_id not in recent_detections_cache['smoke']:
                    recent_detections_cache['smoke'].add(track_id)
                    smoke_detections += 1

                    # Save the frame as an image
                    _, image_buffer = cv2.imencode('.jpg', frame)
                    image_data = image_buffer.tobytes()

                    # Log the detection into the database
                    log_smoke_detection_to_db("Webcam", 1, image_data)
                    location_name = "MainHall"  # Replace dynamically
                    log_alert("Webcam", location_name, "Smoke", 1)

    return smoke_detections


def SMOKE_process_video_for_detections(video_path,model):
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
                label = Smoke_name[class_id]
                x1, y1, x2, y2 = box

                # Draw bounding box and label on the frame
                color = (0, 255, 0) if label.lower() not in ["cigarette", "smoke"] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Check for "No Queue" class (Assume class 1 = "No Queue")
                if label.lower() in ["cigarette", "smoke"] and track_id not in recent_detections:
                    recent_detections.add(track_id)  # Add to cache

                    # Save the frame with bounding boxes as an image
                    _, image_buffer = cv2.imencode('.jpg', frame)
                    image_data = image_buffer.tobytes()

                    # Log detection to the database
                    log_smoke_detection_to_db(camera_id, 1, image_data)
                    location_name = "MainHall"  # Fetch dynamically
                    log_alert(camera_id, location_name, "Smoke", 1)

    cap.release()

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video uploads and process selected models concurrently."""
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

        # Define tasks for the selected models
        tasks = []
        responses = {}
        with ThreadPoolExecutor() as executor:
            if 'crowd' in selected_models:
                output_path_crowd = os.path.join(app.config['OUTPUT_FOLDER'], 'crowd_' + file.filename)
                tasks.append(
                    executor.submit(
                        CC_process_video_alternative, input_path, CROWD_MODEL, output_path_crowd, 0.25, 10
                    )
                )

            if 'mask' in selected_models:
                tasks.append(
                    executor.submit(MASK_process_video_for_detections, input_path, MASK_MODEL)
                )

            if 'queue' in selected_models:
                tasks.append(
                    executor.submit(QUEUE_process_video_for_detections, input_path, QUEUE_MODEL)
                )

            if 'smoke' in selected_models:
                tasks.append(
                    executor.submit(SMOKE_process_video_for_detections, input_path, SMOKE_MODEL)
                )

            # Collect results as tasks complete
            for future in as_completed(tasks):
                try:
                    result = future.result()  
                    if isinstance(result, tuple):  # For models with outputs (like crowd count)
                        model_type, output = result
                        responses[model_type] = str(output)  # Ensure values are JSON serializable
                    else:
                        # For other models like mask, queue, smoke
                        responses[str(result)] = f"{result} detection logged successfully."
                except Exception as e:
                    responses['error'] = f"Error during processing: {str(e)}"

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

@app.route('/webcam', methods=['GET','POST'])
def webcam_feed():
    """Handle live webcam feed and process selected models concurrently."""
    if request.method == 'POST':
   # Read JSON data from POST request
        selected_models = request.json.get('modelType', [])
    else:  # GET method
        selected_models = request.args.getlist('models')

    if not selected_models:
        return jsonify({'error': 'No model selected'}), 400

    responses = {}
    
    try:
        # Initialize the webcam
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            return jsonify({'error': 'Unable to access webcam'}), 500

        def process_frame(frame):
            """Process a single frame with all selected models concurrently."""
            tasks = []
            results = {}
            
            with ThreadPoolExecutor() as executor:
                if 'crowd' in selected_models:
                    tasks.append(
                        executor.submit(
                            CC_process_webcam_feed, frame, CROWD_MODEL, 0.25, 0, 10
                        )
                    )

                if 'mask' in selected_models:
                    tasks.append(
                        executor.submit(MASK_detect_objects_from_webcam, frame, MASK_MODEL)
                    )

                if 'queue' in selected_models:
                    tasks.append(
                        executor.submit(QUEUE_detect_objects_from_webcam, frame, QUEUE_MODEL)
                    )

                if 'smoke' in selected_models:
                    tasks.append(
                        executor.submit(SMOKE_detect_objects_from_webcam, frame, SMOKE_MODEL)
                    )

                # Collect results as tasks complete
                for future in as_completed(tasks):
                    try:
                        result = future.result()
                        if isinstance(result, tuple):  # For models with outputs (e.g., crowd count)
                            model_type, output = result
                            results[model_type] = str(output)
                        else:
                            results[str(result)] = f"{result} detection logged successfully."
                    except Exception as e:
                        results['error'] = f"Error during processing: {str(e)}"

            return results

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Process the current frame
            frame_results = process_frame(frame)
            responses.update(frame_results)

            # Optionally, display the frame
            cv2.imshow('Live Feed', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        return jsonify({'error': f'Error during live processing: {str(e)}'}), 500

    finally:
        # Release the webcam and close any OpenCV windows
        if 'video_capture' in locals():
            video_capture.release()
        cv2.destroyAllWindows()

    return jsonify({'message': 'Live processing ended', 'results': responses})

import base64
import sqlite3
from flask import jsonify

@app.route('/alerts', methods=['GET'])
def fetch_alerts():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Fetch all alerts from the Alerts table
        cursor.execute('''
            SELECT Camera_ID, Location_Name, Alert_Type, Detected_Value, Timestamp
            FROM Alerts
            ORDER BY Timestamp DESC
        ''')
        alerts = cursor.fetchall()

        alerts_with_images = []

        for alert in alerts:
            alert_dict = {
                'camera_id': alert[0],
                'location_name': alert[1],
                'alert_type': alert[2],
                'detected_value': alert[3],
                'timestamp': alert[4],
                'image': None  # Default value for image
            }

            # Fetch image for "Mask" alerts from Mask_Detection table
            if alert[2] == "Mask":
                cursor.execute('''
                    SELECT Image FROM Mask_Detection
                    WHERE Camera_ID = ? ORDER BY Timestamp DESC LIMIT 1
                ''', (alert[0],))
                mask_image = cursor.fetchone()
                if mask_image and mask_image[0]:
                    alert_dict['image'] = base64.b64encode(mask_image[0]).decode('utf-8')

            # Fetch image for "Smoke" alerts from Smoking_Detection table
            elif alert[2] == "Smoke":
                cursor.execute('''
                    SELECT Image FROM Smoking_Detection
                    WHERE Camera_ID = ? ORDER BY Timestamp DESC LIMIT 1
                ''', (alert[0],))
                smoke_image = cursor.fetchone()
                if smoke_image and smoke_image[0]:
                    alert_dict['image'] = base64.b64encode(smoke_image[0]).decode('utf-8')

            alerts_with_images.append(alert_dict)

        conn.close()
        return jsonify(alerts_with_images)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)