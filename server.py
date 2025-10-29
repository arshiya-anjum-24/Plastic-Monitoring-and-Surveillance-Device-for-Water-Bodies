# server.py (Corrected & cleaned — keeps your original structure)
import eventlet  # must import/eventlet.monkey_patch early for socketio/eventlet
eventlet.monkey_patch()

# --- standard imports ---
import cv2
import time
import requests
from ultralytics import YOLO
from flask import Flask, Response, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np
from datetime import datetime
from collections import deque
from geopy.distance import geodesic
from pymongo import MongoClient, errors as mongo_errors
import threading
import queue
import math
import torch
import os
import sys

# --- debug device info ---
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print("GPU name:", torch.cuda.get_device_name(0))
    except Exception:
        pass

# --- Configuration ---
PI_IP_ADDRESS = "10.79.70.189"  # set by you
PI_PORT = 5001
PI_VIDEO_URL = f"http://{PI_IP_ADDRESS}:{PI_PORT}/video_feed"
PI_CONTROL_URL = f"http://{PI_IP_ADDRESS}:{PI_PORT}/control"
PI_GPS_URL = f"http://{PI_IP_ADDRESS}:{PI_PORT}/gps"

MODEL_PATH = 'ECS_FINAL/model.pt'
CONF_THRESHOLD = 0.5
CLASS_NAME_TO_DETECT = 'plastic'  # exact class name from your model
MIN_DISTANCE_METERS = 5
MIN_LOG_INTERVAL_SECONDS = 5
RECENT_LOG_HISTORY_SIZE = 15
STREAM_JPEG_QUALITY = 60

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "boatLogs"
COLLECTION_NAME = "detections"

# motor speeds etc (kept as you had)
FWD_L_SPEED = 255
FWD_R_SPEED = -51
BWD_L_SPEED = -128
BWD_R_SPEED = -128
LEFT_L_SPEED = 217
LEFT_R_SPEED = 179
RIGHT_L_SPEED = 179
RIGHT_R_SPEED = -179

TRANSITION_DURATION_MS = 300
TRANSITION_STEP_MS = 30

FRAME_HEIGHT = 480
FRAME_WIDTH = 640

TARGET_FPS = 15.0
MIN_FRAME_INTERVAL = 1.0 / TARGET_FPS

# --- Flask / SocketIO ---
app = Flask(__name__, static_folder='public', static_url_path='')
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Globals ---
model = None
db_client = None
db_collection = None
processed_frame_queue = queue.Queue(maxsize=2)
recent_log_coords = deque(maxlen=RECENT_LOG_HISTORY_SIZE)
coords_lock = threading.Lock()
boat_state = {'power': False}
stop_threads = threading.Event()
pi_connection_status = {"video": False, "control": False, "gps": False}
currentL_Speed = 0
currentR_Speed = 0
transition_timer = None

# load model
try:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = YOLO(MODEL_PATH)
    try:
        model.to(device)
    except Exception:
        # ultralytics may manage device internally; ignore if .to() not supported
        pass
    print(f"Laptop: ML Model loaded (device={device}).")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model '{MODEL_PATH}': {e}")
    model = None

# connect DB (optional)
try:
    db_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = db_client[DB_NAME]
    db_collection = db[COLLECTION_NAME]
    db_client.admin.command('ping')
    print(f"Laptop: MongoDB connected to '{DB_NAME}'.")
except mongo_errors.ConnectionFailure as e:
    print(f"CRITICAL ERROR: MongoDB connection failed: {e}")
    db_collection = None
except Exception as e:
    print(f"CRITICAL ERROR: Unexpected error connecting to MongoDB: {e}")
    db_collection = None

# helpers
def is_log_valid(new_coords):
    if new_coords is None:
        return False
    try:
        lat = float(new_coords.get('lat', 0.0))
        lon = float(new_coords.get('lon', 0.0))
        new_point = (lat, lon)
        now = time.time()
    except (ValueError, TypeError, AttributeError):
        return False

    with coords_lock:
        if not recent_log_coords:
            return True
        for old_lat, old_lon, old_time in recent_log_coords:
            try:
                distance = geodesic(new_point, (old_lat, old_lon)).meters
                time_diff = now - old_time
            except Exception:
                continue
            if distance < MIN_DISTANCE_METERS:
                # too close to previous hit
                print(f"Log ignored: Too close ({distance:.1f}m)")
                return False
            if time_diff < MIN_LOG_INTERVAL_SECONDS:
                print(f"Log ignored: Too soon ({time_diff:.1f}s)")
                return False
    return True

def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def send_motor_command(targetL, targetR):
    global currentL_Speed, currentR_Speed
    currentL_Speed = int(max(-255, min(255, round(targetL))))
    currentR_Speed = int(max(-255, min(255, round(targetR))))
    payload = {'left': currentL_Speed, 'right': currentR_Speed}
    try:
        # increased timeout so transient wifi doesn't break
        response = requests.post(PI_CONTROL_URL, json=payload, timeout=2.0)
        response.raise_for_status()
        pi_connection_status["control"] = True
    except requests.exceptions.RequestException as e:
        print(f"WARN: Failed send speeds L:{currentL_Speed}, R:{currentR_Speed} to Pi: {e}")
        pi_connection_status["control"] = False
    except Exception as e:
        print(f"ERROR: Sending speeds: {e}")
        pi_connection_status["control"] = False

def transition_to_speeds_pi(targetL, targetR):
    global transition_timer, currentL_Speed, currentR_Speed
    if transition_timer is not None:
        try:
            transition_timer.kill()
        except Exception:
            try:
                transition_timer.cancel()
            except Exception:
                pass
        transition_timer = None

    startL = currentL_Speed
    startR = currentR_Speed
    num_steps = max(1, int(TRANSITION_DURATION_MS / TRANSITION_STEP_MS))
    stepL = (targetL - startL) / num_steps
    stepR = (targetR - startR) / num_steps
    print(f"Laptop: Transitioning L: {startL}->{targetL}, R: {startR}->{targetR} over {num_steps} steps")

    def step_transition(current_step):
        global currentL_Speed, currentR_Speed, transition_timer
        if current_step > num_steps or stop_threads.is_set():
            send_motor_command(targetL, targetR)
            print(f"Laptop: Transition complete. Final L:{targetL}, R:{targetR}")
            transition_timer = None
            return
        intermediateL = startL + stepL * current_step
        intermediateR = startR + stepR * current_step
        send_motor_command(intermediateL, intermediateR)
        transition_timer = eventlet.spawn_after(TRANSITION_STEP_MS / 1000.0, step_transition, current_step + 1)

    transition_timer = eventlet.spawn(step_transition, 1)

# movement helpers unchanged
def moveStop():
    print("Laptop: Command STOP")
    transition_to_speeds_pi(0, 0)

def moveForward():
    print("Laptop: Command FORWARD")
    transition_to_speeds_pi(FWD_R_SPEED, FWD_L_SPEED)

def moveBackward():
    print("Laptop: Command BACKWARD")
    transition_to_speeds_pi(BWD_R_SPEED, BWD_L_SPEED)

def moveLeft():
    print("Laptop: Command LEFT")
    transition_to_speeds_pi(LEFT_R_SPEED, LEFT_L_SPEED)

def moveRight():
    print("Laptop: Command RIGHT")
    transition_to_speeds_pi(RIGHT_R_SPEED, RIGHT_L_SPEED)

# --- Video processing thread (core) ---
def video_processing_thread():
    global recent_log_coords, pi_connection_status, FRAME_HEIGHT, FRAME_WIDTH

    if model is None:
        print("AI Model not loaded. Video thread stopping.")
        return

    if db_collection is None:
        print("MongoDB not connected. DB logging disabled.")

    print(f"Video processing thread started (device={device})...")

    retry_delay = 3
    max_retries = 5
    retries = 0
    frame_counter = 0

    # keep track to avoid too-frequent GPS calls
    while not stop_threads.is_set():
        stream = None
        try:
            stream = requests.get(PI_VIDEO_URL, stream=True, timeout=8)
            stream.raise_for_status()
            print("Laptop: Connected to Pi video stream.")
            pi_connection_status["video"] = True
            retries = 0
            bytes_buffer = bytes()

            for chunk in stream.iter_content(chunk_size=4096):
                if stop_threads.is_set():
                    break
                if not chunk:
                    continue
                bytes_buffer += chunk
                a = bytes_buffer.find(b'\xff\xd8')
                b = bytes_buffer.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_buffer[a:b + 2]
                    bytes_buffer = bytes_buffer[b + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    frame_counter += 1
                    annotated_frame = frame.copy()
                    log_attempted = False
                    plastic_detected = False

                    # process every alternate frame as before
                    if frame_counter % 2 != 0:
                        if model is not None and boat_state.get('power', False):
                            try:
                                # use_half detection (only when CUDA available)
                                use_half = (torch.cuda.is_available())
                                results = model.predict(
                                    frame,
                                    device=device,
                                    imgsz=320,
                                    half=use_half,
                                    verbose=False,
                                    stream=False
                                )

                                if results and len(results) > 0:
                                    r = results[0]
                                    try:
                                        annotated_frame = r.plot()
                                    except Exception:
                                        # fallback if .plot not available
                                        annotated_frame = frame.copy()

                                    for box in r.boxes:
                                        if (box.conf is not None and len(box.conf) > 0):
                                            conf = float(box.conf[0])
                                            cls_id = int(box.cls[0])
                                            cls_name = r.names[cls_id]
                                            # match plastic
                                            if (conf > CONF_THRESHOLD and cls_name == CLASS_NAME_TO_DETECT and not log_attempted):
                                                log_attempted = True
                                                plastic_detected = True

                                                # fetch GPS from pi - defensive network handling
                                                gps_log = None
                                                try:
                                                    gps_res = requests.get(PI_GPS_URL, timeout=2.0)
                                                    if gps_res.status_code == 200:
                                                        gps_data = gps_res.json()
                                                        if "error" not in gps_data:
                                                            gps_log = gps_data
                                                            pi_connection_status["gps"] = True
                                                        else:
                                                            pi_connection_status["gps"] = False
                                                    else:
                                                        pi_connection_status["gps"] = False
                                                except requests.exceptions.RequestException:
                                                    pi_connection_status["gps"] = False

                                                # if GPS available and passes distance/time filters, log + emit
                                                if is_log_valid(gps_log):
                                                    payload = {
                                                        'lat': float(gps_log['lat']),
                                                        'lon': float(gps_log['lon']),
                                                        'confidence': round(conf, 2)
                                                    }
                                                    log_time = time.time()
                                                    log_dt = datetime.now()
                                                    payload_db = payload.copy()
                                                    payload_db['timestamp'] = log_dt
                                                    with coords_lock:
                                                        recent_log_coords.append((payload['lat'], payload['lon'], log_time))
                                                    # insert to DB if configured
                                                    if db_collection is not None:
                                                        try:
                                                            db_collection.insert_one(payload_db)
                                                        except Exception as e_db:
                                                            print(f"DB Insert Error: {e_db}")
                                                    # prepare socket payload (numbers + ISO time)
                                                    try:
                                                        payload_socket = {
                                                            'lat': float(payload['lat']),
                                                            'lon': float(payload['lon']),
                                                            'confidence': float(payload['confidence']),
                                                            'timestamp': log_dt.isoformat()
                                                        }
                                                        # broadcast to all clients
                                                        socketio.emit('new_detection', payload_socket, broadcast=True)
                                                    except Exception as e_sock:
                                                        print(f"Socket Emit Error: {e_sock}")

                                                    print(f"Laptop: Logged VALID detection: {payload}")
                                                    # break to avoid multiple logs for same frame
                                                    break
                                                else:
                                                    # GPS was missing or rejected by filter — emit a lightweight info so front-end can display
                                                    try:
                                                        payload_socket = {
                                                            'gps_available': False,
                                                            'confidence': round(conf, 2),
                                                            'timestamp': datetime.now().isoformat()
                                                        }
                                                        socketio.emit('new_detection', payload_socket, broadcast=True)
                                                    except Exception as e_sock:
                                                        print(f"Socket Emit Error (no-gps): {e_sock}")
                                                    print("Laptop: Plastic detected but GPS invalid or unavailable.")
                                                    break

                            except Exception as e_pred:
                                print(f"AI Predict/Processing Error: {e_pred}")
                                annotated_frame = frame.copy()

                    # put frame in queue for processed feed
                    try:
                        if processed_frame_queue.full():
                            processed_frame_queue.get_nowait()
                        processed_frame_queue.put_nowait(annotated_frame)
                    except queue.Full:
                        pass

            # we've left the for-loop; close stream_response if it exists
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

            if stop_threads.is_set():
                break

        except requests.exceptions.RequestException as e:
            # connection to PI failed or timed out
            print(f"Pi video stream connection error: {e}")
            pi_connection_status["video"] = False
            retries += 1
            if retries > max_retries:
                print("Max retries reached. Video thread stopping.")
                stop_threads.set()
                break
            print(f"Retrying video connection in {retry_delay}s...")
            stop_threads.wait(retry_delay)

        except Exception as e:
            print(f"Unexpected video loop error: {e}")
            stop_threads.wait(retry_delay)

    # clean up on exit
    print("Video processing thread finished.")
    error_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(error_frame, "PI DISCONNECTED", (20, max(20, FRAME_HEIGHT // 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    try:
        processed_frame_queue.put_nowait(error_frame)
    except queue.Full:
        pass

# processed frame generator for webpage
def generate_processed_frames():
    print("Processed frame generator started.")
    while not stop_threads.is_set():
        try:
            frame = processed_frame_queue.get(timeout=1.0)
            if frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            processed_frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Web frame gen error: {e}")
            time.sleep(0.1)
    print("Processed frame generator stopped.")

# --- HTTP routes ---
@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('public', filename)

@app.route('/processed_video_feed')
def processed_video_feed():
    return Response(generate_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/raw_video_feed')
def raw_video_feed():
    """Relay the PI raw feed to the browser; keep the generator local to the request"""
    print("Laptop: Client requested raw video feed.")
    try:
        stream_response = requests.get(PI_VIDEO_URL, stream=True, timeout=5)
        stream_response.raise_for_status()
        print("Laptop: Connected to Pi raw stream for relay.")
        def generate_pi_stream():
            try:
                for chunk in stream_response.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
            except Exception as e:
                print(f"Error relaying raw stream chunk: {e}")
            finally:
                try:
                    stream_response.close()
                except Exception:
                    pass
                print("Laptop: Pi raw stream relay finished.")
        return Response(generate_pi_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except requests.exceptions.RequestException as e:
        print(f"ERROR connecting to Pi raw video feed ({PI_VIDEO_URL}): {e}")
        return f"Error connecting to Pi feed: {e}", 503
    except Exception as e:
        print(f"Unexpected error in raw_video_feed route: {e}")
        return f"Server error: {e}", 500

@app.route('/api/state')
def get_state():
    return jsonify(boat_state)

@app.route('/control/<cmd>')
def control_command_web(cmd):
    global boat_state, recent_log_coords
    print(f"Laptop received web command: {cmd}")
    response_message = "Unknown command"
    status_code = 400
    if cmd == "start":
        if not boat_state['power']:
            if db_collection is not None:
                try:
                    db_collection.delete_many({})
                    print("Laptop: Database cleared.")
                except Exception as e_db:
                    print(f"DB Clear Error: {e_db}")
            with coords_lock:
                recent_log_coords.clear()
        boat_state['power'] = True
        response_message = "Boat activated"
        status_code = 200
    elif cmd == "stop":
        boat_state['power'] = False
        moveStop()
        response_message = "Boat stopping"
        status_code = 200
    elif cmd in ["forward", "backward", "left", "right"]:
        if not boat_state['power']:
            response_message = "Start boat first"
            status_code = 400
        else:
            if cmd == "forward":
                moveForward()
            elif cmd == "backward":
                moveBackward()
            elif cmd == "left":
                moveLeft()
            elif cmd == "right":
                moveRight()
            response_message = f"Boat transitioning to {cmd}"
            status_code = 200
    else:
        status_code = 400
    return jsonify({"message": response_message}), status_code

@app.route('/logs')
def get_logs():
    if db_collection is None:
        return jsonify({"error": "DB not connected"}), 500
    try:
        logs_cursor = db_collection.find().sort("timestamp", -1)
        logs_list = []
        for log in logs_cursor:
            log['_id'] = str(log['_id'])
            try:
                log['timestamp'] = log['timestamp'].isoformat()
            except Exception:
                pass
            logs_list.append(log)
        return jsonify(logs_list)
    except Exception as e_db:
        print(f"Log fetch Error: {e_db}")
        return jsonify({"error": "DB fetch failed"}), 500

# --- socket events ---
@socketio.on('connect')
def handle_connect():
    print('Laptop: Website client connected via SocketIO')

@socketio.on('disconnect')
def handle_disconnect():
    print('Laptop: Website client disconnected')

# --- main ---
if __name__ == '__main__':
    print("--- Laptop Server Initializing ---")
    if model is None:
        print("CRITICAL: Model NOT loaded.")
    if db_collection is None:
        print("CRITICAL: MongoDB NOT connected.")
    print("----------------------------------")
    print("Starting video processing thread...")
    video_thread = threading.Thread(target=video_processing_thread, daemon=True)
    video_thread.start()
    print(f"Starting Flask-SocketIO server. Website at http://localhost:3000")
    try:
        socketio.run(app, host='0.0.0.0', port=3000, debug=False)
    except KeyboardInterrupt:
        print("Ctrl+C received...")
    finally:
        print("Server shutting down...")
        stop_threads.set()
        print("Waiting for video thread...")
        video_thread.join(timeout=3.0)
        if not video_thread.is_alive():
            print("Video thread finished.")
        else:
            print("WARN: Video thread did not finish cleanly.")
        if db_client:
            db_client.close()
            print("MongoDB connection closed.")
        print("Shutdown complete.")
