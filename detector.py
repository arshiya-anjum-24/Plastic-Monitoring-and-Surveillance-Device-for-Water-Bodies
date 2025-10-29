# ECS/detector.py

import torch
import cv2
import time
import requests
import random
import numpy as np
from ultralytics import YOLO # CRITICAL: Imports the necessary class for the model

# --- Configuration ---
MODEL_PATH = 'model.pt' 
NODE_DETECTION_URL = 'http://localhost:3000/api/detection' 
CAMERA_ID = 0 

# 1. Load the .pt model here (FINAL, WORKING LOAD)
try:
    # Use the YOLO class to load the model. This handles the model architecture 
    # and weight loading automatically, bypassing state dictionary issues.
    model = YOLO(MODEL_PATH)
    
    # model.eval() is now managed internally by the YOLO object
    print("ML Model loaded successfully.")
except Exception as e:
    # This message is sent to the Node.js console via stderr/stdout
    print(f"ERROR: Failed to load model. Check 'model.pt' file: {e}")
    exit(1)

# 2. Function to generate a random GPS log for the hotspot (Simulation)
def generate_random_gps():
    # Simulating a location near the general project area
    lat = 16.507 + random.uniform(-0.005, 0.005)
    lon = 80.640 + random.uniform(-0.005, 0.005)
    return {'lat': round(lat, 5), 'lon': round(lon, 5)}

# 3. Main detection loop
def run_detection():
    # OpenCV accesses the laptop camera directly
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera ID {CAMERA_ID}. Please check laptop camera permissions.")
        exit(1)

    print("Starting ML detection loop (checking camera frames)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue
        
        # --- ML INFERENCE AND LOGIC ---
        
        # In a real scenario, you would run model.predict(frame) here to get results.
        # We are simulating a detection 8% of the time:
        is_plastic = random.random() < 0.08 
        confidence = random.uniform(0.7, 0.99) if is_plastic else 0.0
        
        # 4. LOG THE OUTPUT IF PLASTIC IS DETECTED
        if is_plastic and confidence > 0.6:
            gps_log = generate_random_gps()
            
            payload = {
                'lat': gps_log['lat'],
                'lon': gps_log['lon'],
                'confidence': round(confidence, 2)
            }
            
            # Post detection log to Node.js server
            try:
                requests.post(NODE_DETECTION_URL, json=payload, timeout=2)
                print(f"Detection Logged: {payload}") 
            except requests.exceptions.RequestException as e:
                print(f"ERROR: Failed to POST detection log: Is Node.js server running? {e}")
        
        time.sleep(0.5) 
        
    cap.release()

if __name__ == "__main__":
    run_detection()