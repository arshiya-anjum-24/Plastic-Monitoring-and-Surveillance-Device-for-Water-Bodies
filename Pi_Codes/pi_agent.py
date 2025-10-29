# pi_agent.py (RUNS ON RASPBERRY PI - FINAL CORRECTED)

import cv2
import time
import serial
import pynmea2
import numpy as np
from flask import Flask, Response, request, jsonify
import pigpio
import sys

# --- Configuration ---
CAMERA_ID = 0
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
STREAM_JPEG_QUALITY = 70

# Motor GPIO Pins (BCM numbering)
MOTOR_L_FWD_PIN = 17
MOTOR_L_REV_PIN = 27
MOTOR_R_FWD_PIN = 22
MOTOR_R_REV_PIN = 23

# GPS Serial Port
GPS_SERIAL_PORT = '/dev/ttyS0'
GPS_BAUDRATE = 9600

# --- Flask App ---
app = Flask(__name__)

# --- Globals ---
camera = None
pi_gpio = None
ser_gps = None

# --- Connect to pigpio Daemon ---
try:
    pi_gpio = pigpio.pi()
    if not pi_gpio.connected:
        raise RuntimeError("pigpio daemon not connected")
    print("Agent: Connected to pigpio daemon.")

    # Initialize motor pins
    pi_gpio.set_mode(MOTOR_L_FWD_PIN, pigpio.OUTPUT)
    pi_gpio.set_mode(MOTOR_L_REV_PIN, pigpio.OUTPUT)
    pi_gpio.set_mode(MOTOR_R_FWD_PIN, pigpio.OUTPUT)
    pi_gpio.set_mode(MOTOR_R_REV_PIN, pigpio.OUTPUT)

    # Set PWM range and stop motors
    for pin in [MOTOR_L_FWD_PIN, MOTOR_L_REV_PIN, MOTOR_R_FWD_PIN, MOTOR_R_REV_PIN]:
        pi_gpio.set_PWM_range(pin, 255)
        pi_gpio.write(pin, 0)

    print("Agent: Motor GPIO pins initialized and stopped.")

except Exception as e:
    print(f"CRITICAL ERROR connecting to pigpio: {e}. GPIO disabled.")
    pi_gpio = None


# --- Camera Setup ---
def initialize_camera():
    global camera
    camera = cv2.VideoCapture(CAMERA_ID)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"Agent: Camera {CAMERA_ID} opened ({FRAME_WIDTH}x{FRAME_HEIGHT}).")
    else:
        print(f"CRITICAL ERROR: Cannot open camera {CAMERA_ID}.")
        camera = None


# --- GPS Setup ---
def initialize_gps():
    global ser_gps
    try:
        ser_gps = serial.Serial(GPS_SERIAL_PORT, baudrate=GPS_BAUDRATE, timeout=1)
        print(f"Agent: GPS Serial port {GPS_SERIAL_PORT} opened.")
    except Exception as e:
        print(f"WARNING: Could not open GPS port {GPS_SERIAL_PORT}: {e}")
        ser_gps = None


def get_latest_gps():
    """Reads GPS, returns {'lat': float, 'lon': float} or None."""
    global ser_gps
    if not ser_gps:
        return None
    try:
        start_time = time.time()
        lat, lon = None, None
        fix_found_level = 0

        while time.time() - start_time < 1.5:
            line = ser_gps.readline().decode('ascii', errors='replace').strip()
            if not line:
                continue

            if line.startswith('$GPRMC'):
                try:
                    msg = pynmea2.parse(line)
                    if (
                        msg.status == 'A'
                        and hasattr(msg, 'latitude')
                        and msg.latitude != 0.0
                        and hasattr(msg, 'longitude')
                        and msg.longitude != 0.0
                    ):
                        lat = msg.latitude
                        lon = msg.longitude
                        fix_found_level = 2
                except pynmea2.ParseError:
                    pass

            elif line.startswith('$GPGGA') and fix_found_level < 2:
                try:
                    msg = pynmea2.parse(line)
                    if (
                        msg.gps_qual > 0
                        and hasattr(msg, 'latitude')
                        and msg.latitude != 0.0
                        and hasattr(msg, 'longitude')
                        and msg.longitude != 0.0
                    ):
                        if fix_found_level < 1:
                            lat = msg.latitude
                            lon = msg.longitude
                            fix_found_level = 1
                except pynmea2.ParseError:
                    pass

        if fix_found_level > 0 and lat is not None and lon is not None:
            print(f"Agent: GPS Fix Sent: {lat:.5f}, {lon:.5f}")
            return {'lat': lat, 'lon': lon}
        else:
            return None

    except serial.SerialException as e:
        print(f"Agent: ERROR reading GPS (Serial): {e}")
        try:
            ser_gps.close()
        except:
            pass
        ser_gps = None
        print("Agent: GPS port closed due to error.")
        return None
    except Exception as e:
        print(f"Agent: ERROR reading GPS (Other): {e}")
        return None


# --- Motor Control ---
def set_pi_motor_speed(motor_pin_fwd, motor_pin_rev, speed):
    """Sets motor speed using pigpio PWM. Speed -255 to 255."""
    if not pi_gpio:
        return
    speed = int(max(-255, min(255, round(speed))))
    try:
        if speed > 0:
            pi_gpio.set_PWM_dutycycle(motor_pin_fwd, speed)
            pi_gpio.write(motor_pin_rev, 0)
        elif speed < 0:
            pi_gpio.write(motor_pin_fwd, 0)
            pi_gpio.set_PWM_dutycycle(motor_pin_rev, abs(speed))
        else:
            pi_gpio.set_PWM_dutycycle(motor_pin_fwd, 0)
            pi_gpio.set_PWM_dutycycle(motor_pin_rev, 0)
    except Exception as e:
        print(f"Agent: ERROR setting motor speed on pins {motor_pin_fwd}/{motor_pin_rev}: {e}")


# --- RAW Video Streaming Generator ---
def generate_raw_frames():
    """Yields unmodified raw camera frames as JPEG bytes."""
    if camera is None:
        print("Agent: Camera not available. Streaming error frame.")
        error_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(error_frame, "NO CAMERA", (30, FRAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes() if ret else b''
        while True:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')
            time.sleep(1.0)

    print("Agent: Starting RAW video frame generation...")
    while True:
        try:
            success, frame = camera.read()
            if not success:
                print("Agent: WARN: Failed to read raw frame from camera.")
                time.sleep(0.1)
                continue

            ret, buffer = cv2.imencode('.jpg', frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Agent: ERROR in raw frame generation loop: {e}")
            time.sleep(0.5)


# --- Flask Routes ---
@app.route('/video_feed')
def video_feed():
    return Response(generate_raw_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/gps')
def get_gps_data():
    coords = get_latest_gps()
    if coords:
        return jsonify(coords)
    else:
        return jsonify({"error": "No valid GPS fix available"}), 503


@app.route('/control', methods=['POST'])
def motor_control():
    if not pi_gpio:
        print("Agent: WARN: Motor command received but GPIO unavailable.")
        return jsonify({"error": "GPIO service not available on Pi"}), 503

    data = request.get_json()
    if not data:
        print("Agent: WARN: Invalid/Empty JSON received in /control POST.")
        return jsonify({"error": "Invalid JSON request body"}), 400

    left_speed = data.get('left', None)
    right_speed = data.get('right', None)
    print(f"Agent: Received motor command: Left={left_speed}, Right={right_speed}")

    if not isinstance(left_speed, (int, float)) or not isinstance(right_speed, (int, float)):
        print("Agent: WARN: Invalid speed types received.")
        return jsonify({"error": "Invalid or missing speed values"}), 400

    set_pi_motor_speed(MOTOR_L_FWD_PIN, MOTOR_L_REV_PIN, left_speed)
    set_pi_motor_speed(MOTOR_R_FWD_PIN, MOTOR_R_REV_PIN, right_speed)

    return jsonify({"status": "ok",
                    "left_set": int(round(left_speed)),
                    "right_set": int(round(right_speed))}), 200


@app.route('/health')
def health_check():
    status = {
        "pigpio_connected": pi_gpio is not None and pi_gpio.connected,
        "camera_opened": camera is not None and camera.isOpened(),
        "gps_port_opened": ser_gps is not None and ser_gps.is_open
    }
    return jsonify(status)


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Raspberry Pi Agent Initializing ---")
    initialize_camera()
    initialize_gps()

    print("--- Agent Status ---")
    if pi_gpio is None:
        print("CRITICAL: pigpio NOT available. Motor control disabled.")
    else:
        print(" pigpio Status: Connected.")
    if camera is None:
        print("CRITICAL: Camera NOT available. Video stream disabled.")
    else:
        print(" Camera Status: Opened.")
    if ser_gps is None:
        print("WARNING: GPS Serial NOT available. GPS readings disabled.")
    else:
        print(" GPS Status: Port Opened.")
    print("--------------------")

    if camera is None:
        print("FATAL ERROR: Cannot start server without camera.")
        sys.exit(1)

    print("Starting Flask server for Pi Agent on port 5001...")
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except Exception as e:
        print(f"FATAL ERROR running Flask server: {e}")
    finally:
        print("\nAgent shutting down...")
        try:
            if pi_gpio:
                print("Stopping motors...")
                set_pi_motor_speed(MOTOR_L_FWD_PIN, MOTOR_L_REV_PIN, 0)
                set_pi_motor_speed(MOTOR_R_FWD_PIN, MOTOR_R_REV_PIN, 0)
                time.sleep(0.1)
                pi_gpio.stop()
                print("pigpio connection stopped.")
        except Exception as e:
            print(f"Error during pigpio cleanup: {e}")

        try:
            if camera:
                camera.release()
                print("Camera released.")
        except Exception as e:
            print(f"Error releasing camera: {e}")

        try:
            if ser_gps:
                ser_gps.close()
                print("GPS serial port closed.")
        except Exception as e:
            print(f"Error closing GPS port: {e}")

        print("Agent shutdown complete.")
