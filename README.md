 Overview

This project presents a Raspberry Pi–based Automatic Surface Vehicle (ASV) capable of detecting and monitoring plastic waste in water bodies using a machine learning model. The ASV captures live video feed, processes it in real-time to identify plastic objects, and transmits both live and processed feeds to a web-based control interface.

The system aims to support environmental monitoring and waste management initiatives by providing data-driven insights into the spread of plastic pollution.

 Key Features

Autonomous surface vehicle controlled via a web interface.

Live camera feed streamed to the website using Raspberry Pi.

Real-time plastic detection using a trained ML model (CNN).

GPS tracking of the vehicle with live location logs.

Heatmap visualization of plastic concentration levels.

Dashboard interface showing live and processed video feeds simultaneously.

 Technology Stack
 Hardware

Raspberry Pi 4

Pi Camera Module

GPS Module

Motor Driver (L298N)

DC Motors

Battery and Propeller System

 Software & Frameworks

Programming: Python, HTML, CSS, JavaScript

Backend: Flask (for communication between Pi and website)

ML/AI: TensorFlow, Keras, OpenCV

Frontend: Bootstrap, Chart.js (for heatmap visualization)

 *Working Principle*

The ASV moves across the water surface, capturing live video using the Pi camera.

The video feed is processed by the ML model running on the Raspberry Pi to detect plastic waste in real-time.

The results are streamed to the web dashboard, where users can view:

The live feed from the camera

The processed feed with detection overlays

The GPS coordinates of the ASV

A heatmap showing areas with higher detection frequency

All data is logged for further analysis and visualization.

🌍 Impact

This project contributes toward clean water initiatives by offering an affordable and scalable way to:

Detect and map plastic pollution zones in lakes, ponds, and rivers.

Support real-time environmental monitoring.

Encourage IoT + AI–based sustainability solutions.

 Team Members

This project was developed collaboratively as part of our academic coursework.

Arshiya Anjum Shaik – @arshiya-anjum-24

Mouli - @mouli

Member 3 @github-username

Member 4 @github-username

Member 5 @github-username

Member 6 @github-username

(Replace the placeholders above with your teammates’ actual GitHub usernames to tag them properly.)

 Results

Achieved 98% accuracy on test dataset for plastic classification.

Successfully demonstrated real-time detection and heatmap generation during prototype testing.

 How to Run

Clone this repository

git clone https://github.com/arshiya-anjum-24/Plastic-Monitoring-and-Surveillance-Device-for-Water-Bodies.git
cd Plastic-Monitoring-and-Surveillance-Device-for-Water-Bodies


Install dependencies

pip install -r requirements.txt


Run the Flask app

python app.py


Access the dashboard in your browser at http://<raspberry-pi-ip>:5000


 License

This project is for educational and research purposes only.
