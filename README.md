# Automatic Surface Vehicle for Plastic Detection in Water Bodies  

### Overview  
This project presents a **Raspberry Pi–based Automatic Surface Vehicle (ASV)** capable of **detecting and monitoring plastic waste in water bodies** using a **machine learning model**.  
The ASV captures live video feed, processes it in real time to identify plastic objects, and transmits both **live and processed feeds** to a web-based control interface.  

The system aims to support **environmental monitoring** and **waste management initiatives** by providing data-driven insights into the spread of plastic pollution.  

---

### Key Features  
- **Autonomous surface vehicle** controlled via a web interface  
- **Live camera feed** streamed using Raspberry Pi  
- **Real-time plastic detection** using a CNN-based ML model  
- **GPS tracking** and live location logging  
- **Heatmap visualization** of pollution density  
- **Dashboard interface** displaying live and processed feeds  

---

### Technology Stack  

#### Hardware  
- Raspberry Pi 4  
- Pi Camera Module  
- GPS Module  
- Motor Driver (L298N)  
- DC Motors  
- Battery and Propeller System  

#### Software & Frameworks  
- **Programming:** Python, HTML, CSS, JavaScript  
- **Backend:** Flask  
- **ML/AI:** TensorFlow, Keras, OpenCV  
- **Frontend:** Bootstrap, Chart.js (for heatmap visualization)  

---

### Working Principle  
1. The ASV navigates on the water surface and captures live video using the Pi camera.  
2. The video feed is processed by the ML model running on the Raspberry Pi to **detect plastic waste** in real time.  
3. The results are displayed on the **web dashboard**, where users can view:  
   - The **live feed** from the camera  
   - The **processed feed** with detection overlays  
   - The **GPS coordinates** of the ASV  
   - A **heatmap** showing areas with higher detection frequency  
4. All data is logged for analysis and environmental mapping.  

---

### Impact  
This project contributes to **sustainable water management** by:  
- Detecting and mapping **plastic pollution zones**  
- Enabling **real-time environmental monitoring**  
- Demonstrating an **AI + IoT–based solution** for aquatic waste detection  

---

### Team Members  
This project was developed collaboratively as part of our academic coursework.  

- **Arshiya Anjum Shaik** – [@arshiya-anjum-24](https://github.com/arshiya-anjum-24)  
- **Mouli** – [@velagamouli18](https://github.com/)  
- **Chaitanya** – [@github-username](https://github.com/)  
- **Sameera** – [@github-username](https://github.com/)  
- **Adithi** – [@github-username](https://github.com/)  
- **Pranith** – [@github-username](https://github.com/)  

---

### Results  
- Achieved **98% classification accuracy** on the plastic detection dataset.  
- Successfully demonstrated **live detection, GPS logging, and heatmap visualization** using the prototype.  
- Real-time monitoring through a **web interface** showing both live and processed feeds.  

---
