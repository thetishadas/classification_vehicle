#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from time import sleep


# In[10]:


import cv2
import numpy as np
from time import sleep

# Load YOLO
net = cv2.dnn.readNet("C:/Users/hp/Downloads/yolov4.weights", "C:/Users/hp/Downloads/yolov4.cfg")  
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open("C:/Users/hp/Downloads/coco.names", "r") as f:  
    classes = [line.strip() for line in f.readlines()]

# Settings
pos_line_up = 500  
pos_line_down = 550  
delay = 60  # Video FPS

# Vehicle counters
upward_cars = 0
upward_buses = 0
upward_trucks = 0
upward_bikes = 0

downward_cars = 0
downward_buses = 0
downward_trucks = 0
downward_bikes = 0

# Initialize video capture
cap = cv2.VideoCapture("C:/Users/hp/Downloads/video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tempo = float(1 / delay)
    sleep(tempo)

    height, width, channels = frame.shape

    # YOLO preprocessing
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Analyze detections
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if indexes is not empty before iterating
    if len(indexes) > 0:
        # Draw bounding boxes and count vehicles
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            # Draw the rectangle for the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
            # Display the label of the vehicle on the bounding box
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            center_y = y + h // 2

            # Detect upward movement
            if pos_line_up - 5 < center_y < pos_line_up + 5:
                if label == "car":
                    upward_cars += 1
                elif label == "bus":
                    upward_buses += 1
                elif label == "truck":
                    upward_trucks += 1
                elif label in ["bicycle", "motorbike"]:
                    upward_bikes += 1

            # Detect downward movement
            if pos_line_down - 5 < center_y < pos_line_down + 5:
                if label == "car":
                    downward_cars += 1
                elif label == "bus":
                    downward_buses += 1
                elif label == "truck":
                    downward_trucks += 1
                elif label in ["bicycle", "motorbike"]:
                    downward_bikes += 1


    # Draw counting lines
    cv2.line(frame, (0, pos_line_up), (frame.shape[1], pos_line_up), (0, 255, 255), 2)
    cv2.line(frame, (0, pos_line_down), (frame.shape[1], pos_line_down), (255, 255, 0), 2)

    # Display counts for both directions
    cv2.putText(frame, f"Upward Cars: {upward_cars}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Downward Cars: {downward_cars}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Upward Buses: {upward_buses}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Downward Buses: {downward_buses}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Upward Trucks: {upward_trucks}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Downward Trucks: {downward_trucks}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Upward Bikes: {upward_bikes}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Downward Bikes: {downward_bikes}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show the video
    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




