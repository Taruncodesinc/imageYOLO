import cv2
import pyttsx3
import numpy as np
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to speak out the provided text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load YOLO
net = cv2.dnn.readNet("yolov3-spp.weights", "yolov3-spp.cfg")

# Load names of classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the indices of the output layers
layer_indices = net.getUnconnectedOutLayers()

# Get the names of the output layers using indices
layer_names = net.getLayerNames()
output_layers = []
for idx in layer_indices:
    layer_name = layer_names[idx - 1]
    output_layers.append(layer_name)

# Directory containing images to be processed
image_dir = r"C:\Users\visha\OneDrive\Desktop\facedetection"




# Iterate over images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Read the image
        image_path = os.path.join(image_dir, filename)
        frame = cv2.imread(image_path)

        # Resize the frame to speed up the detection process
        frame = cv2.resize(frame, None, fx=3, fy=3)
        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        spoken_labels = set()  # Set to keep track of spoken labels

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
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

        # Apply non-max suppression to remove overlapping bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels on the detected objects
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

                # Speak out the detected object's label if it hasn't been spoken before
                if label not in spoken_labels:
                    speak(label)
                    spoken_labels.add(label)

        # Show the frame
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)  # Wait for key press to move to the next image

# Close all windows
cv2.destroyAllWindows()
