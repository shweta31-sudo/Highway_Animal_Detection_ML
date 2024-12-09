import random
import cv2
from ultralytics import YOLO

# Open the file in read mode
with open("utils/coco.txt", "r") as my_file:
    # Read the file and split the text when a newline ('\n') is seen
    class_list = my_file.read().split("\n")

# Specify the animal classes you want to detect
animal_classes = [
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
]

# Generate random colors for class list
detection_colors = {}
for cls in animal_classes:
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors[cls] = (b, g, r)

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Video frame dimensions
frame_wid = 800
frame_hyt = 500

#cap = cv2.VideoCapture(0)
# Open a video capture stream (change the source as needed)
cap = cv2.VideoCapture("inference/videos/animal6.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize the frame
    frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on the frame
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Get the class name
            class_name = class_list[int(clsID)]

            # Check if the detected class is an animal
            if class_name in animal_classes:
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[class_name],
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_name + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate the loop when "Q" is pressed
    if cv2.waitKey(1) == ord("x"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
