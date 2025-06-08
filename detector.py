import cv2
import numpy as np
from ultralytics import YOLO

# print(cv2.__version__)
# print(np.__version__)

# model
model = YOLO('yolov8n.pt')

vid_path = "object_car.mp4"
cap = cv2.VideoCapture(vid_path)

# Loop through each frame of the video
while cap.isOpened:

    ret , frame = cap.read()

    if not ret:
        print("End of video stream.")
        break

    cv2.imshow("Player Tracking'", frame)

    if cv2.waitKey(25) & 0xFF == ord('c'):
        break



cap.release()
cv2.destroyAllWindows()