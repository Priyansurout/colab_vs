import cv2
from ultralytics import YOLO
import sys

# --- TODO: Step 1 ---
# a) Put the path to your custom player detection model here
#    (This is the '.pt' file you downloaded for the assignment)
model_path = 'best.pt' 

# b) Put the path to the 15-second player video here
video_path = '15sec_input_720p.mp4' # 

# --- Load Your Custom Model ---
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# --- Open the Video File ---
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
except IOError as e:
    print(f"Error: {e}")
    sys.exit()

# --- Process the Video ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video.")
        break

    # Run detection on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("Player Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()