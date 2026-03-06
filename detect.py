from ultralytics import YOLO
import cv2

# load model
model = YOLO("yolov8m.pt")

# open webcam
cap = cv2.VideoCapture(0)

# fullscreen window
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLO Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_count = 0
skip_frames = 3   # run detection every 3 frames
annotated = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # run YOLO only every few frames
    if frame_count % skip_frames == 0:
        results = model.predict(
            source=frame,
            imgsz=320,
            conf=0.5,
            classes=[0,2,3,5,7],
            device="cpu"
        )

        annotated = results[0].plot()

    # show last detected frame
    if annotated is not None:
        cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()