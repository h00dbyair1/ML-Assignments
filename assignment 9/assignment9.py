import cv2
import os
from ultralytics import YOLO

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

VIDEO_PATH = "cctv.mp4"
OUTPUT_PATH = "output_cctv.mp4"

# Pretrained YOLO model
MODEL_PATH = "yolo11n.pt"

# Minimum confidence for a detection
CONFIDENCE = 0.40

# Set to False if you do not want the video window
SHOW_VIDEO = True


# --------------------------------------------------
# CHECK INPUT VIDEO
# --------------------------------------------------

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Video file '{VIDEO_PATH}' was not found.\n"
        "Place the CCTV video in the same folder as this Python file."
    )


# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# OPEN CCTV VIDEO
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Unable to open the CCTV video.")


# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Some videos may not provide a valid FPS
if fps <= 0:
    fps = 25.0


# --------------------------------------------------
# CREATE OUTPUT VIDEO
# --------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Unable to create the output video.")


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

frame_count = 0

print("\nStarting object detection...")
print("Press ESC to stop the program.\n")

try:

    while True:

        # Read one frame
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # --------------------------------------------------
        # RUN YOLO OBJECT DETECTION
        # --------------------------------------------------

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]

        # YOLO automatically draws bounding boxes,
        # object names and confidence values.
        annotated_frame = result.plot()


        # --------------------------------------------------
        # COUNT DETECTED OBJECTS
        # --------------------------------------------------

        object_counts = {}

        if result.boxes is not None:

            for class_id in result.boxes.cls.tolist():

                class_name = model.names[int(class_id)]

                if class_name not in object_counts:
                    object_counts[class_name] = 0

                object_counts[class_name] += 1


        # Total objects detected in current frame
        total_objects = sum(object_counts.values())


        # --------------------------------------------------
        # DISPLAY TOTAL OBJECT COUNT
        # --------------------------------------------------

        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (260, 55),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated_frame,
            f"Objects Detected: {total_objects}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # --------------------------------------------------
        # DISPLAY OBJECT COUNTS
        # --------------------------------------------------

        y_position = 85

        for object_name, count in object_counts.items():

            text = f"{object_name}: {count}"

            cv2.putText(
                annotated_frame,
                text,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            y_position += 25


        # --------------------------------------------------
        # SAVE PROCESSED FRAME
        # --------------------------------------------------

        writer.write(annotated_frame)


        # --------------------------------------------------
        # SHOW VIDEO
        # --------------------------------------------------

        if SHOW_VIDEO:

            cv2.imshow(
                "CCTV Object Detection",
                annotated_frame
            )

            # ESC key terminates the program
            if cv2.waitKey(1) & 0xFF == 27:
                print("\nDetection stopped by user.")
                break


finally:

    # --------------------------------------------------
    # RELEASE RESOURCES
    # --------------------------------------------------

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print("\nObject detection completed successfully.")
print(f"Total frames processed: {frame_count}")
print(f"Output video saved as: {OUTPUT_PATH}")