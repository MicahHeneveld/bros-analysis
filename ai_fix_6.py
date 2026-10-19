import cv2
import numpy as np

def label_frames(video_path, output_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('a'):
            # Adopt model's boxes (placeholder for actual model output)
            model_boxes = get_model_boxes(frame)
            draw_boxes(frame, model_boxes)
        elif key == ord('d'):
            # Draw labels manually
            draw_boxes(frame, draw_boxes_manually(frame))

        out.write(frame)

        if key == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def get_model_boxes(frame):
    # Placeholder function to get model boxes
    return []

def draw_boxes(frame, boxes):
    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

def draw_boxes_manually(frame):
    # Placeholder function to draw boxes manually
    boxes = []
    while True:
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('d'):
            x, y, w, h = cv2.selectROI('Frame', frame, fromCenter=False, showCrosshair=True)
            boxes.append((x, y, x + w, y + h))
    return boxes

# Example usage
label_frames('ultrank_11_20.mp4', 'labeled_ultrank_11_20.mp4', 250)