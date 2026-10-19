import os
import cv2
import numpy as np

def load_frames(dataset_path):
    frames = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            frame_path = os.path.join(dataset_path, filename)
            frame = cv2.imread(frame_path)
            frames.append(frame)
    return frames

def draw_bounding_boxes(frame, boxes):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def save_labels(dataset_path, labels):
    for i, label in enumerate(labels):
        with open(os.path.join(dataset_path, f"frame_{i+1}.txt"), "w") as f:
            for box in label:
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")

def main():
    dataset_path = "path/to/ultrank_11_20"
    frames = load_frames(dataset_path)
    labels = []

    for i, frame in enumerate(frames):
        print(f"Labeling frame {i+1} of {len(frames)}")
        boxes = model.predict(frame)  # Assuming model.predict returns bounding boxes
        labeled_frame = draw_bounding_boxes(frame, boxes)
        cv2.imshow("Frame", labeled_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        label = []
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):
                label.extend(boxes)
                break
            elif key == ord('d'):
                x, y, w, h = cv2.selectROI("Frame", labeled_frame, fromCenter=False, showCrosshair=True)
                label.append([x, y, w, h])
                cv2.destroyWindow("Frame")
        labels.append(label)

    save_labels(dataset_path, labels)
    print("Labels saved successfully.")

if __name__ == "__main__":
    main()