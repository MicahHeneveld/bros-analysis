import cv2
import os

def label_frames(dataset_path, num_frames):
    # Load the model's boxes (assuming they are stored in a file)
    model_boxes = load_model_boxes(dataset_path)

    for i in range(num_frames):
        frame_path = os.path.join(dataset_path, f'frame_{i}.jpg')
        frame = cv2.imread(frame_path)

        # Adopt and correct model's boxes
        for box in model_boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw new boxes if necessary
        while True:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):
                # Adopt all model's boxes
                break
            elif key == ord('d'):
                # Draw new box
                x, y = cv2.getMousePos()
                cv2.rectangle(frame, (x, y), (x + 10, y + 10), (0, 0, 255), 2)

        # Save the labeled frame
        cv2.imwrite(os.path.join(dataset_path, f'frame_{i}_labeled.jpg'), frame)

def load_model_boxes(dataset_path):
    # Placeholder function to load model's boxes
    # Replace this with actual logic to load model's boxes
    return [(10, 10, 50, 50), (100, 100, 30, 30)]

if __name__ == "__main__":
    dataset_path = 'path_to_ultrank_11_20_dataset'
    num_frames = 250
    label_frames(dataset_path, num_frames)