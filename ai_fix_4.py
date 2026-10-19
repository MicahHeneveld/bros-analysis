import cv2
import os
import json

# Constants
DATASET_PATH = 'path/to/ultrank_11_20'
LABELS_PATH = 'path/to/labels'
FRAME_COUNT = 250

# Load model predictions (assuming a function that loads predictions)
def load_model_predictions(frame_index):
    # Placeholder function
    return {'boxes': [[100, 100, 200, 200], [300, 300, 400, 400]]}

# Save labels to JSON
def save_labels(frame_index, labels):
    with open(os.path.join(LABELS_PATH, f'frame_{frame_index}.json'), 'w') as f:
        json.dump(labels, f)

# Main labeling function
def label_frames():
    for frame_index in range(FRAME_COUNT):
        frame_path = os.path.join(DATASET_PATH, f'frame_{frame_index}.jpg')
        frame = cv2.imread(frame_path)
        predictions = load_model_predictions(frame_index)
        
        # Draw model predictions
        for box in predictions['boxes']:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Display frame and get user input
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('a'):
            # Adopt model predictions
            labels = predictions
        elif key == ord('d'):
            # Draw new boxes
            labels = {'boxes': []}
            while True:
                cv2.imshow('Frame', frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s'):
                    break
                elif key == ord('n'):
                    new_box = [int(x) for x in input("Enter new box coordinates (x1, y1, x2, y2): ").split()]
                    labels['boxes'].append(new_box)
        
        # Save labels
        save_labels(frame_index, labels)
    
    cv2.destroyAllWindows()

# Run the labeling process
label_frames()