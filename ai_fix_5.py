import cv2
import os

# Constants
DATASET_PATH = 'path/to/ultrank_11_20'
LABELS_PATH = 'path/to/labels'
MODEL_BOXES_PATH = 'path/to/model_boxes'
FRAMES_TO_LABEL = 250

# Load model boxes
model_boxes = {}
with open(MODEL_BOXES_PATH, 'r') as f:
    for line in f:
        frame, box = line.strip().split(',')
        model_boxes[frame] = eval(box)

# Function to draw labels
def draw_labels(frame, labels):
    for label in labels:
        x, y, w, h = label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

# Function to save labels
def save_labels(frame, labels, frame_index):
    label_file = os.path.join(LABELS_PATH, f'frame_{frame_index}.txt')
    with open(label_file, 'w') as f:
        for label in labels:
            f.write(f'{label[0]} {label[1]} {label[2]} {label[3]}\n')

# Main labeling loop
for frame_index in range(FRAMES_TO_LABEL):
    frame_path = os.path.join(DATASET_PATH, f'frame_{frame_index}.jpg')
    frame = cv2.imread(frame_path)
    
    if frame_index in model_boxes:
        labels = model_boxes[frame_index]
    else:
        labels = []
    
    # Draw model boxes
    frame_with_boxes = draw_labels(frame.copy(), labels)
    
    # Display frame and get user input
    cv2.imshow('Frame', frame_with_boxes)
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('a'):
        # Adopt model boxes and fix inaccuracies
        for i, label in enumerate(labels):
            x, y, w, h = label
            new_x = int(input(f'Enter new x for box {i+1}: '))
            new_y = int(input(f'Enter new y for box {i+1}: '))
            new_w = int(input(f'Enter new w for box {i+1}: '))
            new_h = int(input(f'Enter new h for box {i+1}: '))
            labels[i] = (new_x, new_y, new_w, new_h)
    elif key == ord('d'):
        # Draw labels directly
        labels = []
        while True:
            x, y = cv2.getMousePos()
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('r'):
                frame = cv2.imread(frame_path)
    
    # Save labels
    save_labels(frame, labels, frame_index)
    
    # Clean up
    cv2.destroyAllWindows()

# Claim the bounty
print("Bounty claimed. All frames labeled.")