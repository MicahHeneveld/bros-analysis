import cv2
import os

# Constants
DATASET_PATH = 'path/to/ultrank_11_20'
LABELS_PATH = 'path/to/labels'
FRAME_COUNT = 250

# Function to draw bounding boxes
def draw_bounding_boxes(image, boxes):
    for box in boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    return image

# Function to save labels
def save_labels(frame_number, boxes):
    label_path = os.path.join(LABELS_PATH, f'frame_{frame_number}.txt')
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(f'{int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n')

# Main labeling loop
for frame_number in range(1, FRAME_COUNT + 1):
    frame_path = os.path.join(DATASET_PATH, f'frame_{frame_number}.jpg')
    image = cv2.imread(frame_path)
    
    # Load model's boxes (assuming they are stored in a list of [x1, y1, x2, y2] format)
    model_boxes = load_model_boxes(frame_number)  # You need to implement this function
    
    # Draw model's boxes
    image_with_boxes = draw_bounding_boxes(image.copy(), model_boxes)
    
    # Display the image and get user input
    cv2.imshow('Label Frame', image_with_boxes)
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('a'):
        # Adopt model's boxes and fix
        final_boxes = adopt_and_fix_boxes(model_boxes)
    elif key == ord('d'):
        # Draw labels manually
        final_boxes = draw_boxes_manually(image)
    
    # Save labels
    save_labels(frame_number, final_boxes)
    
    # Clean up
    cv2.destroyAllWindows()

print("Labeling completed. Please review and claim the bounty.")