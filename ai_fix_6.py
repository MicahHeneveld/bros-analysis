import cv2
import os

def label_frames(dataset_path, num_frames):
    for i in range(num_frames):
        frame_path = os.path.join(dataset_path, f'frame_{i}.jpg')
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Frame {i} not found.")
            continue
        
        model_boxes = get_model_boxes(frame_path)  # Assume this function fetches model's bounding boxes
        cv2.imshow('Frame', frame)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):
                for box in model_boxes:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.imshow('Frame', frame)
            elif key == ord('d'):
                new_box = draw_new_box(frame)  # Assume this function draws a new bounding box
                cv2.rectangle(frame, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (0, 0, 255), 2)
                cv2.imshow('Frame', frame)
            elif key == ord('s'):
                save_labels(frame_path, model_boxes, new_boxes)  # Assume this function saves the labels
                break
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()

def get_model_boxes(frame_path):
    # Placeholder function to fetch model's bounding boxes
    return [(10, 10, 50, 50), (100, 100, 150, 150)]

def draw_new_box(frame):
    # Placeholder function to draw a new bounding box
    return (200, 200, 250, 250)

def save_labels(frame_path, model_boxes, new_boxes):
    # Placeholder function to save the labels
    with open(frame_path.replace('.jpg', '.txt'), 'w') as f:
        for box in model_boxes + new_boxes:
            f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")

if __name__ == "__main__":
    dataset_path = 'path_to_dataset'
    num_frames = 250
    label_frames(dataset_path, num_frames)