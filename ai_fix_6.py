import cv2
import os

def label_frames(dataset_path, num_frames):
    for i in range(num_frames):
        frame_path = os.path.join(dataset_path, f'frame_{i}.jpg')
        frame = cv2.imread(frame_path)
        model_boxes = get_model_boxes(frame_path)  # Assume this function retrieves model's bounding boxes

        while True:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('a'):
                for box in model_boxes:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.imwrite(frame_path, frame)
                break
            elif key == ord('d'):
                new_box = draw_new_box(frame)
                model_boxes.append(new_box)
                cv2.rectangle(frame, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (0, 0, 255), 2)
                cv2.imwrite(frame_path, frame)

        cv2.destroyAllWindows()

def get_model_boxes(frame_path):
    # Placeholder function to simulate model's bounding box retrieval
    return [(100, 100, 200, 200), (300, 300, 400, 400)]

def draw_new_box(frame):
    # Placeholder function to simulate drawing a new bounding box
    return (150, 150, 250, 250)

if __name__ == "__main__":
    dataset_path = 'path_to_dataset'
    num_frames = 250
    label_frames(dataset_path, num_frames)