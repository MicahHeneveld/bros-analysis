import cv2
import os

def label_frames(dataset_path, num_frames):
    for i in range(num_frames):
        frame_path = os.path.join(dataset_path, f'frame_{i}.jpg')
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Unable to read frame {i}")
            continue
        
        while True:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('a'):
                # Adopt model's boxes and correct them
                # Placeholder for model's boxes adoption and correction logic
                pass
            elif key == ord('d'):
                # Draw new boxes
                # Placeholder for drawing new boxes logic
                pass
            elif key == ord('q'):
                # Save labels and break the loop
                cv2.imwrite(frame_path, frame)
                break
            elif key == ord('c'):
                # Cancel and break the loop
                break
        
        if key == ord('c'):
            print(f"Cancelled labeling of frame {i}")
            continue

if __name__ == "__main__":
    dataset_path = 'path_to_dataset'
    num_frames = 250
    label_frames(dataset_path, num_frames)