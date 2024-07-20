import warnings
import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.ssd import SSDHead, det_utils
from torchvision.models.detection import ssd300_vgg16
from image_preprocess import *
warnings.filterwarnings("ignore")
import time

# Function to calculate IoU (Intersection over Union) between two bounding boxes
def calculate_iou(box1, box2):
    # Calculate coordinates of intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate area of box1
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)

    # Calculate percentage of box1 area that is inside box2
    if box1_area > 0 and box1[3]<box2[3]:
        percentage_inside = intersection_area / float(box1_area)
    else:
        percentage_inside = 0.0

    return percentage_inside


def imageTest():
    # Predict on a new image
    image_path = "Car01.jpg"  # Replace with your image path
    input_image = preprocess_image(image_path)

    with torch.no_grad():
        prediction = model(input_image)[0]

    # Postprocess the prediction
    processed_prediction = preprocess_bbox(prediction)

    print(processed_prediction)

    # Display the image with bounding boxes and class names
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    show_bbox(input_image[0], processed_prediction, ax)
    plt.title("Predicted Bounding Boxes with Class Names")
    plt.show()

def resize_frame(frame, max_width=800, max_height=600):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    
    if width > max_width or height > max_height:
        if width > height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height))
    else:
        resized_frame = frame

    return resized_frame


def resize_video(frame, max_width=800, max_height=600):
    or_height, or_width = frame.shape[:2]
    frame_heigth , frame_width = 300 , 300

    return or_height/frame_heigth, or_width/frame_width

def video_monitoring():
    # Initialize variables for object counting
    print("Monitoring Initializing......")
    object_count = {cls: 0 for cls in classes}

    # Open a video file
    video_path = "4K Road traffic.mp4"  # Replace with your video path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file or camera.")
        return

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    # frame_rate=1
    current_frame = 0
    last_frame = None

# and current_frame < target_frame_count
    # Process the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1
        last_frame = frame

        # if current_frame % frame_rate == 1:
        #     # Process the first frame of each second
        #     first_input_image = preprocess_frame(frame)
        #     with torch.no_grad():
        #         prediction = model(first_input_image)[0]
        #         processed_prediction = preprocess_bbox(prediction)
        #         print("First frame:",processed_prediction)
        # last_frame = frame

        # Process only the last frame of each second
        if current_frame % frame_rate == 0:
            current_frame = 0
            # Preprocess the last frame
            input_image = preprocess_frame(last_frame)

            # Perform object detection
            with torch.no_grad():
                prediction = model(input_image)[0]

            processed_prediction = preprocess_bbox(prediction)

            # print(processed_prediction)
            # Extract boxes, scores, and labels from prediction
            boxes = processed_prediction['boxes'].cpu().numpy()
            scores = processed_prediction['scores'].cpu().numpy()
            labels = processed_prediction['labels'].cpu().numpy()

            # Calculate center of each box and compare with fixed y-axis point

            for box, score, label in zip(boxes, scores, labels):
                # Compare box center with fixed y-axis point

                iou = calculate_iou(box, boundarBox)
                if(label==1 or label==3):
                     iou_thresh=0.5
                else:
                     iou_thresh=0.2

                if iou>iou_thresh:  # Adjust threshold as needed
                    class_name = classes[label]
                    object_count[class_name] += 1
                    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    # show_bbox(input_image[0], processed_prediction, ax)
                    # plt.title("Predicted Bounding Boxes with Class Names")
                    # plt.show()
                    # Draw bounding boxes and labels on the frame
                    # Draw bounding boxes and labels on the frame
                            # Resize frame to fit display if needed
                            
                ratio_heigth, ratio_width= resize_video(last_frame)


                # Scale bounding boxes to the original frame size
                scaled_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    scaled_box = [
                        int(x1*ratio_width),
                        int(y1*ratio_heigth),
                        int(x2*ratio_width),
                        int(y2*ratio_heigth)
                    ]
                    scaled_boxes.append(scaled_box)

                scaled_boxes = np.array(scaled_boxes)

                # Update the processed prediction with scaled boxes
                processed_prediction['boxes'] = torch.tensor(scaled_boxes, dtype=torch.float32)

            video_fram = preprocess_video(last_frame)
            annotated_frame = show_bbox(video_fram[0], processed_prediction,ratio_heigth, ratio_width)

                    # Resize the frame to fixed dimensions
            resized_frame = resize_frame(annotated_frame)
                # Prepare the count text
            count_text = "\n".join([f"{cls}: {count}" for cls, count in list(object_count.items())[1:]])
            y0, dy = 12, 15  # Starting y position and line height

                # Calculate text size
            text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_width, text_height = text_size
            box_height = (dy * len(count_text.split('\n'))) + 1  # Add padding

                # Draw a black background rectangle
            cv2.rectangle(resized_frame, (2, y0 - 20), (130, y0 + box_height), (0, 0, 0), cv2.FILLED)

                # Draw the text
            for i, line in enumerate(count_text.split('\n')):
                    cv2.putText(resized_frame, line, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Video Monitoring', resized_frame)

                    # Exit the video display window when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring Finished......")
    
    # Print the final object counts
    print("Object Counts:")
    for cls, count in object_count.items():
        print(f"{cls}: {count}")


def realtime_camera_video():
    # Initialize variables for object counting
    print("Monitoring Initializing......")
    object_count = {cls: 0 for cls in classes}
    processed_prediction = {'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])}

    # Open a video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    last_frame = None

    # Process the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame

        # Preprocess the last frame
        input_image = preprocess_frame(last_frame)

        # Perform object detection
        with torch.no_grad():
            prediction = model(input_image)[0]

        processed_prediction = preprocess_bbox(prediction)

        # Extract boxes, scores, and labels from prediction
        boxes = processed_prediction['boxes'].cpu().numpy()
        scores = processed_prediction['scores'].cpu().numpy()
        labels = processed_prediction['labels'].cpu().numpy()

        # Calculate center of each box and compare with fixed y-axis point
        for box, score, label in zip(boxes, scores, labels):
            iou = calculate_iou(box, boundarBox)
            if label == 1 or label == 3:
                iou_thresh = 0.5
            else:
                iou_thresh = 0.2

            if iou > iou_thresh:  # Adjust threshold as needed
                class_name = classes[label]
                object_count[class_name] += 1

        # Resize frame to fit display if needed
        ratio_height, ratio_width = resize_video(last_frame)

        # Scale bounding boxes to the original frame size
        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            scaled_box = [
                int(x1 * ratio_width),
                int(y1 * ratio_height),
                int(x2 * ratio_width),
                int(y2 * ratio_height)
            ]
            scaled_boxes.append(scaled_box)

        scaled_boxes = np.array(scaled_boxes)

        # Update the processed prediction with scaled boxes
        processed_prediction['boxes'] = torch.tensor(scaled_boxes, dtype=torch.float32)

        video_frame = preprocess_video(last_frame)
        annotated_frame = show_bbox(video_frame[0], processed_prediction, ratio_height, ratio_width)

        # Resize the frame to fixed dimensions
        resized_frame = resize_frame(annotated_frame)

        # Prepare the count text
        count_text = "\n".join([f"{cls}: {count}" for cls, count in list(object_count.items())[1:]])
        y0, dy = 12, 15  # Starting y position and line height

        # Calculate text size
        text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_width, text_height = text_size
        box_height = (dy * len(count_text.split('\n'))) + 1  # Add padding

        # Draw a black background rectangle
        cv2.rectangle(resized_frame, (2, y0 - 20), (130, y0 + box_height), (0, 0, 0), cv2.FILLED)

        # Draw the text
        for i, line in enumerate(count_text.split('\n')):
            cv2.putText(resized_frame, line, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Video Monitoring', resized_frame)

        # Sleep for 500 milliseconds before processing the next frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.5)  # Sleep for 500 milliseconds

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring Finished......")

    # Print the final object counts
    print("Object Counts:")
    for cls, count in object_count.items():
        print(f"{cls}: {count}")



boundarBox =[0, 200 ,300, 300]
num_classes = 9
model_learned_weights = "model.pth"
model = ssd300_vgg16(weights=None, weights_backbone=None)
in_channels = det_utils.retrieve_out_channels(model.backbone, (300, 300))
num_anchors = model.anchor_generator.num_anchors_per_location()
model.head = SSDHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes)
model.load_state_dict(torch.load(model_learned_weights, map_location=torch.device('cpu')))
model.eval()

start_time = time.time()
video_monitoring()
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.4f} seconds")