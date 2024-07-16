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


def video_monitoring():
    # Initialize variables for object counting
    print("Monitoring Initializing......")
    object_count = {cls: 0 for cls in classes}

    # Open a video file
    video_path = "videoplayback.mp4"  # Replace with your video path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file or camera.")
        return

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    target_frame_count=frame_rate
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


            # Extract boxes, scores, and labels from prediction
            boxes = processed_prediction['boxes'].cpu().numpy()
            scores = processed_prediction['scores'].cpu().numpy()
            labels = processed_prediction['labels'].cpu().numpy()

            # Calculate center of each box and compare with fixed y-axis point

            for box, score, label in zip(boxes, scores, labels):
                # Calculate box center
                box_center_y = ((box[3] - box[1]) / 2) + box[1]
                box_center_y = round(box_center_y, 1)
                # Compare box center with fixed y-axis point

                iou = calculate_iou(box, [50, 50 ,250, 150])
                if iou>0.5:  # Adjust threshold as needed
                    class_name = classes[label]
                    object_count[class_name] += 1
            # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            # show_bbox(input_image[0], processed_prediction, ax)
            # plt.title("Predicted Bounding Boxes with Class Names")
            # plt.show()

    # Release video capture and close windows
    cap.release()
    print("Monitoring Finished......")
    # Print the final object counts
    print("Object Counts:")
    for cls, count in object_count.items():
        print(f"{cls}: {count}")




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