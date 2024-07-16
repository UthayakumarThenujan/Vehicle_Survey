import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision

classes = ["background", "Auto", "Bus", "Car", "LCV", "Motorcycle", "Truck", "Tractor", "Multi-Axle"]
threshold = 0.5
iou_threshold = 0.8

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def show_bbox(img, target, axis, color=(0, 255, 0)):
    img = np.transpose(img.cpu().numpy(), (1, 2, 0))
    boxes = target["boxes"].cpu().numpy().astype("int")
    labels = target["labels"].cpu().numpy()
    img = img.copy()
    for i, box in enumerate(boxes):
        idx = int(labels[i])
        text = classes[idx]

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 10
        cv2.putText(img, text, (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    specific_box = [(50, 50), (250, 150)]
    cv2.rectangle(img, specific_box[0], specific_box[1], color, 2)
    axis.imshow(img)
    axis.axis("off")

def preprocess_bbox(prediction):
    processed_bbox = {}
    boxes = prediction["boxes"][prediction["scores"] >= threshold]
    scores = prediction["scores"][prediction["scores"] >= threshold]
    labels = prediction["labels"][prediction["scores"] >= threshold]
    nms = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold)

    processed_bbox["boxes"] = boxes[nms]
    processed_bbox["scores"] = scores[nms]
    processed_bbox["labels"] = labels[nms]

    return processed_bbox


def preprocess_frame(frame):
    image = Image.fromarray(np.uint8(frame)).convert("RGB")
    transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)

