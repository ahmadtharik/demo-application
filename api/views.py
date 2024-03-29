import os
from collections import defaultdict

import cv2
import torch
import yolov7
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.utils.datastructures import MultiValueDictKeyError
from PIL import Image
from rest_framework import generics, status
from rest_framework.response import Response
from torchvision import transforms
from ultralytics import YOLO

from api.models import ProcessedImage, Tag

from .serializers import ProcessedImageSerializer


class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

class UploadImageView(generics.CreateAPIView):
    queryset = ProcessedImage.objects.all()
    serializer_class = ProcessedImageSerializer

    def perform_create(self, serializer):
        instance = serializer.save()

        processed_image, base_processed_image, tags, base_tags = generate_tags(instance.image)

        instance.processed_image = processed_image
        instance.base_processed_image = base_processed_image
        tag_objects = []
        for tag_name, value_percentage in tags:
            tag = Tag.objects.create(tag_name=tag_name,
                                     value_percentage=value_percentage)
            tag_objects.append(tag)
        instance.tags.set(tag_objects)

        base_tag_objects = []

        for tag_name, value_percentage, _ in base_tags:

            tag = Tag.objects.create(tag_name=tag_name,
                                     value_percentage=value_percentage)
            base_tag_objects.append(tag)
        instance.base_tags.set(base_tag_objects)

        instance.save()
        return Response({"message": "Image uploaded and tags generated successfully."}, status=status.HTTP_201_CREATED)



OBJECT_DETECTION_THRESHOLD=0.15
CLASS_WEIGHT_THRESHOLD=0.1

def calculate_class_weights(detections):
    if not detections:
        print("No detections found")
        return None

    class_stats = defaultdict(lambda: {'total_weight': 0, 'count': 0})

    max_confidence = max(confidence for _, confidence, _ in detections)
    max_box_size = max(
        (x2 - x1) * (y2 - y1) for  _, _, (x1, y1, x2, y2) in detections)

    for class_id, confidence, box in detections:
        x1, y1, x2, y2 = box
        box_size = (x2 - x1) * (y2 - y1)
        print(class_id,confidence)
        if confidence < OBJECT_DETECTION_THRESHOLD:
            continue

        weight = (box_size / max_box_size) * (confidence / max_confidence)

        class_stats[class_id]['total_weight'] += weight
        class_stats[class_id]['count'] += 1

    for class_id, stats in class_stats.items():
        stats['total_weight'] /= max(stats['count'], 1)

    total_confidence = sum(
        stats['total_weight'] for stats in class_stats.values())

    class_weights = {class_id: stats['total_weight'] / total_confidence for
                     class_id, stats in class_stats.items()}
    # after weights are determined, delete those with low weights
    class_weights = {class_id: weight for class_id, weight in class_weights.items() if weight > CLASS_WEIGHT_THRESHOLD}

    return class_weights


def generate_tags(image):
    base_predictions = []
    predictions = []

    class_names = {
        0: 'beach',
        1: 'field',
        2: 'forest',
        3: 'mountain',
        4: 'underwater',
    }


    try:

        image = image
        path = str(settings.MEDIA_ROOT) + "/" + image.name
        imag = cv2.imread(path)

        # Classification Model

        model = torch.load('classifier.pth')
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        transformed_image = Image.open(path)
        transformed_image = transform(transformed_image)
        transformed_image = transformed_image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(transformed_image)

        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        predictions.append((class_names[predicted_class], confidence))

        # Base Model
        model2 = YOLO('yolov8x-oiv7.pt')

        results2 = model2.predict(source=imag, conf=0.25)

        boxes = results2[0].boxes.xyxy.tolist()
        classes = results2[0].boxes.cls.tolist()
        names = results2[0].names
        confidences = results2[0].boxes.conf.tolist()

        for box, class_id, confidence in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            label = names[class_id]
            base_predictions.append((label, confidence,box))
            color = (0, 255, 0)
            cv2.rectangle(imag, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(imag, f'{label} {confidence:.2f}',
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        base_image_path = os.path.join('modified', (image.name))

        base_modified_image_path = os.path.join(settings.MEDIA_ROOT, 'modified', 'uploads',
                                           os.path.basename(image.name))
        cv2.imwrite(base_modified_image_path, imag)

        # General Model
        model = yolov7.load('best.pt')
        imag = cv2.imread(path)
        results = model(imag)

        predictions_general = results.pred[0]
        boxes_general = predictions_general[:, :4].tolist()
        classes_general = predictions_general[:, 5].tolist()
        names_general = model.names
        confidences_general = predictions_general[:, 4].tolist()

        for box_general, class_id_general, confidence_general in zip(
                boxes_general, classes_general, confidences_general):
            label_general = names_general[int(class_id_general)]
            predictions.append((label_general, confidence_general, box_general))

        # Animal Model
        model1 = yolov7.load('animals-best.pt')
        imag2 = cv2.imread(path)
        results_animal  = model1(imag2)

        predictions_animal = results_animal.pred[0]
        boxes_animal = predictions_animal[:, :4].tolist()
        classes_animal = predictions_animal[:, 5].tolist()
        names_animal = model1.names
        confidences_animal = predictions_animal[:, 4].tolist()

        for box_animal, class_id_animal, confidence_animal in zip(boxes_animal,
                                                                  classes_animal,
                                                                  confidences_animal):
            label_animal = names_animal[int(class_id_animal)]
            predictions.append((label_animal, confidence_animal, box_animal))



        # performing the Tags
        weighted_classes = (calculate_class_weights(predictions[1:]))
        # in case no classes are found return an empty array
        weighted_classes = [(class_name, weight)
                            for class_name, weight in weighted_classes.items()] if weighted_classes else []

        weighted_classes.sort(key=lambda x: x[1], reverse=True)

        weighted_classes.insert(0, predictions[0])

        i = -1
        for label, confidence, box in predictions[1:]:
            i = i+1
            x1, y1, x2, y2 = box

            # if the object detection confidence or the weighted significance is below the threshold, skip
            if confidence < OBJECT_DETECTION_THRESHOLD:
                continue
            color = (0, 255, 0)
            cv2.rectangle(imag, (int(x1), int(y1)), (int(x2), int(y2)), color,
                          2)
            cv2.putText(imag, f'{label} {confidence:.2f}',
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        image_path = os.path.join('base_modified', (image.name))

        modified_image_path = os.path.join(settings.MEDIA_ROOT, 'base_modified',
                                           'uploads',
                                           os.path.basename(image.name))
        cv2.imwrite(modified_image_path, imag)




    except MultiValueDictKeyError:
        print("No image selected")

    return image_path, base_image_path,  weighted_classes, base_predictions

