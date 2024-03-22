from django.shortcuts import render
from rest_framework import generics, status
from .models import ProcessedImage, Tag
from rest_framework.response import Response
from .serializers import ProcessedImageSerializer
from rest_framework.views import APIView
from ultralytics import YOLO
import os
import cv2



from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError

from django.core.files.storage import FileSystemStorage


class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

class UploadImageView(generics.CreateAPIView):
    queryset = ProcessedImage.objects.all()
    serializer_class = ProcessedImageSerializer

    def perform_create(self, serializer):
        instance = serializer.save()

        processed_image, tags = generate_tags(instance.image)
        instance.processed_image = processed_image
        tag_objects = []
        for tag_name, value_percentage in tags:
            tag = Tag.objects.create(tag_name=tag_name,
                                     value_percentage=value_percentage)
            tag_objects.append(tag)
        instance.tags.set(tag_objects)
        instance.save()
        return Response({"message": "Image uploaded and tags generated successfully."}, status=status.HTTP_201_CREATED)




def generate_tags(image):
    predictions = []


    try:
        image = image
        path = str(settings.MEDIA_ROOT) + "/" + image.name
        imag = cv2.imread(path)

        model2 = YOLO('yolov8x-oiv7.pt')

        results2 = model2.predict(source=imag, conf=0.25)

        boxes = results2[0].boxes.xyxy.tolist()
        classes = results2[0].boxes.cls.tolist()
        names = results2[0].names
        confidences = results2[0].boxes.conf.tolist()

        for box, class_id, confidence in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            label = names[class_id]
            predictions.append((label, confidence))
            color = (0, 255, 0)
            cv2.rectangle(imag, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(imag, f'{label} {confidence:.2f}',
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        image_path = os.path.join('modified', (image.name))

        modified_image_path = os.path.join(settings.MEDIA_ROOT, 'modified', 'uploads',
                                           os.path.basename(image.name))
        cv2.imwrite(modified_image_path, imag)

    except MultiValueDictKeyError:

        print("No image selected")

    return image_path, predictions

