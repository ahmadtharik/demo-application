from django.shortcuts import render
from rest_framework import generics, status
from .models import ProcessedImage, Tag
from rest_framework.response import Response
from .serializers import ProcessedImageSerializer
from rest_framework.views import APIView

import os
import cv2
from PIL import Image
import numpy as np


import tensorflow as tf
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
        for tag_name, value_percentage in tags.items():
            tag = Tag.objects.create(tag_name=tag_name,
                                     value_percentage=value_percentage)
            tag_objects.append(tag)
        instance.tags.set(tag_objects)
        instance.save()
        return Response({"message": "Image uploaded and tags generated successfully."}, status=status.HTTP_201_CREATED)




def generate_tags(image):
    message = ""
    prediction = ""

    try:
        image = image
        path = str(settings.MEDIA_ROOT) + "/" + image.name
        imag=cv2.imread(path)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((50, 50))

        test_image =np.expand_dims(resized_image, axis=0) 

        # load model
        model = tf.keras.models.load_model(os.getcwd() + '/model.h5')

        result = model.predict(test_image)

        print("Prediction: " + str(np.argmax(result)))

        if (np.argmax(result) == 0):
            prediction = "Cat"
        elif (np.argmax(result) == 1):
            prediction = "Dog"
        else:
            prediction = "Unknown"

        print(prediction)
    except MultiValueDictKeyError:

        print("No image selected")

    return "path/to/image", {'tag1': 0.75, 'tag2': 0.9, 'tag3': 0.5}

