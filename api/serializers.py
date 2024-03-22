from rest_framework import serializers
from .models import ProcessedImage, Tag

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ('tag_name', 'value_percentage')

class ProcessedImageSerializer(serializers.ModelSerializer):
    tags = TagSerializer(many=True, read_only=True)

    class Meta:
        model = ProcessedImage
        fields = ('id', 'image', 'processed_image', 'tags')

