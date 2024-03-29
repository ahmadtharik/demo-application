from django.db import models

# Create your models here.

class Tag(models.Model):
    tag_name = models.CharField(max_length=255)
    value_percentage = models.FloatField()


class ProcessedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    processed_image = models.ImageField(upload_to='processed_images/', blank=True)
    tags = models.ManyToManyField(Tag, blank=True, related_name='tag')
    base_processed_image = models.ImageField(upload_to='processed_images/',
                                             blank=True)
    base_tags = models.ManyToManyField(Tag, blank=True, related_name='base_tag')

