# Generated by Django 5.0.3 on 2024-03-29 03:52

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0002_processedimage_processed_image"),
    ]

    operations = [
        migrations.AddField(
            model_name="processedimage",
            name="base_processed_image",
            field=models.ImageField(blank=True, upload_to="processed_images/"),
        ),
        migrations.AddField(
            model_name="processedimage",
            name="base_tags",
            field=models.ManyToManyField(
                blank=True, related_name="base_tag", to="api.tag"
            ),
        ),
        migrations.AlterField(
            model_name="processedimage",
            name="tags",
            field=models.ManyToManyField(blank=True, related_name="tag", to="api.tag"),
        ),
    ]