from django.urls import path
from .views import UploadImageView

urlpatterns = [
    path('upload_image', UploadImageView.as_view(), name='upload_image'),

]