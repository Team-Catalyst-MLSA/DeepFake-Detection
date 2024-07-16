from django.urls import path, include
from deepfake_detection_app import views

urlpatterns = [
    path('', include('deepfake_detection_app.urls')),
]


