from django.urls import path
from . import views

urlpatterns = [
    path('', views.deepfake_detection, name='deepfake_detection'),
]
