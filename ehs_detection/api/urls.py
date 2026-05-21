from django.contrib import admin
from django.urls import path,include
from api.views import *
urlpatterns = [
    path('frame', video_feed, name='video_feed'),
    path('person-track/', person_tracking_feed, name='person_tracking_feed'),
    path('image_upload/', DetectionCreateAPI.as_view(), name='detect'),
    path('get_image/', DetectionListAPI.as_view(), name='detections'),
]
