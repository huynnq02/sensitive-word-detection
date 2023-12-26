from django.urls import path
from .views import hate_speech_detection

urlpatterns = [
    path('detect-hate-speech/', hate_speech_detection, name='detect-hate-speech'),
]