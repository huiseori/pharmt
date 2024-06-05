# urls.py
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from pharmtracker import settings
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home_view, name='home'),
    path('home/', views.home_view, name='home'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('ocr/', views.ocr_view, name='ocr'),
path('ocr_process/', views.ocr_process, name='ocr_process')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

