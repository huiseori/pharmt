# urls.py
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from pharmtracker import settings
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.splash_view, name='home'),
    path('home/', views.home_view, name='home'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('ocr/', views.ocr_view, name='ocr'),
    path('news/', views.news_view, name='news'),
    path('ocr_process/', views.ocr_process, name='ocr_process'),
    path('drug_list/', views.drug_list_view, name='drug_list'),
    path('drug_detail/<str:item_seq>/', views.drug_detail_view, name='drug_detail'),
    # path('drug_detail/', views.drug_detail_view, name='drug_detail'),
    path('news/summary/<int:article_id>/', views.news_summary_view, name='news_summary'),  # 요약 상세 페이지 URL 패턴 추가
    path('mypage/', views.mypage_view, name='mypage'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    path('pharmacy_list/', views.pharmacy_list_view, name='pharmacy_list'),
path('drug_interaction/', views.drug_interaction_view, name='drug_interaction'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

