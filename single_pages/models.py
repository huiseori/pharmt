# Create your models here.

import openai
from django.conf import settings
import os
from django.db import models


openai.api_key = settings.OPENAI_API_KEY  # OpenAI API 키 설정

class ChatbotModel:
    def __init__(self):
        self.model_engine = "text-davinci-002"
        self.max_tokens = 2048
        openai.api_key = 'v1RI90KCDrEbFktChe7AJrqaQZjthMNnpk5aWRwbuLjTYADawSEBxDlfJu5LBctjCRYs%2BfB4MR0eLuBrIdQB4Q%3D%3D'

    def generate_response(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model_engine,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=0.7,
        )


        return response.choices[0].message['content'].strip()


class DrugModel(models.Model):
   name = models.CharField(max_length=255)
   image_url = models.URLField()
   ingredients = models.TextField()
   indications = models.TextField()
   dosage = models.TextField()
   precautions = models.TextField()

   def __str__(self):
       return self.name


class Medicine(models.Model):
    # 핵심 식별자
    item_seq = models.CharField(max_length=50, unique=True, primary_key=True, verbose_name="품목기준코드")
    item_name = models.CharField(max_length=200, verbose_name="품목명")
    entp_name = models.CharField(max_length=100, verbose_name="업체명", null=True, blank=True)
    
    # 상세 정보 (RAG 및 상세화면용)
    efficacy = models.TextField(verbose_name="효능효과", null=True, blank=True)  # EE_DOC_DATA
    usage_dosage = models.TextField(verbose_name="용법용량", null=True, blank=True) # UD_DOC_DATA
    precautions = models.TextField(verbose_name="주의사항", null=True, blank=True) # NB_DOC_DATA
    
    # 추가 메타 데이터
    image_url = models.URLField(verbose_name="이미지 URL", null=True, blank=True)
    
    # 검색 최적화를 위한 인덱스 설정
    class Meta:
        indexes = [
            models.Index(fields=['item_name']), # 이름 검색 속도 향상
        ]
        verbose_name = "의약품"
        verbose_name_plural = "의약품 목록"

    def __str__(self):
        return self.item_name
