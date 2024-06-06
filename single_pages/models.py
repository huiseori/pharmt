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
