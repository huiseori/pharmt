# Create your models here.

import openai
from django.conf import settings

openai.api_key = settings.OPENAI_API_KEY  # OpenAI API 키 설정

class ChatbotModel:
    def __init__(self):
        self.model_engine = "text-davinci-002"
        self.max_tokens = 2048

    def generate_response(self, prompt):
        """
        OpenAI GPT-3 API를 사용하여 사용자 입력에 대한 응답을 생성합니다.
        """
        response = openai.Completion.create(
            engine=self.model_engine,
            prompt=prompt,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].text.strip()

