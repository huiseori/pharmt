# views.py
from django.shortcuts import render
from .models import ChatbotModel

def home_view(request):
    return render(request, 'home.html')

def chatbot_view(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        chatbot = ChatbotModel()
        response = chatbot.generate_response(user_input)
        context = {'response': response}
        return render(request, 'chatbot.html', context)
    else:
        context = {'response': ''}
        return render(request, 'chatbot.html', context)


# views.py
import requests
from django.contrib import messages
from django.conf import settings
from django.shortcuts import render, redirect



def ocr_view(request):
    if request.method == 'POST':
        image = request.FILES.get('image')

        # Naver Clova OCR API 호출
        ocr_url = 'https://naveropenapi.apigw.ntruss.com/vision-ocr/v1/recognize'
        headers = {
            'X-NCP-APIGW-API-KEY-ID': settings.NAVER_OCR_API_KEY,
            'X-NCP-APIGW-API-KEY': settings.NAVER_OCR_API_KEY,
        }
        files = {'image': image.read()}
        ocr_response = requests.post(ocr_url, headers=headers, files=files)

        if ocr_response.status_code == 200:
            ocr_text = ocr_response.json()['images'][0]['fields'][0]['inferText']

            # 검색 데이터 생성
            search_query = ocr_text

            # 검색 결과를 세션에 저장
            request.session['search_query'] = search_query

            return render(request, 'ocr_result.html', {'ocr_text': ocr_text})
        else:
            messages.error(request, 'OCR 처리 중 오류가 발생했습니다.')

    return render(request, 'ocr.html')


def detail_view(request):
    if request.method == 'POST':
        search_query = request.session.get('search_query')

        if search_query:
            # 공공데이터 API 검색
            public_data_url = 'http://apis.data.go.kr/1471000/DURPrdlstInfoService03'
            params = {
                'query': search_query,
                'serviceKey': settings.PUBLIC_DATA_API_KEY,
            }
            response = requests.get(public_data_url, params=params)

            if response.status_code == 200:
                data = response.json()
                # 데이터 처리 로직

                return render(request, 'drug_detail.html', {'drug_data': data})
            else:
                messages.error(request, '공공데이터 API 호출 중 오류가 발생했습니다.')
        else:
            messages.error(request, '검색 쿼리가 없습니다.')

    return render(request, 'ocr_result.html')



