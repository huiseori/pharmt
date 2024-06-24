# views.py
from django.shortcuts import render
# from .models import ChatbotModel
# import openai
# from openai.error import RateLimitError
import requests
from bs4 import BeautifulSoup
import pandas as pd
from django.shortcuts import render
from django.conf import settings
import urllib.parse
import difflib

from django.http import JsonResponse
import os
import uuid
import json
import time
import platform
import numpy as np
import cv2
import requests
from PIL import ImageFont, ImageDraw, Image


from .models import ChatbotModel
# secret_key = 'VFp4emJvZ2dlZENQRm9Pa3RmVlVhWENFRXhncGZIYWo='


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



# views.py
import requests
from django.shortcuts import render, redirect

api_url = 'https://f2njh1jvk0.apigw.ntruss.com/custom/v1/31289/20f7f9592ec261660e6a41d64f3cd240d068f9e641c7fe728341b9a2e0979ff0/general'
secret_key = 'VFp4emJvZ2dlZENQRm9Pa3RmVlVhWENFRXhncGZIYWo='


def ocr_view(request):
    return render(request, 'ocr.html')

def put_text(image, text, x, y, color=(0, 255, 0), font_size=22):
    if type(image) == np.ndarray:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_coverted)

    if platform.system() == 'Darwin':
        font = 'AppleGothic.ttf'
    elif platform.system() == 'Windows':
        font = 'malgun.ttf'

    image_font = ImageFont.truetype(font, font_size)
    draw = ImageDraw.Draw(image)
    draw.text((x, y), text, font=image_font, fill=color)

    numpy_image = np.array(image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image

def ocr_process(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        path = os.path.join('uploads', image_file.name)
        with open(path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # 파일이 제대로 업로드되었는지 확인
        if not os.path.exists(path):
            return JsonResponse({'error': 'File upload failed.'}, status=500)

        files = [('file', open(path, 'rb'))]

        request_json = {
            'images': [{'format': 'jpg', 'name': 'demo'}],
            'requestId': str(uuid.uuid4()),
            'version': 'V2',
            'timestamp': int(round(time.time() * 1000))
        }

        payload = {'message': json.dumps(request_json).encode('UTF-8')}

        headers = {
            'X-OCR-SECRET': secret_key,
        }

        response = requests.post(api_url, headers=headers, data=payload, files=files)
        result = response.json()

        img = cv2.imread(path)
        # 이미지 로드 확인
        if img is None:
            return JsonResponse({'error': 'Failed to read image file.'}, status=500)

        roi_img = img.copy()

        # 인식된 모든 텍스트 추출
        recognized_texts = []

        for field in result['images'][0]['fields']:
            text = field['inferText']
            vertices_list = field['boundingPoly']['vertices']
            pts = [tuple(vertice.values()) for vertice in vertices_list]
            topLeft = [int(_) for _ in pts[0]]
            topRight = [int(_) for _ in pts[1]]
            bottomRight = [int(_) for _ in pts[2]]
            bottomLeft = [int(_) for _ in pts[3]]

            cv2.line(roi_img, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(roi_img, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(roi_img, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(roi_img, bottomLeft, topLeft, (0, 255, 0), 2)
            roi_img = put_text(roi_img, text, topLeft[0], topLeft[1] - 10, font_size=30)

            recognized_texts.append(text)  # 모든 인식된 텍스트 추가
            print(f"Recognized Texts: {recognized_texts}")  # 디버깅: 인식된 텍스트 출력

        # Save the processed image to serve it to the frontend
        processed_image_path = os.path.join('uploads', 'processed_' + image_file.name)
        cv2.imwrite(processed_image_path, roi_img)

        # 검색 결과가 있는 텍스트 필터링
        valid_texts = []
        for text in recognized_texts:
            encoded_text = urllib.parse.quote(text)
            api_key = settings.DUR_API_KEY  # 공공데이터포털에서 발급받은 인증 키
            url = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getUsjntTabooInfoList03?serviceKey={api_key}&itemName={encoded_text}&type=xml"
            response = requests.get(url)

            print(f"API URL: {url}")  # 디버깅: 호출한 API URL 출력
            print(f"API Response Status: {response.status_code}")  # 디버깅: API 응답 코드 출력

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                if soup.find_all("item"):
                    valid_texts.append(text)  # 검색 결과가 있는 텍스트 추가

        print(f"Valid Texts: {valid_texts}")  # 디버깅: 유효한 텍스트 출력

        if valid_texts:
            # 검색 결과가 있는 첫 번째 유효 텍스트로 리다이렉트
            return redirect(f'/drug_detail?drug_name={urllib.parse.quote(valid_texts[0])}')

        # 유효한 검색 결과가 없을 경우 오류 메시지 반환
        return JsonResponse({'error': 'No valid drug information found in OCR result.'}, status=404)

    return JsonResponse({'error': 'Invalid request'}, status=400)

    # return render(request, 'ocr.html', {'processed_image_path': '/' + processed_image_path})

    #return JsonResponse({'error': 'Invalid request'}, status=400)

import difflib  # 문자열 유사도 비교를 위해 사용할 모듈


def drug_detail_view(request):
    drug_info = None
    error_message = None
    if request.method == 'GET':
        api_key = settings.DUR_API_KEY  # 공공데이터포털에서 발급받은 인코딩된 인증 키
        drug_name = request.GET.get('drug_name')  # 사용자 입력을 받음

        if drug_name:
            # URL 인코딩된 의약품 이름
            encoded_drug_name = urllib.parse.quote(drug_name)

            # 공공데이터포탈 API URL 설정
            url = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getUsjntTabooInfoList03?serviceKey={api_key}&itemName={encoded_drug_name}&type=xml"

            # API로 데이터 불러오기
            response = requests.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")

                # 데이터를 저장할 리스트 초기화
                drug_list = []

                # XML 데이터를 파싱하여 리스트에 저장
                for item in soup.find_all("item"):
                    ingr_code = item.find("ingr_code").text if item.find("ingr_code") else None
                    ingr_kor_name = item.find("ingr_kor_name").text if item.find("ingr_kor_name") else None
                    mix = item.find("mix").text if item.find("mix") else None
                    mix_type = item.find("mix_type").text if item.find("mix_type") else None
                    se_qesitm = item.find("se_qesitm").text if item.find("se_qesitm") else None
                    item_name = item.find("item_name").text if item.find("item_name") else None
                    entp_name = item.find("entp_name").text if item.find("entp_name") else None
                    item_seq = item.find("item_seq").text if item.find("item_seq") else None  # itemSeq 값 추출

                    # 검색어와 일치하는 항목만 추가
                    if drug_name.lower() in item_name.lower():
                        # nedrug.mfds.go.kr의 상세 정보 링크 생성
                        detail_link = f"https://nedrug.mfds.go.kr/pbp/CCBBB01/getItemDetail?itemSeq={item_seq}"

                        drug_list.append({
                            "ingr_code": ingr_code,
                            "ingr_kor_name": ingr_kor_name,
                            "mix": mix,
                            "mix_type": mix_type,
                            "se_qesitm": se_qesitm,
                            "item_name": item_name,
                            "entp_name": entp_name,
                            "detail_link": detail_link
                        })

                if not drug_list:
                    # 유사도 비교를 사용하여 가장 유사한 이름 찾기
                    possible_matches = [item.find("item_name").text for item in soup.find_all("item") if
                                        item.find("item_name")]
                    closest_matches = difflib.get_close_matches(drug_name, possible_matches, n=1, cutoff=0.7)

                    if closest_matches:
                        closest_match = closest_matches[0]
                        for item in soup.find_all("item"):
                            item_name = item.find("item_name").text if item.find("item_name") else None
                            if item_name and closest_match in item_name:
                                ingr_code = item.find("ingr_code").text if item.find("ingr_code") else None
                                ingr_kor_name = item.find("ingr_kor_name").text if item.find("ingr_kor_name") else None
                                mix = item.find("mix").text if item.find("mix") else None
                                mix_type = item.find("mix_type").text if item.find("mix_type") else None
                                se_qesitm = item.find("se_qesitm").text if item.find("se_qesitm") else None
                                entp_name = item.find("entp_name").text if item.find("entp_name") else None
                                item_seq = item.find("item_seq").text if item.find("item_seq") else None

                                # nedrug.mfds.go.kr의 상세 정보 링크 생성
                                detail_link = f"https://nedrug.mfds.go.kr/pbp/CCBBB01/getItemDetail?itemSeq={item_seq}"

                                drug_list.append({
                                    "ingr_code": ingr_code,
                                    "ingr_kor_name": ingr_kor_name,
                                    "mix": mix,
                                    "mix_type": mix_type,
                                    "se_qesitm": se_qesitm,
                                    "item_name": item_name,
                                    "entp_name": entp_name,
                                    "detail_link": detail_link
                                })

                    if drug_list:
                        drug_info = drug_list  # 데이터프레임 변환이 필요없습니다.
                    else:
                        error_message = f"'{drug_name}'에 대한 유사한 의약품 정보를 찾을 수 없습니다."

                else:
                    drug_info = drug_list  # 데이터프레임 변환이 필요없습니다.
            else:
                # 응답 데이터 출력
                print(f"Response Status Code: {response.status_code}")
                print(f"Response Content: {response.content}")
                error_message = f"API 요청 실패: {response.status_code} - {response.text}"

    return render(request, 'drug_detail.html', {'drug_info': drug_info, 'error_message': error_message})

# def drug_detail_view(request):
#     drug_info = None
#     error_message = None
#     if request.method == 'GET':
#         api_key = settings.DUR_API_KEY  # 공공데이터포털에서 발급받은 인코딩된 인증 키
#         drug_name = request.GET.get('drug_name')  # 사용자 입력을 받음
#
#         if drug_name:
#             # URL 인코딩된 의약품 이름
#             encoded_drug_name = urllib.parse.quote(drug_name)
#
#             # 공공데이터포탈 API URL 설정
#             url = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getUsjntTabooInfoList03?serviceKey={api_key}&itemName={encoded_drug_name}&type=xml"
#
#             # API로 데이터 불러오기
#             response = requests.get(url)
#
#             if response.status_code == 200:
#                 soup = BeautifulSoup(response.content, "html.parser")
#
#                 # 데이터를 저장할 리스트 초기화
#                 drug_list = []
#
#                 # XML 데이터를 파싱하여 리스트에 저장
#                 for item in soup.find_all("item"):
#                     ingr_code = item.find("ingr_code").text if item.find("ingr_code") else None
#                     ingr_kor_name = item.find("ingr_kor_name").text if item.find("ingr_kor_name") else None
#                     mix = item.find("mix").text if item.find("mix") else None
#                     mix_type = item.find("mix_type").text if item.find("mix_type") else None
#                     se_qesitm = item.find("se_qesitm").text if item.find("se_qesitm") else None
#                     item_name = item.find("item_name").text if item.find("item_name") else None
#                     entp_name = item.find("entp_name").text if item.find("entp_name") else None
#                     #item_seq = item.find("item_seq").text if item.find("item_seq") else None  # itemSeq 값 추출
#
#                     # 검색어와 일치하는 항목만 추가
#                     if drug_name in item_name:
#                         drug_list.append([ingr_code, ingr_kor_name, mix, mix_type, se_qesitm, item_name, entp_name])
#
#                 if drug_list:
#                     drug_info = pd.DataFrame(drug_list, columns=["ingr_code", "ingr_kor_name", "mix", "mix_type", "se_qesitm", "item_name", "entp_name"])
#                     drug_info = drug_info.to_dict(orient='records')  # DataFrame을 리스트의 딕셔너리로 변환
#                 else:
#                     error_message = f"'{drug_name}'에 대한 의약품 정보를 찾을 수 없습니다."
#             else:
#                 # 응답 데이터 출력
#                 print(f"Response Status Code: {response.status_code}")
#                 print(f"Response Content: {response.content}")
#                 error_message = f"API 요청 실패: {response.status_code} - {response.text}"
#
#     return render(request, 'drug_detail.html', {'drug_info': drug_info, 'error_message': error_message})

def get_articles(page=1):
    url = f"https://www.kpanews.co.kr/article/list.asp?page={page}"
    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    articles = []

    for item in soup.select('.lst_article1 ul li'):
        title = item.select_one('.subj').text.strip()
        summary = item.select_one('.t1').text.strip() if item.select_one('.t1') else ''
        date = item.select_one('.botm span').text.strip()
        author = item.select('.botm span')[1].text.strip() if len(item.select('.botm span')) > 1 else ''
        link = "https://www.kpanews.co.kr/" + item.a['href']
        articles.append({
            'title': title,
            'summary': summary,
            'date': date,
            'author': author,
            'link': link
        })

    return articles

def news_view(request):
    page = request.GET.get('page', 1)
    articles = get_articles(page)
    return render(request, 'news.html', {'articles': articles})



