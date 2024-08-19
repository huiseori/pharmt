# views.py
from datetime import date

from django.shortcuts import render
# from .models import ChatbotModel
# from openai.error import RateLimitError
import requests
from bs4 import BeautifulSoup
import pandas as pd
from django.shortcuts import render
from django.conf import settings
import urllib.parse
import difflib # 문자열 유사도 비교를 위해 사용할 모듈
import logging
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
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
import openai  # Assuming you're using OpenAI's GPT-3/4 as the chatbot backend
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from langchain.chains import conversation
from langchain.chains import conversation
from django.contrib import messages
from django.conf import settings
from django.shortcuts import render, redirect
import requests
from django.shortcuts import render, redirect
from langchain_community.tools import authenticate
from django.views.decorators.http import require_http_methods
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import xml.etree.ElementTree as ET
from pharmtracker import settings
from xml.etree import ElementTree
from typing import Optional, List, Tuple
from konlpy.tag import Okt
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
import aiohttp
from django.utils.html import escape
from urllib.parse import quote




from .models import ChatbotModel
# secret_key = 'VFp4emJvZ2dlZENQRm9Pa3RmVlVhWENFRXhncGZIYWo='

@login_required
def mypage_view(request):
    user = request.user

    if request.method == 'POST':
        if 'gender' in request.POST and 'age' in request.POST:
            gender = request.POST.get('gender')
            age = request.POST.get('age')
            user.first_name = gender
            user.last_name = age
            user.save()
            messages.success(request, '프로필이 업데이트되었습니다.')

        if 'blood_pressure' in request.POST and 'blood_sugar' in request.POST and 'weight' in request.POST:
            blood_pressure = request.POST.get('blood_pressure')
            blood_sugar = request.POST.get('blood_sugar')
            weight = request.POST.get('weight')
            today = date.today().strftime('%Y-%m-%d')

            health_record = {
                'blood_pressure': blood_pressure,
                'blood_sugar': blood_sugar,
                'weight': weight,
            }

            # 세션 데이터 초기화 시 딕셔너리로 설정
            if 'health_records' not in request.session:
                request.session['health_records'] = {}

            if not isinstance(request.session['health_records'], dict):
                request.session['health_records'] = {}

            if today not in request.session['health_records']:
                request.session['health_records'][today] = []

            request.session['health_records'][today].append(health_record)
            request.session.modified = True

            messages.success(request, '오늘의 건강 기록이 저장되었습니다.')

        return redirect('mypage')

    age_range = range(1, 101)
    health_records = request.session.get('health_records', {})
    context = {
        'age_range': age_range,
        'user': user,
        'health_records': health_records,
    }
    return render(request, 'mypage.html', context)


def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if User.objects.filter(username=username).exists():
            error_message = "이미 사용 중인 아이디입니다."
            return render(request, 'register.html', {'error_message': error_message})

        user = User.objects.create_user(username=username, password=password)
        user = authenticate(request, username=username, password=password)
        login(request, user)
        return redirect('login')  # 회원가입 후 로그인 페이지로 리디렉션

    return render(request, 'register.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # 로그인 후 홈으로 리디렉션
        else:
            error_message = "아이디 또는 비밀번호가 잘못되었습니다."
            return render(request, 'login.html', {'error_message': error_message})
    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect('home')



def home_view(request):
    return render(request, 'home.html')

OPENAI_API_KEY = settings.OPENAI_API_KEY
# DUR 품목정보 API 설정
DUR_API_KEY = settings.DUR_API_KEY
# RAG 설정
llm = ChatOpenAI(temperature=0.1, openai_api_key=OPENAI_API_KEY)
cache_dir = LocalFileStore("./.cache/practice/")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

# # FAISS vector store 설정
# BASE_DIR = Path(__file__).resolve().parent.parent
# VECTORSTORE_DIR = BASE_DIR / 'vectorstore'
# VECTORSTORE_PATH = VECTORSTORE_DIR / 'faiss_index'

# Initialize the Okt tokenizer
okt = Okt()

# 전역 변수로 vectorstore 선언
vectorstore = None

# RAG 설정
llm = ChatOpenAI(temperature=0.1, openai_api_key=OPENAI_API_KEY)
cache_dir = LocalFileStore("./.cache/practice/")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

def initialize_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = FAISS.from_texts(["초기화 문서"], cached_embeddings)

# 앱 시작 시 호출
initialize_vectorstore()

# 전역 변수로 의약품 목록 선언
medicine_list = []


def fetch_medicine_list() -> List[str]:
    """
    API를 호출하여 의약품 목록 중 1000개를 가져와 리스트로 반환합니다.
    """
    num_of_rows = 100  # 한 페이지에 가져올 데이터 수
    page_no = 1  # 시작 페이지
    max_pages = 10  # 최대 페이지 수 (1000개 수집 목표)
    medicines = []

    while page_no <= max_pages:
        url = (
            f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/"
            f"getDrugPrdtPrmsnInq05?serviceKey={DUR_API_KEY}&pageNo={page_no}&numOfRows={num_of_rows}&type=xml"
        )
        response = requests.get(url)

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            items = root.findall('.//item')

            for item in items:
                item_name = item.find('ITEM_NAME').text
                if item_name:
                    medicines.append(item_name)

            # print(f"페이지 {page_no}에서 {len(items)}개의 의약품을 가져왔습니다.")

            # 목표 수집량인 1000개에 도달하면 중단
            if len(medicines) >= 1000:
                medicines = medicines[:1000]  # 정확히 1000개로 자르기
                break

            page_no += 1  # 다음 페이지로 이동
        else:
            print(f"API 호출 오류 발생: {response.status_code}")
            break

    return medicines

def load_pdf_chunks(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    # PDF 문서를 로드합니다.
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 청크로 나누기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(pages)
    return chunks

def add_pdf_to_vectorstore(pdf_path: str, vectorstore, OPENAI_API_KEY: str):
    # PDF 내용을 청크로 로드합니다.
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 텍스트 분할기를 사용하여 문서를 청크로 나눔
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,  # 청크의 크기 설정
        chunk_overlap=200  # 청크 간 중복 설정
    )
    split_docs = text_splitter.split_documents(documents)

    # 문서에서 텍스트를 추출
    texts = [doc.page_content for doc in split_docs]

    # 문서를 벡터 저장소에 추가
    vectorstore.add_texts(texts)
    print(f"벡터 저장소에 {len(texts)}개의 텍스트 청크가 추가되었습니다.")

def initialize_data():
    global vectorstore

    # 벡터 저장소가 초기화되지 않았으면 초기화
    if vectorstore is None:
        vectorstore = FAISS.from_texts(["초기화 문서"], cached_embeddings)

    # PDF 데이터를 벡터 저장소에 추가
    pdf_path = r"C:\Users\se711\PycharmProjects\graduation-project\uploads\test_1.pdf"  # Raw String 사용
    add_pdf_to_vectorstore(pdf_path, vectorstore, OPENAI_API_KEY)

# 앱 시작 시 초기화 호출
initialize_data()

def initialize_medicine_list():
    """
    의약품 목록을 초기화합니다.
    """
    global medicine_list
    if not medicine_list:  # 초기화되지 않은 경우에만 수행
        medicine_list = fetch_medicine_list()
        print(f"의약품 목록 초기화 완료: {len(medicine_list)}개 항목")

# 앱 시작 시 의약품 목록 초기화
initialize_medicine_list()

def extract_medicine_name_from_question(question: str) -> Optional[str]:
    """
    질문에서 의약품 이름을 추출하여 반환합니다.
    """
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            당신은 의약품 이름을 식별할 수 있는 AI 모델입니다. 
            아래의 질문에서 의약품 이름을 정확히 추출해 주세요. 
            만약 질문에 의약품 이름이 없다면 'None'을 반환하세요.
            """),
        ("human", question)
    ])

    # Run the LLM chain to get the extracted name
    extraction_chain = extraction_prompt | llm
    result = extraction_chain.invoke({"question": question})
    extracted_name = result.content.strip()

    # If the LLM's result is 'None' or empty, we assume no medicine name was found
    if extracted_name.lower() in ['none', '']:
        return None

    return extracted_name

    # tokens = okt.morphs(question)
    # # print(f"질문 토큰: {tokens}")  # 디버깅을 위해 출력
    #
    # # 질문의 각 토큰을 의약품 목록과 비교
    # for token in tokens:
    #     if token in medicine_list:
    #         print(f"매칭된 의약품 이름: {token}")  # 디버깅을 위해 출력
    #         return token

    print("의약품 이름을 찾지 못했습니다.")  # 디버깅을 위해 출력
    return None

def preprocess_dur_data(xml_data):
    # Parse the XML data
    root = ET.fromstring(xml_data)
    items = root.findall('.//item')
    processed_data = []

    # print(f"API 응답에서 {len(items)}개의 아이템을 찾았습니다.")

    for item in items:
        processed_item = f"품목기준코드: {item.find('ITEM_SEQ').text if item.find('ITEM_SEQ') is not None else '정보 없음'}\n"
        processed_item += f"품목명: {item.find('ITEM_NAME').text if item.find('ITEM_NAME') is not None else '정보 없음'}\n"
        processed_item += f"제조사: {item.find('ENTP_NAME').text if item.find('ENTP_NAME') is not None else '정보 없음'}\n"
        processed_item += f"업종: {item.find('INDUTY_TYPE').text if item.find('INDUTY_TYPE') is not None else '정보 없음'}\n"
        processed_item += f"제형: {item.find('CHART').text if item.find('CHART') is not None else '정보 없음'}\n"
        processed_item += f"보관 방법: {item.find('STORAGE_METHOD').text if item.find('STORAGE_METHOD') is not None else '정보 없음'}\n"
        processed_item += f"유효 기간: {item.find('VALID_TERM').text if item.find('VALID_TERM') is not None else '정보 없음'}\n"
        processed_item += f"포장 단위: {item.find('PACK_UNIT').text if item.find('PACK_UNIT') is not None else '정보 없음'}\n"
        processed_item += f"총 내용량: {item.find('TOTAL_CONTENT').text if item.find('TOTAL_CONTENT') is not None else '정보 없음'}\n"
        # 효능 문서 데이터 추출
        ee_doc_data = item.find('EE_DOC_DATA/DOC/SECTION/ARTICLE/PARAGRAPH')
        ee_doc_text = ee_doc_data.text if ee_doc_data is not None else '정보 없음'
        processed_item += f"효능 문서 데이터: {ee_doc_text}\n"
        # 사용 방법 문서 데이터 추출
        ud_doc_data = item.find('UD_DOC_DATA/DOC/SECTION/ARTICLE/PARAGRAPH')
        ud_doc_text = ud_doc_data.text if ud_doc_data is not None else '정보 없음'
        processed_item += f"사용 방법 문서 데이터: {ud_doc_text}\n"
        # 사용상 주의사항 문서 데이터 추출
        nb_doc_data = item.find('NB_DOC_DATA/DOC/SECTION/ARTICLE/PARAGRAPH')
        nb_doc_text = nb_doc_data.text if nb_doc_data is not None else '정보 없음'
        processed_item += f"주의 사항 문서 데이터: {nb_doc_text}\n"
        processed_item += f"주요 성분: {item.find('MAIN_ITEM_INGR').text if item.find('MAIN_ITEM_INGR') is not None else '정보 없음'}\n"
        processed_data.append(processed_item)

    return processed_data

def add_documents_to_vectorstore(documents):
    """
    벡터 저장소에 문서를 추가합니다.
    """
    global vectorstore
    vectorstore.add_texts(documents)
    print(f"벡터 저장소에 {len(documents)}개의 문서가 추가되었습니다.")
    print(f"현재 벡터 저장소의 크기: {vectorstore.index.ntotal}")

async def fetch_data(session, url):
    """
    주어진 URL에서 비동기적으로 데이터를 가져옵니다.
    """
    async with session.get(url) as response:
        return await response.text()

async def load_initial_data_async():
    """
    초기 데이터를 비동기적으로 로드하여 벡터 저장소에 추가합니다.
    """
    encoded_drug_name = urllib.parse.quote("")  # 모든 품목 가져오기
    num_of_rows = 100  # 한 페이지에 100개의 결과
    page_no = 1  # 첫 번째 페이지
    max_pages = 100  # 최대 10페이지
    all_processed_data = []

    async with aiohttp.ClientSession() as session:
        tasks = []

        while page_no <= max_pages:
            dur_api_url = (
                f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnDtlInq04?serviceKey={DUR_API_KEY}&pageNo={page_no}&numOfRows={num_of_rows}&type=xml&item_name={encoded_drug_name}")

            tasks.append(fetch_data(session, dur_api_url))
            page_no += 1

        responses = await asyncio.gather(*tasks)

        for xml_data in responses:
            processed_data = preprocess_dur_data(xml_data)
            if not processed_data:
                break  # 더 이상 데이터가 없으면 반복 종료

            all_processed_data.extend(processed_data)

    print(f"전처리된 전체 데이터 수: {len(all_processed_data)}")
    add_documents_to_vectorstore(all_processed_data)

def load_initial_data():
    """
    비동기 초기 데이터 로드 함수를 실행합니다.
    """
    asyncio.run(load_initial_data_async())

# 앱 시작 시 초기 데이터 로드
load_initial_data()

def retrieve_relevant_context(question: str) -> Tuple[str, Optional[str]]:
    """
    질문에서 약물 이름을 추출하고, 벡터 저장소에서 관련 문맥을 검색합니다.
    """
    medicine_name = extract_medicine_name_from_question(question)

    if not medicine_name:
        print("의약품 이름을 추출할 수 없습니다.")
        return "의약품 이름을 질문에서 파악할 수 없습니다. 다시 시도해 주세요.", None

    print(f"추출된 의약품 이름: {medicine_name}")

    # docs_medicine = vectorstore.similarity_search(medicine_name, k=3)
    #print(f"'{medicine_name}'에 대해 검색된 문서 수: {len(docs)}")
    docs_medicine = []
    if medicine_name:
        docs_medicine = vectorstore.similarity_search(medicine_name, k=3)
        print(f"'{medicine_name}'에 대해 검색된 문서 수: {len(docs_medicine)}")

    # PDF 데이터와 관련된 문서 검색
    docs_pdf = vectorstore.similarity_search(question, k=2)
    print(f"질문에 대해 검색된 PDF 문서 수: {len(docs_pdf)}")

    # # PDF 문서만으로 문맥을 생성하여 테스트
    # if docs_pdf:
    #     context = "\n".join([doc.page_content for doc in docs_pdf])
    #     print(f"PDF에서 생성된 문맥:\n{context[:500]}...")
    #     return context, None  # 임시로 약물 이름 없이 테스트

    # 두 검색 결과를 결합
    docs = docs_medicine + docs_pdf
    if not docs:
        print("관련 문서를 찾을 수 없습니다.")
        return "관련 문서를 찾을 수 없습니다.", None

    # context = "\n".join([doc.page_content for doc in docs])
    # print(f"결합된 문맥:\n{context[:500]}...")
    # return context, None

    top_item_name = None
    for doc in docs_medicine:
        lines = doc.page_content.splitlines()
        for line in lines:
            if line.startswith("품목명:"):
                top_item_name = line.split(":")[1].strip()
                break

    context = "\n".join([doc.page_content for doc in docs])
    return context, top_item_name

def add_button_to_response(result_content: str, drug_name: Optional[str]) -> str:
    """
    결과 응답에 클릭 가능한 버튼을 추가하여 상세 약물 정보를 제공하는 링크를 포함합니다.
    """
    if not drug_name:
        return result_content  # 약물 이름이 없으면 버튼을 추가하지 않음

    encoded_drug_name = quote(drug_name)
    detail_link = f"/drug_list/?drug_name={encoded_drug_name}"

    escaped_drug_name = escape(drug_name)

    button_html = (
        f'<a href="{detail_link}" target="_blank" '
        f'style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; '
        f'color: white; text-align: center; text-decoration: none; font-size: 16px; '
        f'border-radius: 5px;">{escaped_drug_name} 상세 정보</a>'
    )

    return result_content + "\n\n" + button_html

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 중장년층과 노인층을 위한 친절하고 이해하기 쉬운 의약품 정보 제공 도우미입니다. 제공된 의약품 정보를 바탕으로 다음 지침을 따라주세요:
    1. 개조식으로 설명해 주세요.
    2. 여러번 생각하고 정확한 정보를 제공해주세요.
    3. 정보를 제공할 때는 천천히, 명확하게, 그리고 단계적으로 설명해 주세요.
    4. 질문의 맥락을 고려하여, 관련된 추가 정보나 조언을 제공하세요.
    5. 약물 복용과 관련된 주의사항을 강조하고, 의사나 약사와 상담할 것을 권장하세요.
    6. 건강에 도움이 되는 일반적인 조언(예: 규칙적인 운동, 균형 잡힌 식단)도 함께 제공하세요.
    7. 필요한 경우 반복해서 설명하고, 이해했는지 확인하는 질문을 해 주세요.
    8. 답변은 간결하게 5문장 이하로 답변해 주세요.
    9. 필요한 경우 의약품 성분을 포함해서 답변해 주세요.
    10.항상 존댓말을 사용하고, 따뜻하고 공감적인 톤으로 대화하세요.
    아래의 정보를 바탕으로 질문에 정확하고 이해하기 쉽게 답변해 주세요:

    {context}
    """),
    ("human", "{question}")
])

rag_chain = (
    {"context": lambda x: retrieve_relevant_context(x["question"])[0], "question": RunnablePassthrough()}
    | prompt
    | llm
)

def generate_independent_response(question):
    prompt = f"질문에 대해 자세하고 유익한 답변을 생성해 주세요:\n\n{question}"
    response = llm.generate(prompt, max_tokens=150)
    return response.content


@csrf_exempt
@require_http_methods(["GET", "POST"])
def chatbot_view(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
            question = body.get("question")
            if not question:
                return JsonResponse({"error": "질문이 필요합니다."}, status=400)

            context, extracted_drug_name = retrieve_relevant_context(question)
            # print(f"검색된 컨텍스트: {context[:1000]}...")  # 디버깅을 위한 컨텍스트 출력
            if "관련 문서를 찾을 수 없습니다." in context:
                # Context가 충분하지 않을 때 독립적인 답변 생성
                independent_response = generate_independent_response(question)
                return JsonResponse({"answer": independent_response})

            # 기존의 RAG 로직을 사용하여 답변 생성
            result = rag_chain.invoke({"question": question, "context": context})


            # 링크 버튼 추가
            if extracted_drug_name:
                result.content = add_button_to_response(result.content, extracted_drug_name)
                print(f"Added button for drug name: {extracted_drug_name}")

            print(f"생성된 답변: {result.content}")
            return JsonResponse({"answer": result.content})
            print(f"현재 벡터 저장소의 크기: {vectorstore.index.ntotal}")

        except json.JSONDecodeError:
            return JsonResponse({"error": "잘못된 JSON 형식입니다."}, status=400)

    elif request.method == "GET":
        return render(request, "chatbot.html")







# api_url = 'https://f2njh1jvk0.apigw.ntruss.com/custom/v1/31289/20f7f9592ec261660e6a41d64f3cd240d068f9e641c7fe728341b9a2e0979ff0/general'
# secret_key = 'VFp4emJvZ2dlZENQRm9Pa3RmVlVhWENFRXhncGZIYWo='

api_url = 'https://bi9fchpejl.apigw.ntruss.com/custom/v1/31331/3f422bc603539f1d16208ebac83f049343eccc4d8c53e11052a0ce0e1ce01490/general'
secret_key = 'VWdFWmp3SkVYdkRNdFJTWFRQdlRXQ3hiYlVoQmNlWFo='

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
            # print(f"Recognized Texts: {recognized_texts}")  # 디버깅: 인식된 텍스트 출력

        # Save the processed image to serve it to the frontend
        processed_image_path = os.path.join('uploads', 'processed_' + image_file.name)
        cv2.imwrite(processed_image_path, roi_img)

        # 검색 결과가 있는 텍스트 필터링
        valid_texts = []
        for text in recognized_texts:
            encoded_text = urllib.parse.quote(text)
            api_key = settings.DUR_API_KEY  # 공공데이터포털에서 발급받은 인증 키
            # url = f"http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getUsjntTabooInfoList03?serviceKey={api_key}&itemName={encoded_text}&type=xml"
            url = f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnDtlInq04?serviceKey={api_key}&type=xml&item_name={encoded_text}"
            response = requests.get(url)

            # print(f"API URL: {url}")  # 디버깅: 호출한 API URL 출력
            # print(f"API Response Status: {response.status_code}")  # 디버깅: API 응답 코드 출력

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                if soup.find_all("item"):
                    valid_texts.append(text)  # 검색 결과가 있는 텍스트 추가

        print(f"Valid Texts: {valid_texts}")  # 디버깅: 유효한 텍스트 출력

        if valid_texts:
            # 검색 결과가 있는 첫 번째 유효 텍스트로 리다이렉트
            return redirect(f'/drug_list?drug_name={urllib.parse.quote(valid_texts[0])}')

        # 유효한 검색 결과가 없을 경우 오류 메시지 반환
        return JsonResponse({'error': 'No valid drug information found in OCR result.'}, status=404)

    return JsonResponse({'error': 'Invalid request'}, status=400)

    # return render(request, 'ocr.html', {'processed_image_path': '/' + processed_image_path})

    #return JsonResponse({'error': 'Invalid request'}, status=400)

def drug_list_view(request):
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        # AJAX 요청 처리 (자동완성)
        drug_name = request.GET.get('term', '')
        drugs = get_drug_info(drug_name, settings.DUR_API_KEY)
        suggestions = [drug['ITEM_NAME'] for drug in drugs] if drugs else []
        return JsonResponse(suggestions, safe=False)

    drug_info = None
    error_message = None

    if request.method == 'GET':
        api_key = settings.DUR_API_KEY  # 공공데이터포털에서 발급받은 인코딩된 인증 키
        drug_name = request.GET.get('drug_name')  # 사용자 입력을 받음

        if drug_name:
            encoded_drug_name = urllib.parse.quote(drug_name)
            url = f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnInq05?serviceKey={api_key}&pageNo=1&numOfRows=10&type=xml&item_name={encoded_drug_name}"
            response = requests.get(url, verify=False)  # SSL 검증 비활성화

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "lxml-xml")

                drug_list = []

                for item in soup.find_all("item"):
                    item_name = item.find("ITEM_NAME").text if item.find("ITEM_NAME") else ""
                    print(f"Found item: {item_name}")  # 디버깅용 출력
                    if drug_name.lower() in item_name.lower():
                        entp_name = item.find("ENTP_NAME").text if item.find("ENTP_NAME") else None
                        spclty_pblc = item.find("SPCLTY_PBLC").text if item.find("SPCLTY_PBLC") else None
                        prduct_type = item.find("PRDUCT_TYPE").text if item.find("PRDUCT_TYPE") else None
                        item_ingr_name = item.find("ITEM_INGR_NAME").text if item.find("ITEM_INGR_NAME") else None
                        item_seq = item.find("ITEM_SEQ").text if item.find("ITEM_SEQ") else None

                        detail_link = f"/drug_detail/{item_seq}/"  # 새로운 상세 정보 페이지로의 링크
                        image_url = get_drug_image(item_name)  # 약품 이미지 URL 가져오기
                        # print(f"Image URL for {item_name}: {image_url}")  # 디버깅용 출력

                        drug_list.append({
                            "item_name": item_name,
                            "entp_name": entp_name,
                            "spclty_pblc": spclty_pblc,
                            "prduct_type": prduct_type,
                            "item_ingr_name": item_ingr_name,
                            "detail_link": detail_link,
                            "image_url": image_url  # 약품 이미지 URL 추가
                        })

                if not drug_list:
                    error_message = f"'{drug_name}'에 대한 의약품 정보를 찾을 수 없습니다."
                else:
                    drug_info = drug_list
            else:
                error_message = f"API 요청 실패: {response.status_code} - {response.text}"

    return render(request, 'drug_list.html', {'drug_info': drug_info, 'error_message': error_message})

def get_drug_image(drug_name):
    api_key = 'v1RI90KCDrEbFktChe7AJrqaQZjthMNnpk5aWRwbuLjTYADawSEBxDlfJu5LBctjCRYs%2BfB4MR0eLuBrIdQB4Q%3D%3D'  # 공공데이터포털에서 발급받은 인코딩된 인증 키
    encoded_drug_name = urllib.parse.quote(drug_name)
    url = f"http://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService01/getMdcinGrnIdntfcInfoList01?serviceKey={api_key}&item_name={encoded_drug_name}&pageNo=1&numOfRows=3&type=xml"
    response = requests.get(url, verify=False)  # SSL 검증 비활성화

    if response.status_code == 200:
        # print(f"Image API Response: {response.content}")  # 디버깅용 출력
        soup = BeautifulSoup(response.content, "lxml-xml")
        item = soup.find("item")
        if item:
            image_url = item.find("ITEM_IMAGE").text if item.find("ITEM_IMAGE") else None
            # print(f"Extracted Image URL: {image_url}")  # 디버깅용 출력
            return image_url
    return None

def drug_detail_view(request, item_seq):
    api_key = settings.DUR_API_KEY  # 공공데이터포털에서 발급받은 인코딩된 인증 키
    encoded_item_seq = urllib.parse.quote(item_seq)
    url = f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnDtlInq04?serviceKey={api_key}&item_seq={encoded_item_seq}&type=xml"
    response = requests.get(url, verify=False)  # SSL 검증 비활성화

    # print(f"Detail API Response: {response.content}")  # API 응답 확인용

    drug_detail = {}
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "lxml-xml")
        item = soup.find("item")
        if item:
            drug_detail = {
                "item_name": item.find("ITEM_NAME").text if item.find("ITEM_NAME") else None,
                "entp_name": item.find("ENTP_NAME").text if item.find("ENTP_NAME") else None,
                "efficacy": item.find("EE_DOC_DATA").text if item.find("EE_DOC_DATA") else None,
                "usage_dosage": item.find("UD_DOC_DATA").text if item.find("UD_DOC_DATA") else None,
                "precautions": item.find("NB_DOC_DATA").text if item.find("NB_DOC_DATA") else None,
                "image_url": get_drug_image(item.find("ITEM_NAME").text) if item.find("ITEM_NAME") else None
            }
    else:
        drug_detail = None

    return render(request, 'drug_detail.html', {'drug_detail': drug_detail})

def get_drug_info(drug_name, api_key):
    print(f"Fetching drug info for: {drug_name}")
    encoded_drug_name = urllib.parse.quote(drug_name)
    url = f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnInq05?serviceKey={api_key}&pageNo=1&numOfRows=10&type=xml&item_name={encoded_drug_name}"
    response = requests.get(url, verify=False)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "lxml-xml")
        items = soup.find_all("item")
        drugs = []
        for item in items:
            drug = {
                'ITEM_NAME': item.find("ITEM_NAME").text if item.find("ITEM_NAME") else "",
                'ITEM_SEQ': item.find("ITEM_SEQ").text if item.find("ITEM_SEQ") else "",
                'ENTP_NAME': item.find("ENTP_NAME").text if item.find("ENTP_NAME") else "",
                'SPCLTY_PBLC': item.find("SPCLTY_PBLC").text if item.find("SPCLTY_PBLC") else "",
                'PRDUCT_TYPE': item.find("PRDUCT_TYPE").text if item.find("PRDUCT_TYPE") else "",
                'ITEM_INGR_NAME': item.find("ITEM_INGR_NAME").text if item.find("ITEM_INGR_NAME") else ""
            }
            print(f"Drug found: {drug}")
            drugs.append(drug)
        return drugs
    else:
        print("Failed to fetch drug info.")
        return None


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
#             url = f"http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService05/getDrugPrdtPrmsnInq05?serviceKey={api_key}&pageNo=1&numOfRows=10&type=xml&item_name={encoded_drug_name}"
#             logging.info(f"Fetching detail data from URL: {url}")
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
#                     item_seq = item.find("item_seq").text if item.find("item_seq") else None  # itemSeq 값 추출
#
#                     # 검색어와 일치하는 항목만 추가
#                     if drug_name.lower() in item_name.lower():
#                         # nedrug.mfds.go.kr의 상세 정보 링크 생성
#                         detail_link = f"https://nedrug.mfds.go.kr/pbp/CCBBB01/getItemDetail?itemSeq={item_seq}"
#
#                         drug_list.append({
#                             "ingr_code": ingr_code,
#                             "ingr_kor_name": ingr_kor_name,
#                             "mix": mix,
#                             "mix_type": mix_type,
#                             "se_qesitm": se_qesitm,
#                             "item_name": item_name,
#                             "entp_name": entp_name,
#                             "detail_link": detail_link
#                         })
#
#                 if not drug_list:
#                     # 유사도 비교를 사용하여 가장 유사한 이름 찾기
#                     possible_matches = [item.find("item_name").text for item in soup.find_all("item") if
#                                         item.find("item_name")]
#                     closest_matches = difflib.get_close_matches(drug_name, possible_matches, n=1, cutoff=0.7)
#
#                     if closest_matches:
#                         closest_match = closest_matches[0]
#                         for item in soup.find_all("item"):
#                             item_name = item.find("item_name").text if item.find("item_name") else None
#                             if item_name and closest_match in item_name:
#                                 ingr_code = item.find("ingr_code").text if item.find("ingr_code") else None
#                                 ingr_kor_name = item.find("ingr_kor_name").text if item.find("ingr_kor_name") else None
#                                 mix = item.find("mix").text if item.find("mix") else None
#                                 mix_type = item.find("mix_type").text if item.find("mix_type") else None
#                                 se_qesitm = item.find("se_qesitm").text if item.find("se_qesitm") else None
#                                 entp_name = item.find("entp_name").text if item.find("entp_name") else None
#                                 item_seq = item.find("item_seq").text if item.find("item_seq") else None
#
#                                 # nedrug.mfds.go.kr의 상세 정보 링크 생성
#                                 detail_link = f"https://nedrug.mfds.go.kr/pbp/CCBBB01/getItemDetail?itemSeq={item_seq}"
#
#                                 drug_list.append({
#                                     "ingr_code": ingr_code,
#                                     "ingr_kor_name": ingr_kor_name,
#                                     "mix": mix,
#                                     "mix_type": mix_type,
#                                     "se_qesitm": se_qesitm,
#                                     "item_name": item_name,
#                                     "entp_name": entp_name,
#                                     "detail_link": detail_link
#                                 })
#
#                     if drug_list:
#                         drug_info = drug_list  # 데이터프레임 변환이 필요없습니다.
#                     else:
#                         error_message = f"'{drug_name}'에 대한 유사한 의약품 정보를 찾을 수 없습니다."
#
#                 else:
#                     drug_info = drug_list  # 데이터프레임 변환이 필요없습니다.
#             else:
#                 # 응답 데이터 출력
#                 print(f"Response Status Code: {response.status_code}")
#                 print(f"Response Content: {response.content}")
#                 error_message = f"API 요청 실패: {response.status_code} - {response.text}"
#
#     return render(request, 'drug_detail.html', {'drug_info': drug_info, 'error_message': error_message})

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
        link = "https://www.kpanews.co.kr/article/" + item.a['href']
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

def news_summary_view(request, article_id):
    page = request.GET.get('page', 1)
    articles = get_articles(page)
    article = articles[article_id]

    article_response = requests.get(article['link'])
    article_response.encoding = 'utf-8'
    article_soup = BeautifulSoup(article_response.text, 'html.parser')

    article_content = article_soup.select_one('.view_con_t')
    if article_content:
        article_text = article_content.get_text(strip=True)
    else:
        article_text = article_soup.get_text(strip=True)

    max_length = 10000  # 길이 증가
    article_text = article_text[:max_length]

    try:
        # LangChain을 사용하여 요약 생성
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes Korean news articles."},
            {"role": "user", "content": f"다음 기사를 자세히 요약해주세요 (약 300-400자):\n\n{article_text}"}
        ]

        response = llm(messages=messages)
        full_summary = response.content.strip()
    except Exception as e:
        full_summary = f"요약 생성 중 오류 발생: {str(e)}"

    article['full_summary'] = full_summary
    return render(request, 'news_summary.html', {'article': article})



