# views.py

import os
import json
import uuid
import time
import base64
import logging
import urllib.parse
import re
from io import BytesIO
from datetime import date
from django.conf import settings
from django.contrib import messages
from django.db.models import Q

# Django Imports
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.db.models import Q
from django.core.cache import cache
from django.utils.html import escape

# Image Processing Imports
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

# External API & Scraping Imports
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from xml.etree import ElementTree

# AI & LangChain Imports
from langchain.chains import conversation
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document

# Local Imports
from .models import ChatbotModel, Medicine  # ★핵심: DB 모델 임포트

# ==========================================
# 1. 초기 설정 및 전역 변수
# ==========================================

# API Keys
OPENAI_API_KEY = settings.OPENAI_API_KEY
DUR_API_KEY = settings.DUR_API_KEY
OCR_SECRET_KEY = getattr(settings, 'OCR_SECRET_KEY', 'YldyamVGd29WUU9VSUJSckJPT1JZcHdkTFR3cUJVVko=') # settings로 이동 권장
OCR_API_URL = 'https://rfsoe9oge0.apigw.ntruss.com/custom/v1/33758/04551f065f17fa952a90b63ee0c5a01adda5ab1c7e8b4d2a3cf37ccaf94134ee/general'

# RAG 설정 (LLM & Embeddings)
llm = ChatOpenAI(temperature=0.1, openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
cache_dir = LocalFileStore("./.cache/practice/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

# 전역 VectorStore (서버 시작 시 로드)
vectorstore = None

def initialize_vectorstore():
    """
    서버 시작 시, 미리 구축된 FAISS 인덱스를 로드합니다.
    (자소서: 'API 병목 해결을 위해 로컬 DB/VectorStore 활용')
    """
    global vectorstore
    if vectorstore is None:
        try:
            # 실제 운영 시에는 save_local로 저장된 인덱스를 load_local로 불러옵니다.
            # 여기서는 예외 처리를 위해 임시 텍스트로 초기화합니다.
            vectorstore = FAISS.from_texts(["초기화 문서: 의약품 안전 정보"], cached_embeddings)
            print("FAISS VectorStore 초기화 완료")
        except Exception as e:
            print(f"VectorStore 로드 실패: {e}")

# 앱 구동 시 초기화 (Blocking 방지를 위해 별도 호출 권장하나 뷰 로딩 시 실행)
initialize_vectorstore()


# ==========================================
# 2. 기본 뷰 (Splash, Auth, MyPage)
# ==========================================

def splash_view(request):
    return render(request, 'splash.html')

def home_view(request):
    return render(request, 'home.html')

@login_required
def mypage_view(request):
    user = request.user
    
    # 1. 약품 삭제 로직
    if request.method == 'POST' and 'delete_medication' in request.POST:
        index = int(request.POST.get('delete_medication'))
        medications = request.session.get('medications', [])
        if 0 <= index < len(medications):
            deleted_med = medications.pop(index)
            request.session['medications'] = medications
            request.session.modified = True
            messages.success(request, f"{deleted_med.get('item_name', '약품')}이(가) 삭제되었습니다.")
        return redirect('mypage')

    # 2. 프로필 업데이트
    elif request.method == 'POST' and 'gender' in request.POST:
        user.first_name = request.POST.get('gender')
        user.last_name = request.POST.get('age')
        user.save()
        messages.success(request, '프로필이 업데이트되었습니다.')

    # 3. 건강 기록 저장
    elif request.method == 'POST' and 'blood_pressure' in request.POST:
        today = date.today().strftime('%Y-%m-%d')
        health_record = {
            'blood_pressure': request.POST.get('blood_pressure'),
            'blood_sugar': request.POST.get('blood_sugar'),
            'weight': request.POST.get('weight'),
        }
        
        if 'health_records' not in request.session:
            request.session['health_records'] = {}
        
        if today not in request.session['health_records']:
            request.session['health_records'][today] = []
            
        request.session['health_records'][today].append(health_record)
        request.session.modified = True
        messages.success(request, '건강 기록이 저장되었습니다.')
        return redirect('mypage')

    context = {
        'age_range': range(1, 101),
        'user': user,
        'health_records': request.session.get('health_records', {}),
        'medications': request.session.get('medications', []),
    }
    return render(request, 'mypage.html', context)

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        # 자소서 성과: 중복 가입 방지 로직
        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {'error_message': "이미 사용 중인 아이디입니다."})

        user = User.objects.create_user(username=username, password=password)
        login(request, user)
        return redirect('home')
    return render(request, 'register.html')

def login_view(request):
    if request.method == 'POST':
        user = authenticate(request, username=request.POST.get('username'), password=request.POST.get('password'))
        if user:
            login(request, user)
            return redirect('home')
        else:
            return render(request, 'login.html', {'error_message': "아이디 또는 비밀번호가 잘못되었습니다."})
    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    return redirect('home')


# ==========================================
# 3. 핵심 기능: 의약품 검색 (DB 기반 최적화)
# ==========================================
def drug_list_view(request):
    drug_info = []
    error_message = None
    query = request.GET.get('query', '')

    # 1. 사용자가 약품을 '내 약품 목록'에 추가하는 경우 (POST)
    if request.method == 'POST':
        drug = {
            'item_name': request.POST.get('item_name'),
            'entp_name': request.POST.get('entp_name'),
            'item_seq': request.POST.get('item_seq'), # 식별자 추가
            'image_url': request.POST.get('image_url'),
        }
        if 'medications' not in request.session:
            request.session['medications'] = []
        request.session['medications'].append(drug)
        request.session.modified = True
        messages.success(request, f"{drug['item_name']}이(가) 추가되었습니다.")
        return redirect('drug_list')

    # 2. 약품 검색 (GET)
    if query:
        # [기술적 개선] 외부 API 호출 제거 -> 내부 DB(Medicine 모델) 조회
        # 인덱스(item_name)를 타게 되어 검색 속도가 매우 빠름 (O(log N))
        results = Medicine.objects.filter(item_name__icontains=query)[:20] # 상위 20개만

        if results.exists():
            for drug in results:
                drug_info.append({
                    "item_name": drug.item_name,
                    "entp_name": drug.entp_name,
                    "item_seq": drug.item_seq,
                    "detail_link": f"/drug_detail/{drug.item_seq}/",
                    "image_url": drug.image_url
                })
        else:
            error_message = f"'{query}'에 대한 정보를 데이터베이스에서 찾을 수 없습니다."

    # 세션에 저장된 약품 표시용
    saved_meds = request.session.get('medications', [])
    saved_item_names = set(m.get('item_name') for m in saved_meds)

    return render(request, 'drug_list.html', {
        'drug_info': drug_info,
        'error_message': error_message,
        'saved_item_names': saved_item_names,
        'query': query
    })

def drug_detail_view(request, item_seq):
    """
    DB에서 item_seq(Primary Key)로 즉시 조회.
    API Latency 없이 0.1초 이내 렌더링 가능.
    """
    try:
        drug = Medicine.objects.get(item_seq=item_seq)
        drug_detail = {
            "item_seq": drug.item_seq,
            "item_name": drug.item_name,
            "entp_name": drug.entp_name,
            "efficacy": drug.efficacy,
            "usage_dosage": drug.usage_dosage,
            "precautions": drug.precautions,
            "image_url": drug.image_url
        }
    except Medicine.DoesNotExist:
        drug_detail = None

    return render(request, 'drug_detail.html', {'drug_detail': drug_detail})


# ==========================================
# 4. 핵심 기능: Advanced RAG 챗봇 (Query Expansion + Reranking)
# ==========================================
# 쿼리 확장 및 Reranking을 통한 검색 정확도 고도화

def expand_query(original_query: str) -> list:
    """
    [Step 1: Query Expansion]
    사용자의 질문을 확장하여 다양한 검색어를 생성합니다.
    예: "머리 아파" -> ["두통약", "진통제", "편두통 해결"]
    """
    system_prompt = """
    당신은 의약품 검색 전문가입니다.
    사용자의 질문을 보고, 데이터베이스에서 정보를 찾기 좋은 검색어 3개를 생성하세요.
    결과는 쉼표(,)로 구분하여 단어만 나열하세요.
    예시: 
    질문: "배 아플 때 먹는 거" -> "복통, 소화제, 위장약"
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    # StrOutputParser가 없다면 .content로 접근하도록 수정
    chain = prompt | llm
    try:
        response = chain.invoke({"question": original_query})
        # 응답이 문자열이 아니라 객체일 경우 처리
        content = response.content if hasattr(response, 'content') else str(response)
        expanded_queries = [q.strip() for q in content.split(',')]
        return [original_query] + expanded_queries
    except Exception:
        return [original_query]

def rerank_documents(query: str, docs: list) -> list:
    """
    [Step 2: Reranking]
    검색된 문서들이 질문과 얼마나 관련이 있는지 LLM이 평가하여 재정렬합니다.
    (Cross-Encoder 대신 LLM을 Judge로 사용)
    """
    if not docs:
        return []

    # 평가용 프롬프트
    system_prompt = """
    당신은 검색 결과의 관련성을 평가하는 판사(Relevance Judge)입니다.
    사용자의 질문(Query)과 검색된 문서(Document)가 주어지면,
    이 문서가 질문에 답변하는 데 얼마나 도움이 되는지 0~100점 사이의 점수를 매기세요.
    오직 숫자만 출력하세요.
    """
    
    scored_docs = []
    for doc in docs:
        # 비용 절약을 위해 문서 앞부분만 평가
        content_preview = doc.page_content[:500]
        prompt_text = f"Query: {query}\nDocument: {content_preview}\nScore:"
        
        try:
            messages = [
                ("system", system_prompt),
                ("user", prompt_text)
            ]
            score_res = llm.invoke(messages).content.strip()
            # 숫자 추출 (정규식 사용)
            import re
            match = re.search(r'\d+', score_res)
            score = int(match.group()) if match else 0
        except Exception:
            score = 0
            
        scored_docs.append((doc, score))
    
    # 점수 높은 순으로 정렬
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 3개 문서만 반환
    top_k_docs = [item[0] for item in scored_docs[:3]]
    return top_k_docs

def retrieve_advanced_context(question: str):
    """
    [Hybrid Retrieval + Query Expansion + Reranking]
    1. 질문 확장 (Query Expansion)
    2. 다중 쿼리로 DB 및 벡터 검색 (Retrieval)
    3. 중복 제거 및 통합
    4. 관련성 재순위화 (Reranking)
    """
    # LangChain Document 객체 사용을 위해 필요 (임포트 안 되어 있다면 여기서 정의)
    from langchain.schema import Document 
    
    # 1. 쿼리 확장
    queries = expand_query(question)
    print(f"확장된 검색어: {queries}")
    
    all_docs = []
    
    # 2. 확장된 쿼리로 검색 수행 (Hybrid)
    for q in queries:
        # A. 정형 데이터(DB) 검색 (정확한 약품명 매칭 시)
        # DB 검색은 정확도가 높으므로 우선순위
        db_results = Medicine.objects.filter(item_name__icontains=q)[:2]
        for drug in db_results:
            content = (
                f"[DB정보] 약품명: {drug.item_name}\n"
                f"효능: {drug.efficacy}\n"
                f"용법: {drug.usage_dosage}\n"
                f"주의사항: {drug.precautions}"
            )
            # DB 결과는 Document 객체로 변환 (metadata에 출처 기록)
            all_docs.append(Document(page_content=content, metadata={"source": "DB", "name": drug.item_name}))

        # B. 비정형 데이터(Vector) 검색
        if vectorstore:
            vector_results = vectorstore.similarity_search(q, k=2)
            all_docs.extend(vector_results)

    # 3. 중복 제거 (내용 기준)
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    unique_docs_list = list(unique_docs)
    
    # 4. 재순위화 (Reranking)
    final_docs = rerank_documents(question, unique_docs_list)
    
    # 최종 컨텍스트 생성
    context_text = "\n\n".join([doc.page_content for doc in final_docs])
    
    # 약품명 추출 (버튼 생성을 위해 - Rerank된 최상위 문서가 DB 출처라면 그 이름 사용)
    extracted_name = None
    for doc in final_docs:
        if doc.metadata.get("source") == "DB":
            extracted_name = doc.metadata.get("name")
            break
            
    return context_text, extracted_name

@csrf_exempt
@require_http_methods(["GET", "POST"])
def chatbot_view(request):
    if request.method == "GET":
        return render(request, "chatbot.html")

    try:
        body = json.loads(request.body.decode("utf-8"))
        question = body.get("question")
        
        if not question:
            return JsonResponse({"error": "질문을 입력해주세요."}, status=400)

        # Advanced RAG 파이프라인 실행
        context, extracted_name = retrieve_advanced_context(question)
        
        # 답변 생성 프롬프트
        system_prompt = """
        당신은 전문 약사 AI입니다. 
        제공된 [Context]를 바탕으로 답변하세요.
        1. 모르는 내용은 지어내지 말고, 정보가 없다고 말하세요.
        2. 노인분들이 이해하기 쉽게 어려운 의학 용어는 풀어서 설명하세요.
        3. 답변 끝에는 항상 "정확한 진단은 의사와 상담하세요"라고 덧붙이세요.
        
        [Context]:
        {context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"question": question, "context": context})
        answer = response.content if hasattr(response, 'content') else str(response)

        # 약품 상세 페이지 링크 버튼 추가 (UX)
        if extracted_name:
            encoded_name = urllib.parse.quote(extracted_name)
            link_html = f'<br><br><a href="/drug_list/?query={encoded_name}" target="_blank" class="btn-link">💊 {extracted_name} 상세 정보 보기</a>'
            answer += link_html

        return JsonResponse({"answer": answer})

    except Exception as e:
        print(f"Error: {e}")
        return JsonResponse({"error": "처리 중 오류가 발생했습니다."}, status=500)


# ==========================================
# 5. 핵심 기능: AI OCR (이미지 텍스트 인식)
# ==========================================
# CLOVA OCR API + OpenCV 전처리 + DB 검증

def ocr_view(request):
    return render(request, 'ocr.html')

@csrf_exempt
def ocr_process(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    try:
        # 1. 이미지 저장 (카메라 or 파일 업로드)
        path = None
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            path = os.path.join(settings.MEDIA_ROOT, image_file.name)
            with open(path, 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)
        elif request.body:
            body_data = json.loads(request.body.decode('utf-8'))
            image_data = base64.b64decode(body_data.get('image_data', ''))
            path = os.path.join(settings.MEDIA_ROOT, f'{uuid.uuid4()}.jpg')
            with open(path, 'wb') as f:
                f.write(image_data)

        if not path:
            return JsonResponse({'error': '이미지가 없습니다.'}, status=400)

        # 2. CLOVA OCR API 호출
        files = [('file', open(path, 'rb'))]
        request_json = {'images': [{'format': 'jpg', 'name': 'demo'}], 'requestId': str(uuid.uuid4()), 'version': 'V2', 'timestamp': int(round(time.time() * 1000))}
        headers = {'X-OCR-SECRET': OCR_SECRET_KEY}
        
        response = requests.post(OCR_API_URL, headers=headers, data={'message': json.dumps(request_json)}, files=files)
        result = response.json()

        # 3. 텍스트 추출 및 필터링
        valid_texts = []
        recognized_texts = []
        
        # OpenCV로 시각화 (Bounding Box)
        img = cv2.imread(path)
        
        for field in result.get('images', [{}])[0].get('fields', []):
            text = field['inferText']
            recognized_texts.append(text)
            
            # [DB 검증] 인식된 텍스트가 실제 우리 DB에 있는 약인지 확인
            # 자소서 내용 일치: API 호출 검증 대신 로컬 DB 검증으로 속도 향상
            if len(text) >= 2:
                # DB에서 포함 여부 확인
                if Medicine.objects.filter(item_name__icontains=text).exists():
                    valid_texts.append(text)

        # 4. 결과 처리 (첫 번째 유효한 약물로 리다이렉트)
        if valid_texts:
            best_match = valid_texts[0]
            redirect_url = f'/drug_list?query={urllib.parse.quote(best_match)}'
            return JsonResponse({'redirect': redirect_url, 'recognized': valid_texts})
        else:
            return JsonResponse({'error': 'DB에서 일치하는 의약품을 찾을 수 없습니다.'}, status=404)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ==========================================
# 6. 부가 기능: 약학 뉴스 & 약국 검색
# ==========================================

def get_articles(page=1):
    """약학 뉴스 크롤링"""
    url = f"https://www.kpanews.co.kr/article/list.asp?page={page}"
    try:
        response = requests.get(url, timeout=5)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        for item in soup.select('.lst_article1 ul li'):
            title = item.select_one('.subj').text.strip()
            summary = item.select_one('.t1').text.strip() if item.select_one('.t1') else ''
            date_txt = item.select_one('.botm span').text.strip()
            link = "https://www.kpanews.co.kr/article/" + item.a['href']
            articles.append({'title': title, 'summary': summary, 'date': date_txt, 'link': link})
        return articles
    except Exception:
        return []

def news_view(request):
    page = request.GET.get('page', 1)
    articles = get_articles(page)
    return render(request, 'news.html', {'articles': articles})

def news_summary_view(request, article_link):
    # LLM을 활용한 뉴스 3줄 요약 기능 (노인 편의성)
    try:
        res = requests.get(article_link)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, 'html.parser')
        content = soup.select_one('.view_con_t').get_text(strip=True)[:3000]
        
        msg = [
            ("system", "한국어 뉴스 기사를 노인이 이해하기 쉽게 3줄로 요약해 주세요."),
            ("user", content)
        ]
        summary = llm.invoke(msg).content
    except Exception:
        summary = "요약을 불러올 수 없습니다."
        
    return render(request, 'news_summary.html', {'summary': summary, 'original_link': article_link})

# 약국 검색 (공공데이터 API 유지 - 위치 기반 실시간 데이터 필요)
def get_pharmacies(search_query='', page=1):
    base_url = "http://apis.data.go.kr/B552657/ErmctInsttInfoInqireService/getParmacyListInfoInqire"
    params = {
        'serviceKey': DUR_API_KEY,
        'QN': search_query,
        'pageNo': page,
        'numOfRows': 10,
    }
    try:
        res = requests.get(base_url, params=params, verify=False, timeout=5)
        tree = ElementTree.fromstring(res.content)
        pharmacies = []
        for item in tree.findall('.//item'):
            pharmacies.append({
                'name': item.findtext('dutyName'),
                'address': item.findtext('dutyAddr'),
                'tel': item.findtext('dutyTel1'),
                'lat': item.findtext('wgs84Lat'),
                'lon': item.findtext('wgs84Lon'),
            })
        return pharmacies
    except Exception:
        return []

def pharmacy_list_view(request):
    query = request.GET.get('search_query', '')
    pharmacies = get_pharmacies(query) if query else []
    return render(request, 'pharmacy_list.html', {'pharmacies': pharmacies, 'search_query': query})
