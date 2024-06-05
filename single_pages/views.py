# views.py
from django.shortcuts import render

from django.shortcuts import render
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
        roi_img = img.copy()

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

        # Save the processed image to serve it to the frontend
        processed_image_path = os.path.join('uploads', 'processed_' + image_file.name)
        cv2.imwrite(processed_image_path, roi_img)

        return render(request, 'ocr.html', {'processed_image_path': '/' + processed_image_path})

    return JsonResponse({'error': 'Invalid request'}, status=400)


