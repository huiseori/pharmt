import requests
import openai
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict
import urllib.parse
import xml.etree.ElementTree as ET
from urllib.parse import quote


from pharmtracker import settings
openai.api_key= settings.OPENAI_API_KEY

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-3-small"
)
# # Chroma 클라이언트 생성
# client = chromadb.Client(Settings(persist_directory="./chroma_db"))
# collection = client.get_or_create_collection(name="drug_embeddings", embedding_function=openai_ef)
# def check_data_loaded():
#     return collection.count() > 0
# def update_drug_database():
#     # encoded_drug_name = urllib.parse.quote("")  # 모든 품목 가져오기
#     encoded_key = quote(settings.DUR_API_KEY)
#     dur_url = settings.DUR_API_URL
#     params = {'serviceKey': settings.DUR_API_KEY, 'pageNo': 1, 'numOfRows': 100, 'type': 'xml'}
#     # params = {'itemName': encoded_drug_name , 'pageNo': 1, 'numOfRows': 100, 'type': 'xml'}
#     drugs = fetch_drug_data_from_api(dur_url, params)
#     if drugs:
#         store_embeddings(drugs)
#         print(f"Updated database with {len(drugs)} drug entries")
#     else:
#         print("No drug data fetched from API")
#     # store_embeddings(drugs)
#
# def fetch_drug_data_from_api(dur_url: str, params: Dict) -> List[Dict]:
#     try:
#         logging.info(f"Fetching data from URL: {dur_url}")
#         logging.info(f"With params: {params}")
#         # response = requests.get(dur_url)
#         response = requests.get(dur_url, params=params)
#         response.raise_for_status()
#         # XML 파싱
#         root = ET.fromstring(response.content)
#         items = root.findall('.//item')
#
#         if not items:
#             logging.warning("No items found in the API response")
#             return []
#
#         return [
#             {child.tag: child.text for child in item}
#             for item in items
#         ]
#
#         # items = []
#         # for item in root.findall('.//item'):
#         #     drug_info = {}
#         #     for child in item:
#         #         drug_info[child.tag] = child.text
#         #     items.append(drug_info)
#         #
#         # if items:
#         #     return items
#         # else:
#         #     logging.warning("No items found in the API response")
#         #     return []
#     except requests.exceptions.RequestException as e:
#         logging.error(f"API request failed: {e}")
#         return []
#     except ET.ParseError as e:
#         logging.error(f"XML parsing error: {e}")
#         return []
#         # data = response.xml()
#         # processed_data = []
#         # for item in data['items']:
#         #     processed_item = {
#         #         'id': item['itemSeq'],
#         #         'item_name': item['itemName'].strip(),
#         #         'ingr_code': item['ingrCode'],
#         #         'ingr_kor_name': item['ingrKorName'].strip(),
#         #         'mix': item['mix'],
#         #         'mix_type': item['mixType'],
#         #         'se_qesitm': item['seQesitm'].strip(),
#         #         'entp_name': item['entpName'].strip(),
#         #     }
#     #     if 'body' in data and 'items' in data['body']:
#     #         return data['body']['items']
#     #     else:
#     #         logging.warning("No items found in the API response")
#     #         return []
#     # except requests.exceptions.RequestException as e:
#     #     logging.error(f"API request failed: {e}")
#     #     return []
#     # processed_data.append(processed_item)
#     # return processed_data
# # def get_embedding(text: str) -> List[float]:
# #     response = openai.Embedding.create(
# #         input=text,
# #         model="text-embedding-3-small"
# #     )
# #     return response['data'][0]['embedding']
# def store_embeddings(drugs: List[Dict]):
#     ids = []
#     documents = []
#     metadatas = []
#     for drug in drugs:
#         drug_id = drug.get('ITEM_SEQ', '')
#         item_name = drug.get('ITEM_NAME', '')
#         entp_name = drug.get('ENTP_NAME', '')
#         item_ingr_name = drug.get('ITEM_INGR_NAME', '')
#         # 모든 필드를 포함한 문서 생성
#         doc_text = f"{item_name} {entp_name} {item_ingr_name}"
#         ids.append(drug_id)
#         documents.append(doc_text)
#         metadatas.append(drug)  # 전체 drug 딕셔너리를 메타데이터로 저장
#         # ids.append(drug['id'])
#         # documents.append(f"{drug['item_name']} {drug['ingr_kor_name']} {drug['se_qesitm']}")
#         # metadatas.append({
#         #     'item_name': drug['item_name'],
#         #     'ingr_kor_name': drug['ingr_kor_name'],
#         #     'se_qesitm': drug['se_qesitm']
#         # })
#
#     collection.add(
#         ids=ids,
#         documents=documents,
#         metadatas=metadatas
#     )
#
# # def store_embeddings(drugs: List[Dict]):
# #     for drug in drugs:
# #         combined_text = f"{drug['item_name']} {drug['ingr_kor_name']} {drug['entp_name']} {drug['se_qesitm']}"
# #         embedding = get_embedding(combined_text)
# #
# #         collection.add(
# #             documents=[combined_text],
# #             metadatas=[drug],
# #             ids=[drug['id']],
# #             embeddings=[embedding]
# #         )
#
# import logging
# def query_similar_drugs(query: str, n_results: int = 3)-> List[Dict]:
#     try:
#         results = collection.query(
#             query_texts=[query],
#             n_results=n_results
#         )
#         logging.info(f"Query results: {results}")
#
#         similar_drugs = []
#         for metadata in results['metadatas'][0]:
#             drug_info = {
#                 'ITEM_NAME': metadata.get('ITEM_NAME', ''),
#                 'ENTP_NAME': metadata.get('ENTP_NAME', ''),
#                 'ITEM_INGR_NAME': metadata.get('ITEM_INGR_NAME', ''),
#                 'SPCLTY_PBLC': metadata.get('SPCLTY_PBLC', ''),
#                 'SE_QESITM': metadata.get('SE_QESITM', '')
#             }
#             similar_drugs.append(drug_info)
#
#         return similar_drugs
#     except Exception as e:
#         logging.error(f"Error in querying similar drugs: {e}")
#         return []
# # def query_similar_drugs(query: str, n_results: int = 3) -> List[Dict]:
# #     query_embedding = get_embedding(query)
# #     results = collection.query(
# #         query_embeddings=[query_embedding],
# #         n_results=n_results
# #     )
# #     return results['metadatas'][0]
#
#
