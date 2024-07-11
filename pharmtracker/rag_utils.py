import requests
import openai
import chromadb
from chromadb.config import Settings
from typing import List, Dict

from pharmtracker import settings

# Chroma 클라이언트 생성
client = chromadb.Client(Settings(allow_reset=True))
collection = client.create_collection("drug_embeddings")


def fetch_drug_data_from_api(dur_url: str, params: Dict) -> List[Dict]:
    response = requests.get(dur_url, params=params)
    data = response.json()

    processed_data = []
    for item in data['items']:
        processed_item = {
            'id': item['itemSeq'],
            'item_name': item['itemName'].strip(),
            'ingr_code': item['ingrCode'],
            'ingr_kor_name': item['ingrKorName'].strip(),
            'mix': item['mix'],
            'mix_type': item['mixType'],
            'se_qesitm': item['seQesitm'].strip(),
            'entp_name': item['entpName'].strip(),
            'detail_link': item['detailLink']  # 상세 정보 링크 추가
        }
        processed_data.append(processed_item)

    return processed_data


def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


def store_embeddings(drugs: List[Dict]):
    for drug in drugs:
        combined_text = f"{drug['item_name']} {drug['ingr_kor_name']} {drug['entp_name']} {drug['se_qesitm']}"
        embedding = get_embedding(combined_text)

        collection.add(
            documents=[combined_text],
            metadatas=[drug],
            ids=[drug['id']],
            embeddings=[embedding]
        )


def query_similar_drugs(query: str, n_results: int = 3) -> List[Dict]:
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results['metadatas'][0]


def update_drug_database():
    dur_url = settings.DUR_API_KEY
    params = {'key': 'API_KEY', 'pageNo': 1, 'numOfRows': 100}
    drugs = fetch_drug_data_from_api(dur_url, params)
    store_embeddings(drugs)